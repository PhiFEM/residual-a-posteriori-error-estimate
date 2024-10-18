import dolfinx as dfx
from dolfinx.fem import assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, set_bc
from dolfinx.io import XDMFFile, gmshio
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
from petsc4py import PETSc
import ufl
import basix as bx
from basix.ufl import element
from basix import CellType, ElementFamily
from utils.negative_laplacian import negative_laplacian

output_dir = "output"
ref_max = 5

if not os.path.isdir(output_dir):
	print(f"{output_dir} directory not found, we create it.")
	os.mkdir(os.path.join(".", output_dir))

def project(func, esp, meas):
    # Definition variational formulation
    u = ufl.TrialFunction(esp)
    v = ufl.TestFunction(esp)
    a = ufl.inner(u, v) * meas
    L = ufl.inner(func, v) * meas

    proj = dfx.fem.Function(esp)
    proj.vector.set(0.0)

    # Linear system assembly
    a_form = dfx.fem.form(a)
    b_form = dfx.fem.form(L)
    A = assemble_matrix(a_form)
    A.assemble()
    b = assemble_vector(b_form)

    # Parametrization of the PETSc solver
    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-7
    options["pc_hypre_type"] = "boomeramg"
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()

    # Solve
    solver.solve(b, proj.vector)
    return proj

"""
Read mesh
"""
mesh, cell_tags, facet_tags = gmshio.read_from_msh("./square.msh", MPI.COMM_WORLD, gdim=2)

results = {"dofs": [], "H10 error": [], "L2 error": []}
for i in range(ref_max):
    with XDMFFile(MPI.COMM_WORLD, os.path.join(".", output_dir, f"mesh_{str(i).zfill(2)}.xdmf"), "w") as of:
        of.write_mesh(mesh)
    """
    FE problem set up
    """
    CG1element = element("CG", mesh.topology.cell_name(), 1)
    DG0Element = element("DG", mesh.topology.cell_name(), 0)
    CGFElement = element("CG", mesh.topology.cell_name(), 5)
    V  = dfx.fem.functionspace(mesh, CG1element)
    V0 = dfx.fem.functionspace(mesh, DG0Element)
    Vf = dfx.fem.functionspace(mesh, CGFElement)

    results["dofs"].append(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 2})

    # Definition exact solution
    class exact_solution():
        def __init__(self, x0, y0):
            self.x0 = x0
            self.y0 = y0
        
        def __call__(self, x, y):
            return jnp.exp(-(((x-self.x0)*(x-self.x0) + (y-self.y0)*(y-self.y0))/2.))*jnp.sin(2.*x)*jnp.sin(2.*y)
        
        def negative_laplacian(self):
            func = lambda x, y: self.__call__(x, y) # Dirty workaround because negative_laplacian looks for the number of arguments in order to determine the dimension and "self" messes up the count.
            nlap = negative_laplacian(func)
            return nlap
        
        def dolfin_call(self):
            return lambda x: self(x[0], x[1])

    # Instantiation of the analytical solution
    u_exact = exact_solution(0., 0.)
    # Analytical solution call compatible with dolfinx
    df_u_exact = u_exact.dolfin_call()
    # Computation of the right-hand side with Jax <3
    f = u_exact.negative_laplacian()
    f_V = dfx.fem.Function(V)
    f_V.interpolate(f)
    f_Vf = dfx.fem.Function(Vf)
    f_Vf.interpolate(f)
    with XDMFFile(MPI.COMM_WORLD, os.path.join(".", output_dir, f"f_{str(i).zfill(2)}.xdmf"), "w") as of:
        of.write_mesh(mesh)
        of.write_function(f_V)

    # Definition variational formulation
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f_Vf, v) * dx

    uh = dfx.fem.Function(V)
    uh.vector.set(0.0)

    # Definition boundary conditions (homogeneous Dirichlet)
    u0 = dfx.fem.Function(V)
    u0.vector.set(0.0)

    facets = dfx.mesh.locate_entities_boundary(
                mesh,
                1,
                lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = dfx.fem.locate_dofs_topological(V, 1, facets)
    bcs = [dfx.fem.dirichletbc(u0, dofs)]

    # Linear system assembly
    a_form = dfx.fem.form(a)
    b_form = dfx.fem.form(L)
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = assemble_vector(b_form)
    # dfx.fem.apply_lifting(b, [a_form], [bcs])
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Parametrization of the PETSc solver
    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-7
    options["pc_hypre_type"] = "boomeramg"
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()

    # Solve
    solver.solve(b, uh.vector)
    # u_param.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
    #                            mode=PETSc.ScatterMode.FORWARD)
    with XDMFFile(mesh.comm, f"output/uh_{str(i).zfill(2)}.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(uh)

    """
    Error study
    """
    u_exact_Vf = dfx.fem.Function(Vf)
    u_exact_Vf.interpolate(df_u_exact)
    u_exact_V = dfx.fem.Function(V)
    u_exact_V.interpolate(df_u_exact)

    with XDMFFile(mesh.comm, "output/u_exact.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(u_exact_V)
    
    uh_Vf = dfx.fem.Function(Vf)
    uh_Vf.interpolate(uh)

    e = uh_Vf - u_exact_Vf
    L_H10 = ufl.inner(ufl.grad(e), ufl.grad(e)) * dx
    L_H10_form = dfx.fem.form(L_H10)
    results["H10 error"].append(np.sqrt(assemble_scalar(L_H10_form)))

    L_L2 = ufl.inner(e,e) * dx
    L_L2_form = dfx.fem.form(L_L2)
    results["L2 error"].append(np.sqrt(assemble_scalar(L_L2_form)))

    #v0 = ufl.TestFunction(V0)
    #L_H10err = ufl.inner(grad_e2, v0) * dx
    #L_H10err_form = dfx.fem.form(L_H10err)
    #b_H10err = assemble_vector(L_H10err_form)

    #H10error = dfx.fem.Function(V0)
    #H10error.vector.array = b_H10err.array

    # with XDMFFile(mesh.comm, f"output/H10error_{str(i).zfill(2)}.xdmf", "w") as of:
    #     of.write_mesh(mesh)
    #     of.write_function(H10error)

    df = pd.DataFrame(results)
    df.to_csv("results.csv")
    print(f"Step {str(i).zfill(2)}")
    print(df)
    print("\n")

    mesh = dfx.mesh.refine(mesh)