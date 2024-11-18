from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
import os
from petsc4py import PETSc
import ufl
from ufl import inner, jump, grad, div, avg

from utils.saver import ResultsSaver
from utils.estimation import marking

parent_dir = os.path.dirname(__file__)

def cprint(str2print, print_save=True, file=None):
    if print_save:
        print(str2print, file=file)

def poisson_dirichlet(max_it,
                      quadrature_degree=4,
                      print_save=False,
                      ref_method="background"):
    with XDMFFile(MPI.COMM_WORLD, "lshaped.xdmf", "r") as fi:
        mesh = fi.read_mesh(name="Grid")

    output_dir = os.path.join(parent_dir, "output", ref_method)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data = ["dofs", "H10 estimator"]
    if print_save:
        results_saver = ResultsSaver(output_dir, data)

    FE_degree = 1
    for i in range(max_it):
        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", print_save)
        CG1Element = element("CG", mesh.topology.cell_name(), 1)
        quadrature_degree = 2 * (FE_degree + 1)

        FE_space = dfx.fem.functionspace(mesh, CG1Element)
        num_dofs = FE_space.dofmap.index_map.size_global * FE_space.dofmap.index_map_bs

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Data interpolation.", print_save)
        f_h = dfx.fem.Function(FE_space)
        f_h.vector.set(1.)

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Variational formulation set up.", print_save)
        
        uD = dfx.fem.Function(FE_space)
        uD.vector.set(0.)

        cdim = mesh.topology.dim
        fdim = cdim - 1
        mesh.topology.create_connectivity(fdim, cdim)
        boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
        boundary_dofs = dfx.fem.locate_dofs_topological(FE_space, fdim, boundary_facets)
        bc = dfx.fem.dirichletbc(uD, boundary_dofs)

        u = ufl.TrialFunction(FE_space)
        v = ufl.TestFunction(FE_space)

        dx = ufl.Measure("dx",
                         domain=mesh,
                         metadata={"quadrature_degree": quadrature_degree})
        
        a = inner(grad(u), grad(v)) * dx
        rhs = inner(f_h, v) * dx
        
        bilinear_form = dfx.fem.form(a)
        linear_form = dfx.fem.form(rhs)

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Linear system assembly.", print_save)
        A = assemble_matrix(bilinear_form, bcs=[bc])
        A.assemble()
        b = assemble_vector(linear_form)

        dfx.fem.apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dfx.fem.set_bc(b, [bc])

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Solve.", print_save)
        uh = dfx.fem.Function(FE_space)

        # Parametrization of the PETSc solver
        options = PETSc.Options()
        options["ksp_type"] = "preonly"
        options["pc_type"] = "lu"
        petsc_solver = PETSc.KSP().create(mesh.comm)
        petsc_solver.setOperators(A)
        petsc_solver.setFromOptions()
        petsc_solver.solve(b, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", print_save)
        est_quadrature_degree_cells = max(0, FE_degree - 2)
        est_quadrature_degree_facets = max(0, FE_degree - 1)

        dx_est = ufl.Measure("dx",
                             domain=mesh,
                             metadata={"quadrature_degree": est_quadrature_degree_cells})

        dS_est = ufl.Measure("dS",
                             domain=mesh,
                             metadata={"quadrature_degree": est_quadrature_degree_facets})

        h_T = ufl.CellDiameter(mesh)
        h_E = ufl.FacetArea(mesh)
        n = ufl.FacetNormal(mesh)

        r = f_h + div(grad(uh))
        J_h = jump(grad(uh), -n)

        DG0Element = element("DG", mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(mesh, DG0Element)

        v0 = ufl.TestFunction(V0)

        # Interior residual
        eta_T = h_T**2 * inner(inner(r, r), v0) * dx_est

        # Facets residual
        eta_E = avg(h_E) * inner(inner(J_h, J_h), avg(v0)) * dS_est
        eta = eta_T + eta_E
        eta_form = dfx.fem.form(eta)

        eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
        eta_h = dfx.fem.Function(V0)
        eta_h.vector.setArray(eta_vec.array[:])
        h10_est = np.sqrt(sum(eta_h.x.array[:]))

        cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Ouput save.", print_save)
        if print_save:
            results_saver.save_function(uh, f"uh_{str(i).zfill(2)}")
            results_saver.save_function(eta_h, f"eta_h_{str(i).zfill(2)}")
            results_saver.save_values([num_dofs, h10_est], prnt=True)

        # Marking
        if i < max_it - 1:
            cprint(f"Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh refinement.", print_save)

            # Uniform refinement
            if ref_method == "omega_h":
                mesh = dfx.mesh.refine(mesh)

            # Adaptive refinement
            if ref_method == "adaptive":
                facets2ref = marking(eta_h)
                mesh = dfx.mesh.refine(mesh, facets2ref)

        cprint("\n", print_save)

if __name__=="__main__":
    poisson_dirichlet(25,
                      print_save=True,
                      ref_method="adaptive")
    poisson_dirichlet(8,
                      print_save=True,
                      ref_method="omega_h")