import basix as bx
from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem import assemble_scalar, assemble_vector
from dolfinx.io import XDMFFile, gmshio
import jax
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
from petsc4py import PETSc
import ufl
from ufl import inner, jump, grad, div, avg

from utils.compute_meshtags import tag_entities
from utils.classes import Levelset, ExactSolution, PhiFEMSolver
from utils.mesh_scripts import compute_facets_to_refine

output_dir = "output"
N = 20
max_it = 8
quadrature_degree = 4
sigma_D = 1.
tilt_angle = np.pi/6.

def rotation(angle, x):
    return (jnp.cos(angle)*x[0] + jnp.sin(angle)*x[1], -jnp.sin(angle)*x[0] + jnp.cos(angle)*x[1])

# Defines a tilted square
def expression_levelset(x, y):
    return np.exp(-1./(np.abs(rotation(tilt_angle - np.pi/4., [x, y])[0])+np.abs(rotation(tilt_angle - np.pi/4., [x, y])[1]))) - np.exp(-0.75)

def u_exact(x):
    return np.sin(2. * np.pi * rotation(tilt_angle, x)[0]) * np.sin(2. * np.pi * rotation(tilt_angle, x)[1])

u_exact = ExactSolution(lambda x, y: jnp.sin(np.pi * rotation(tilt_angle, (x,y))[0]) *
                                     jnp.sin(np.pi * rotation(tilt_angle, (x,y))[1]))
f = u_exact.negative_laplacian()

if not os.path.isdir(output_dir):
	print(f"{output_dir} directory not found, we create it.")
	os.mkdir(os.path.join(".", output_dir))

phi = Levelset(expression_levelset)

"""
Read mesh
"""
print("Create mesh")
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

with XDMFFile(mesh.comm, "./square.xdmf", "w") as of:
    of.write_mesh(mesh)

# Shift and scale the mesh
mesh.geometry.x[:, 0] -= 0.5
mesh.geometry.x[:, 1] -= 0.5
mesh.geometry.x[:, 0] *= 4.
mesh.geometry.x[:, 1] *= 4.

results = {"dofs": [], "H10 error": [], "L2 error": []}
for i in range(max_it):
    print(f"Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation")
    CG1Element = element("CG", mesh.topology.cell_name(), 1)
    CG2Element = element("CG", mesh.topology.cell_name(), 2)
    V = dfx.fem.functionspace(mesh, CG1Element)
    Vf = dfx.fem.functionspace(mesh, CG2Element)
    phiV = dfx.fem.Function(V)
    phiV.interpolate(phi.dolfinx_call)
    extV = dfx.fem.Function(V)
    extV.interpolate(phi.exterior(0.))

    # TODO: change the way dofs are counted
    results["dofs"].append(sum(np.where(phiV.x.array < 0., 1, 0)))
    fV = dfx.fem.Function(V)
    fV.interpolate(f)

    wh = dfx.fem.Function(V)
    wh.vector.set(0.0)

    # Parametrization of the PETSc solver
    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-7
    options["pc_hypre_type"] = "boomeramg"
    petsc_solver = PETSc.KSP().create(MPI.COMM_WORLD)
    petsc_solver.setFromOptions()

    phiFEM_solver = PhiFEMSolver(petsc_solver, i)
    print(f"Iteration n° {str(i).zfill(2)}: Data interpolation.")
    phiFEM_solver.set_data(fV, phi)
    print(f"Iteration n° {str(i).zfill(2)}: Mesh tags computation.")
    phiFEM_solver.compute_tags(mesh)
    print(f"Iteration n° {str(i).zfill(2)}: Variational formulation set up.")
    v0 = phiFEM_solver.set_variational_formulation(V)
    print(f"Iteration n° {str(i).zfill(2)}: Linear system assembly.")
    phiFEM_solver.assembly()
    print(f"Iteration n° {str(i).zfill(2)}: Solve.")
    phiFEM_solver.solve(wh)

    uh = dfx.fem.Function(V)
    uh.x.array[:] = wh.x.array * phiV.x.array

    print(f"Iteration n° {str(i).zfill(2)}: Ouput save and error computation.")
    u_exact_V = dfx.fem.Function(V)
    u_exact_V.interpolate(u_exact.dolfinx_call)

    CG2Element = element("CG", mesh.topology.cell_name(), 2)
    V2 = dfx.fem.functionspace(mesh, CG2Element)

    u_exact_V2 = dfx.fem.Function(V2)
    u_exact_V2.interpolate(u_exact.dolfinx_call)
    uh_V2 = dfx.fem.Function(V2)
    uh_V2.interpolate(uh)
    e_V2 = dfx.fem.Function(V2)
    e_V2.x.array[:] = u_exact_V2.x.array - uh_V2.x.array

    dx2 = ufl.Measure("dx",
                      domain=mesh,
                      subdomain_data=phiFEM_solver.cells_tags,
                      metadata={"quadrature_degree": 6})

    H10_norm = inner(grad(e_V2), grad(e_V2)) * v0 * dx2(1) + inner(grad(e_V2), grad(e_V2)) * v0 * dx2(2)
    H10_form = dfx.fem.form(H10_norm)
    results["H10 error"].append(np.sqrt(assemble_scalar(H10_form)))
    L2_norm = inner(e_V2, e_V2) * v0 * dx2(1) + inner(e_V2, e_V2) * v0 * dx2(2)
    L2_form = dfx.fem.form(L2_norm)
    results["L2 error"].append(np.sqrt(assemble_scalar(L2_form)))

    DG0Element = element("DG", mesh.topology.cell_name(), 0)
    V0 = dfx.fem.functionspace(mesh, DG0Element)

    w0 = ufl.TrialFunction(V0)
    L2_error_0 = dfx.fem.Function(V0)
    H10_error_0 = dfx.fem.Function(V0)

    L2_norm_local = inner(inner(e_V2, e_V2), w0) * v0 * dx2(1) + inner(inner(e_V2, e_V2), w0) * v0 * dx2(2)
    L2_norm_local_form = dfx.fem.form(L2_norm_local)
    L2_error_0.x.array[:] = assemble_vector(L2_norm_local_form).array
    with XDMFFile(mesh.comm, f"output/L2_error_{str(i).zfill(2)}.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(L2_error_0)

    H10_norm_local = inner(inner(grad(e_V2), grad(e_V2)), w0) * v0 * dx2(1) + inner(inner(grad(e_V2), grad(e_V2)), w0) * v0 * dx2(2)
    H10_norm_local_form = dfx.fem.form(H10_norm_local)
    H10_error_0.x.array[:] = assemble_vector(H10_norm_local_form).array
    with XDMFFile(mesh.comm, f"output/H10_error_{str(i).zfill(2)}.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(H10_error_0)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("./output/results.csv")
    with XDMFFile(mesh.comm, f"output/uh_{str(i).zfill(2)}.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(uh)

    with XDMFFile(mesh.comm, f"output/u_exact_V_{str(i).zfill(2)}.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(u_exact_V)

    with XDMFFile(mesh.comm, f"output/phi_V_{str(i).zfill(2)}.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(phiV)

    with XDMFFile(mesh.comm, f"output/exterior_{str(i).zfill(2)}.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(extV)

    if i < max_it:
        facets_tags = phiFEM_solver.facets_tags
        facets_to_refine = compute_facets_to_refine(mesh, facets_tags)
        print(f"Iteration n° {str(i).zfill(2)}: Mesh refinement.")
        mesh = dfx.mesh.refine(mesh, facets_to_refine)
        # mesh = dfx.mesh.refine(mesh)
    print("\n")