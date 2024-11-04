from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem import assemble_scalar, assemble_vector
from dolfinx.io import XDMFFile
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import os
from petsc4py import PETSc
import ufl
from ufl import inner, grad
from utils.classes import Levelset, ExactSolution, PhiFEMSolver, ResultsSaver
from utils.mesh_scripts import compute_facets_to_refine

def cprint(str2print, print_save=True, file=None):
    if print_save:
        print(str2print, file=file)

def poisson_dirichlet(N,
                      max_it,
                      quadrature_degree=4,
                      sigma_D=1.,
                      print_save=False):
    parent_dir = os.path.dirname(__file__)
    output_dir = os.path.join(parent_dir, "output")

    tilt_angle = np.pi/6.

    def rotation(angle, x):
        R = jnp.array([[ jnp.cos(angle), jnp.sin(angle)],
                       [-jnp.sin(angle), jnp.cos(angle)]])
        return R.dot(jnp.asarray(x))

    # Defines a tilted square
    def expression_levelset(x, y):
        fct = lambda x, y: jnp.exp(-1./(jnp.abs(rotation(tilt_angle - jnp.pi/4., [x, y])[0])+jnp.abs(rotation(tilt_angle - jnp.pi/4., [x, y])[1])))

        shift = fct(np.cos(tilt_angle)/2., np.sin(tilt_angle)/2.)

        return fct(x, y) - shift

    def expression_u_exact(x, y):
        return jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[0]) * jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[1])

    phi = Levelset(expression_levelset)
    u_exact = ExactSolution(expression_u_exact)
    f = u_exact.negative_laplacian()

    """
    Read mesh
    """
    mesh_path_xdmf = os.path.join(parent_dir, "square.xdmf")
    mesh_path_h5   = os.path.join(parent_dir, "square.h5")

    if (not os.path.isfile(mesh_path_h5)) or (not os.path.isfile(mesh_path_xdmf)):
        cprint(f"{mesh_path_h5} or {mesh_path_xdmf} not found, we create them.", print_save)
        mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

        # Shift and scale the mesh
        mesh.geometry.x[:, 0] -= 0.5
        mesh.geometry.x[:, 1] -= 0.5
        mesh.geometry.x[:, 0] *= 1.5
        mesh.geometry.x[:, 1] *= 1.5

        with XDMFFile(mesh.comm, "./square.xdmf", "w") as of:
            of.write_mesh(mesh)
    else:
        with XDMFFile(MPI.COMM_WORLD, "./square.xdmf", "r") as fi:
            mesh = fi.read_mesh()

    data = ["dofs", "H10 error", "L2 error"]
    if print_save:
        results_saver = ResultsSaver(output_dir, data)

    for i in range(max_it):
        cprint(f"Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", print_save)
        CG1Element = element("CG", mesh.topology.cell_name(), 1)
        CG2Element = element("CG", mesh.topology.cell_name(), 2)
        V = dfx.fem.functionspace(mesh, CG1Element)
        Vf = dfx.fem.functionspace(mesh, CG2Element)
        phiV = dfx.fem.Function(V)
        phiV.interpolate(phi.dolfinx_call)
        extV = dfx.fem.Function(V)
        extV.interpolate(phi.exterior(0.))

        # TODO: change the way dofs are counted
        num_dofs = sum(np.where(phiV.x.array < 0., 1, 0))
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
        petsc_solver = PETSc.KSP().create(mesh.comm)
        petsc_solver.setFromOptions()

        phiFEM_solver = PhiFEMSolver(petsc_solver, i)
        cprint(f"Iteration n° {str(i).zfill(2)}: Data interpolation.", print_save)
        phiFEM_solver.set_data(fV, phi)
        cprint(f"Iteration n° {str(i).zfill(2)}: Mesh tags computation.", print_save)
        phiFEM_solver.compute_tags(mesh)
        cprint(f"Iteration n° {str(i).zfill(2)}: Variational formulation set up.", print_save)
        v0 = phiFEM_solver.set_variational_formulation(V)
        cprint(f"Iteration n° {str(i).zfill(2)}: Linear system assembly.", print_save)
        phiFEM_solver.assemble()
        cprint(f"Iteration n° {str(i).zfill(2)}: Solve.", print_save)
        phiFEM_solver.solve(wh)

        uh = dfx.fem.Function(V)
        uh.x.array[:] = wh.x.array * phiV.x.array

        cprint(f"Iteration n° {str(i).zfill(2)}: Ouput save and error computation.", print_save)
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
        h10_error = np.sqrt(assemble_scalar(H10_form))
        L2_norm = inner(e_V2, e_V2) * v0 * dx2(1) + inner(e_V2, e_V2) * v0 * dx2(2)
        L2_form = dfx.fem.form(L2_norm)
        l2_error = np.sqrt(assemble_scalar(L2_form))

        DG0Element = element("DG", mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(mesh, DG0Element)

        w0 = ufl.TrialFunction(V0)
        L2_error_0 = dfx.fem.Function(V0)
        H10_error_0 = dfx.fem.Function(V0)

        L2_norm_local = inner(inner(e_V2, e_V2), w0) * v0 * dx2(1) + inner(inner(e_V2, e_V2), w0) * v0 * dx2(2)
        L2_norm_local_form = dfx.fem.form(L2_norm_local)
        L2_error_0.x.array[:] = assemble_vector(L2_norm_local_form).array

        H10_norm_local = inner(inner(grad(e_V2), grad(e_V2)), w0) * v0 * dx2(1) + inner(inner(grad(e_V2), grad(e_V2)), w0) * v0 * dx2(2)
        H10_norm_local_form = dfx.fem.form(H10_norm_local)
        H10_error_0.x.array[:] = assemble_vector(H10_norm_local_form).array

        # Save results
        if print_save:
            results_saver.save_function(L2_error_0,  f"L2_error_{str(i).zfill(2)}")
            results_saver.save_function(H10_error_0, f"H10_error_{str(i).zfill(2)}")
            results_saver.save_function(u_exact_V,   f"u_exact_V_{str(i).zfill(2)}")
            results_saver.save_function(phiV,        f"phi_V_{str(i).zfill(2)}")
            results_saver.save_values([num_dofs, h10_error, l2_error], prnt=True)

        if i < max_it - 1:
            facets_tags = phiFEM_solver.facets_tags
            facets_to_refine = compute_facets_to_refine(mesh, facets_tags)
            cprint(f"Iteration n° {str(i).zfill(2)}: Mesh refinement.", print_save)
            mesh = dfx.mesh.refine(mesh, facets_to_refine)
            # mesh = dfx.mesh.refine(mesh)
        cprint("\n", print_save)
        
if __name__=="__main__":
    # Size of initial background mesh (NxN cells)
    N = 10
    # Maximum number of iterations in refinement loop
    max_it = 7
    poisson_dirichlet(N, max_it, print_save=True)