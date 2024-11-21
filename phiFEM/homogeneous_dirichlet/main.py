from basix.ufl import element
import dolfinx as dfx
from dolfinx.io import XDMFFile
import jax.numpy as jnp
from mpi4py import MPI
import numpy as np
import os
from petsc4py import PETSc
from utils.solver import PhiFEMSolver, FEMSolver
from utils.continuous_functions import Levelset, ExactSolution
from utils.saver import ResultsSaver
from utils.mesh_scripts import mesh2d_from_levelset

parent_dir = os.path.dirname(__file__)

def definition_continuous_functions():
    tilt_angle = np.pi/6.

    def rotation(angle, x):
        R = jnp.array([[ jnp.cos(angle), jnp.sin(angle)],
                       [-jnp.sin(angle), jnp.cos(angle)]])
        return R.dot(jnp.asarray(x))

    # Defines a tilted square
    def expression_levelset(x, y):
        def fct(x, y):
            return jnp.sum(jnp.abs(rotation(tilt_angle - jnp.pi/4., [x, y])), axis=0)
        return fct(x, y) - np.sqrt(2.)/2.

    def expression_u_exact(x, y):
        return jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[0]) * jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[1])

    return expression_levelset, expression_u_exact

def cprint(str2print, print_save=True, file=None):
    if print_save:
        print(str2print, file=file)

def poisson_dirichlet_phiFEM(cl,
                             max_it,
                             quadrature_degree=4,
                             sigma_D=1.,
                             print_save=False,
                             ref_method="omega_h",
                             compute_exact_error=False):
    output_dir = os.path.join(parent_dir, "output_phiFEM", ref_method)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    expression_levelset, expression_u_exact = definition_continuous_functions()
    phi = Levelset(expression_levelset)
    u_exact = ExactSolution(expression_u_exact)
    u_exact.compute_negative_laplacian()
    f = u_exact.nlap

    """
    Create initial mesh
    """
    cprint(f"Create initial mesh.", print_save)
    scaling_factor = 2.

    N = int(scaling_factor * np.sqrt(2.)/cl)
    bg_mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    # Shift and scale the mesh
    bg_mesh.geometry.x[:, 0] -= 0.5
    bg_mesh.geometry.x[:, 1] -= 0.5
    bg_mesh.geometry.x[:, 0] *= scaling_factor
    bg_mesh.geometry.x[:, 1] *= scaling_factor

    with XDMFFile(bg_mesh.comm, "./square.xdmf", "w") as of:
        of.write_mesh(bg_mesh)

    data = ["dofs", "H10 estimator"]

    if print_save:
        results_saver = ResultsSaver(output_dir, data)

    working_mesh = bg_mesh
    for i in range(max_it):
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", print_save)
        CG1Element = element("CG", working_mesh.topology.cell_name(), 1)

        # Parametrization of the PETSc solver
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = PETSc.KSP().create(working_mesh.comm)
        petsc_solver.setFromOptions()

        phiFEM_solver = PhiFEMSolver(working_mesh, CG1Element, petsc_solver, num_step=i)
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Data interpolation.", print_save)
        phiFEM_solver.set_data(f, phi)
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh tags computation.", print_save)
        phiFEM_solver.compute_tags(padding=True)
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Variational formulation set up.", print_save)
        v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation()
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Linear system assembly.", print_save)
        phiFEM_solver.assemble()
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Solve.", print_save)
        phiFEM_solver.solve()
        uh = phiFEM_solver.solution

        working_mesh = phiFEM_solver.submesh

        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", print_save)
        V = dfx.fem.functionspace(working_mesh, CG1Element)
        phiV = phi.interpolate(V)
        phiFEM_solver.estimate_residual()
        eta_h = phiFEM_solver.eta_h
        h10_est = np.sqrt(sum(eta_h.x.array[:]))

        # Save results
        if print_save:
            results_saver.save_function(eta_h, f"eta_h_{str(i).zfill(2)}")
            results_saver.save_function(phiV,  f"phi_V_{str(i).zfill(2)}")
            results_saver.save_function(v0,    f"v0_{str(i).zfill(2)}")
            results_saver.save_function(uh,    f"uh_{str(i).zfill(2)}")

            results_saver.save_values([num_dofs,
                                       h10_est],
                                       prnt=True)

        # Marking
        if i < max_it - 1:
            cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh refinement.", print_save)

            # Uniform refinement (Omega_h only)
            if ref_method == "omega_h":
                working_mesh = dfx.mesh.refine(working_mesh)

            # Adaptive refinement
            if ref_method == "adaptive":
                facets2ref = phiFEM_solver.marking()
                working_mesh = dfx.mesh.refine(working_mesh, facets2ref)

        cprint("\n", print_save)

if __name__=="__main__":
    poisson_dirichlet_phiFEM(0.2,
                             25,
                             print_save=True,
                             ref_method="adaptive",
                             compute_exact_error=True)
    poisson_dirichlet_phiFEM(0.2,
                             9,
                             print_save=True,
                             ref_method="omega_h")