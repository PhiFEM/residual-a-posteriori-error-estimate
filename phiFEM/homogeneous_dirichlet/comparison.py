from   basix.ufl import element
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   dolfinx.fem.petsc import assemble_vector
import jax.numpy as jnp
from   mpi4py import MPI
import numpy as np
import os
from   petsc4py import PETSc
import ufl
from   ufl import inner, grad

from utils.solver import FEMSolver
from utils.continuous_functions import Levelset, ExactSolution
from utils.saver import ResultsSaver
from utils.mesh_scripts import mesh2d_from_levelset

from main import definition_continuous_functions

parent_dir = os.path.dirname(__file__)

def cprint(str2print, print_save=True, file=None):
    if print_save:
        print(str2print, file=file)

def poisson_dirichlet_FEM(cl,
                          max_it,
                          quadrature_degree=4,
                          print_save=False,
                          ref_method="omega_h",
                          compute_exact_error=False):
    output_dir = os.path.join(parent_dir, "output_FEM", ref_method)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    expression_levelset, expression_u_exact = definition_continuous_functions()
    phi = Levelset(expression_levelset)
    u_exact = ExactSolution(expression_u_exact)
    u_exact.compute_negative_laplacian()
    f = u_exact.nlap

    """
    Generate conforming mesh
    """
    tilt_angle = np.pi/6.

    def rotation(angle, x):
        R = jnp.array([[ jnp.cos(angle), jnp.sin(angle)],
                       [-jnp.sin(angle), jnp.cos(angle)]])
        return R.dot(jnp.asarray(x))

    point_1 = rotation(- tilt_angle - np.pi/4.,
                        np.array([0., -1.]) * np.sqrt(2.)/2.)
    point_2 = rotation(- tilt_angle - np.pi/4.,
                        np.array([1.,  0.]) * np.sqrt(2.)/2.)
    point_3 = rotation(- tilt_angle - np.pi/4.,
                        np.array([0.,  1.]) * np.sqrt(2.)/2.)
    point_4 = rotation(- tilt_angle - np.pi/4.,
                        np.array([-1., 0.]) * np.sqrt(2.)/2.)

    geom_vertices = np.vstack([point_1, point_2, point_3, point_4])
    _ = mesh2d_from_levelset(cl,
                             phi,
                             geom_vertices=geom_vertices,
                             output_dir=output_dir)
    
    with XDMFFile(MPI.COMM_WORLD, os.path.join(output_dir, "conforming_mesh.xdmf"), "r") as fi:
        conforming_mesh = fi.read_mesh(name="Grid")

    if compute_exact_error:
        data = ["dofs", "H10 estimator", "L2 error", "H10 error"]
    else:
        data = ["dofs", "H10 estimator"]

    if print_save:
        results_saver = ResultsSaver(output_dir, data)

    for i in range(max_it):
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", print_save)
        CG1Element = element("CG", conforming_mesh.topology.cell_name(), 1)

        # Parametrization of the PETSc solver
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = PETSc.KSP().create(conforming_mesh.comm)
        petsc_solver.setFromOptions()

        FEM_solver = FEMSolver(conforming_mesh, CG1Element, petsc_solver, num_step=i)
        
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Data interpolation.", print_save)
        dbc = dfx.fem.Function(FEM_solver.FE_space)
        facets = dfx.mesh.locate_entities_boundary(
                                conforming_mesh,
                                1,
                                lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = dfx.fem.locate_dofs_topological(FEM_solver.FE_space, 1, facets)
        bcs = [dfx.fem.dirichletbc(dbc, dofs)]
        FEM_solver.set_data(f, bcs=bcs)
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Variational formulation set up.", print_save)
        num_dofs = FEM_solver.set_variational_formulation()
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Linear system assembly.", print_save)
        FEM_solver.assemble()
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Solve.", print_save)
        FEM_solver.solve()
        uh = FEM_solver.solution

        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", print_save)
        FEM_solver.estimate_residual()
        eta_h = FEM_solver.eta_h
        h10_est = np.sqrt(sum(eta_h.x.array[:]))

        if compute_exact_error:
            cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: compute exact errors.", print_save)
            ref_degree = 2

            CGfElement = element("CG", conforming_mesh.topology.cell_name(), ref_degree)
            reference_space = dfx.fem.functionspace(conforming_mesh, CGfElement)

            uh_ref = dfx.fem.Function(reference_space)
            uh_ref.interpolate(uh)
            u_exact_ref = u_exact.interpolate(reference_space)
            e_ref = dfx.fem.Function(reference_space)
            e_ref.x.array[:] = u_exact_ref.x.array - uh_ref.x.array

            dx2 = ufl.Measure("dx",
                            domain=conforming_mesh,
                            metadata={"quadrature_degree": 2 * (ref_degree + 1)})

            DG0Element = element("DG", conforming_mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(conforming_mesh, DG0Element)
            w0 = ufl.TrialFunction(V0)

            L2_norm_local = inner(inner(e_ref, e_ref), w0) * dx2
            H10_norm_local = inner(inner(grad(e_ref), grad(e_ref)), w0) * dx2

            L2_error_0 = dfx.fem.Function(V0)
            H10_error_0 = dfx.fem.Function(V0)

            L2_norm_local_form = dfx.fem.form(L2_norm_local)
            L2_error_0.x.array[:] = assemble_vector(L2_norm_local_form).array
            L2_error = np.sqrt(sum(L2_error_0.x.array[:]))

            H10_norm_local_form = dfx.fem.form(H10_norm_local)
            H10_error_0.x.array[:] = assemble_vector(H10_norm_local_form).array
            H10_error = np.sqrt(sum(H10_error_0.x.array[:]))

            if print_save:
                results_saver.save_function(L2_error_0,
                                            f"L2_error_{str(i).zfill(2)}")
                results_saver.save_function(H10_error_0,
                                            f"H10_error_{str(i).zfill(2)}")

        # Save results
        if print_save:
            results_saver.save_function(eta_h, f"eta_h_{str(i).zfill(2)}")
            results_saver.save_function(uh,    f"uh_{str(i).zfill(2)}")

            results_saver.save_values([num_dofs, h10_est, L2_error, H10_error], prnt=True)

        # Marking
        if i < max_it - 1:
            cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh refinement.", print_save)

            # Uniform refinement (Omega_h only)
            if ref_method == "omega_h":
                conforming_mesh = dfx.mesh.refine(conforming_mesh)

            # Adaptive refinement
            if ref_method == "adaptive":
                facets2ref = FEM_solver.marking()
                conforming_mesh = dfx.mesh.refine(conforming_mesh, facets2ref)

        cprint("\n", print_save)

if __name__=="__main__":
    poisson_dirichlet_FEM(0.2,
                          20,
                          print_save=True,
                          ref_method="adaptive",
                          compute_exact_error=True)
    # poisson_dirichlet_FEM(0.2,
    #                       9,
    #                       print_save=True,
    #                       ref_method="omega_h")