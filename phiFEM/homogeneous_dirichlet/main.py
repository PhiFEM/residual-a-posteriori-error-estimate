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

from utils.solver import PhiFEMSolver
from utils.continuous_functions import Levelset, ExactSolution
from utils.saver import ResultsSaver

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
    cprint("Create initial mesh.", print_save)
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

    if compute_exact_error:
        phiFEM_solutions = []
    
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

        if compute_exact_error:
            phiFEM_solutions.append(uh)
        
        working_mesh = phiFEM_solver.submesh

        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", print_save)
        V = dfx.fem.functionspace(working_mesh, CG1Element)
        phiV = phi.interpolate(V)
        phiFEM_solver.estimate_residual(boundary_term=True)
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
    
    if compute_exact_error:
        results_error = {"L2 error": [], "H10 error": []}
        extra_ref  = 5
        ref_degree = 5
        cprint(f"Solver: phiFEM. Method: {ref_method}. Compute exact errors.", print_save)

        FEM_dir = os.path.join("output_FEM", *(os.path.split(output_dir)[1:]))
        with XDMFFile(MPI.COMM_WORLD, os.path.join(FEM_dir, "conforming_mesh.xdmf"), "r") as fi:
            reference_mesh = fi.read_mesh(name="Grid")
        
        for j in range(extra_ref):
            reference_mesh.topology.create_entities(reference_mesh.topology.dim - 1)
            reference_mesh = dfx.mesh.refine(reference_mesh)

        CGfElement = element("CG", reference_mesh.topology.cell_name(), ref_degree)
        reference_space = dfx.fem.functionspace(reference_mesh, CGfElement)

        u_exact_ref = u_exact.interpolate(reference_space)

        for i in range(max_it):
            phiFEM_solution = phiFEM_solutions[i]

            uh_ref = dfx.fem.Function(reference_space)
            nmm = dfx.fem.create_nonmatching_meshes_interpolation_data(
                                uh_ref.function_space.mesh,
                                uh_ref.function_space.element,
                                phiFEM_solution.function_space.mesh, padding=1.e-14)
            uh_ref.interpolate(phiFEM_solution, nmm_interpolation_data=nmm)
            e_ref = dfx.fem.Function(reference_space)
            e_ref.x.array[:] = u_exact_ref.x.array - uh_ref.x.array

            dx2 = ufl.Measure("dx",
                              domain=reference_mesh,
                              metadata={"quadrature_degree": 2 * (ref_degree + 1)})

            DG0Element = element("DG", reference_mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(reference_mesh, DG0Element)
            w0 = ufl.TrialFunction(V0)

            L2_norm_local = inner(inner(e_ref, e_ref), w0) * dx2
            H10_norm_local = inner(inner(grad(e_ref), grad(e_ref)), w0) * dx2

            L2_error_0 = dfx.fem.Function(V0)
            H10_error_0 = dfx.fem.Function(V0)

            L2_norm_local_form = dfx.fem.form(L2_norm_local)
            L2_error_0.x.array[:] = assemble_vector(L2_norm_local_form).array
            results_error["L2 error"].append(np.sqrt(sum(L2_error_0.x.array[:])))

            H10_norm_local_form = dfx.fem.form(H10_norm_local)
            H10_error_0.x.array[:] = assemble_vector(H10_norm_local_form).array
            results_error["H10 error"].append(np.sqrt(sum(H10_error_0.x.array[:])))

            if print_save:
                results_saver.save_function(L2_error_0,
                                            f"L2_error_{str(i).zfill(2)}")
                results_saver.save_function(H10_error_0,
                                            f"H10_error_{str(i).zfill(2)}")

        results_saver.add_new_values(results_error, prnt=True)

if __name__=="__main__":
    poisson_dirichlet_phiFEM(0.2,
                             20,
                             print_save=True,
                             ref_method="adaptive",
                             compute_exact_error=True)
    # poisson_dirichlet_phiFEM(0.2,
    #                          8,
    #                          print_save=True,
    #                          ref_method="omega_h",
    #                          compute_exact_error=True)