
from   basix.ufl import element
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   dolfinx.fem.petsc import assemble_vector
from   mpi4py import MPI
import numpy as np
import os
from   petsc4py import PETSc
import ufl
from   ufl import inner, grad

from phiFEM.src.solver import PhiFEMSolver, FEMSolver
from phiFEM.src.continuous_functions import Levelset, ExactSolution, ContinuousFunction
from phiFEM.src.saver import ResultsSaver
from phiFEM.src.mesh_scripts import mesh2d_from_levelset

def cprint(str2print, save_output=True, file=None):
    if save_output:
        print(str2print, file=file)

def poisson_dirichlet_phiFEM(cl,
                             max_it,
                             expression_levelset,
                             source_dir,
                             expression_rhs=None,
                             expression_u_exact=None,
                             bg_mesh_corners=[np.array([0., 0.]),
                                              np.array([1., 1.])],
                             quadrature_degree=4,
                             sigma_D=1.,
                             save_output=True,
                             ref_method="uniform",
                             compute_exact_error=False):
    output_dir = os.path.join(source_dir, "output_phiFEM", ref_method)

    if save_output:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    phi = Levelset(expression_levelset)

    if expression_u_exact is None:
        assert expression_rhs is not None, "At least expression_rhs or expression_u_exact must not be None."
        f = ContinuousFunction(expression_rhs)
    else:
        cprint("expression_u_exact passed, we compute the RHS from it.")
        u_exact = ExactSolution(expression_u_exact)
        u_exact.compute_negative_laplacian()
        f = u_exact.nlap

    """
    Create initial mesh
    """
    cprint("Create initial mesh.", save_output)

    nx = int(np.abs(bg_mesh_corners[1][0] - bg_mesh_corners[0][0]) * np.sqrt(2.) / cl)
    ny = int(np.abs(bg_mesh_corners[1][1] - bg_mesh_corners[0][1]) * np.sqrt(2.) / cl)
    bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bg_mesh_corners, [nx, ny])

    with XDMFFile(bg_mesh.comm, "./bg_mesh.xdmf", "w") as of:
        of.write_mesh(bg_mesh)

    data = ["dofs", "H10 estimator"]

    if save_output:
        results_saver = ResultsSaver(output_dir, data)

    if compute_exact_error:
        phiFEM_solutions = []
    
    working_mesh = bg_mesh
    for i in range(max_it):
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", save_output)
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
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Data interpolation.", save_output)
        phiFEM_solver.set_data(f, phi)
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh tags computation.", save_output)
        phiFEM_solver.compute_tags(padding=True)
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Variational formulation set up.", save_output)
        v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation(sigma=20.)
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Linear system assembly.", save_output)
        phiFEM_solver.assemble()
        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Solve.", save_output)
        phiFEM_solver.solve()
        uh = phiFEM_solver.solution

        if compute_exact_error:
            phiFEM_solutions.append(uh)
        
        working_mesh = phiFEM_solver.submesh

        cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", save_output)
        V = dfx.fem.functionspace(working_mesh, CG1Element)
        phiV = phi.interpolate(V)
        phiFEM_solver.estimate_residual(boundary_term=True)
        eta_h_H10 = phiFEM_solver.eta_h_H10
        H10_est = np.sqrt(sum(eta_h_H10.x.array[:]))
        eta_h_L2 = phiFEM_solver.eta_h_L2
        L2_est = np.sqrt(sum(eta_h_L2.x.array[:]))

        # Save results
        if save_output:
            results_saver.save_function(eta_h_H10, f"eta_h_H10_{str(i).zfill(2)}")
            results_saver.save_function(eta_h_L2,  f"eta_h_L2_{str(i).zfill(2)}")
            results_saver.save_function(phiV,      f"phi_V_{str(i).zfill(2)}")
            results_saver.save_function(v0,        f"v0_{str(i).zfill(2)}")
            results_saver.save_function(uh,        f"uh_{str(i).zfill(2)}")

            results_saver.save_values([num_dofs,
                                       H10_est,
                                       L2_est],
                                       prnt=True)

        # Marking
        if i < max_it - 1:
            cprint(f"Solver: phiFEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh refinement.", save_output)

            # Uniform refinement (Omega_h only)
            if ref_method == "uniform":
                working_mesh = dfx.mesh.refine(working_mesh)

            # Adaptive refinement
            if ref_method == "adaptive":
                facets2ref = phiFEM_solver.marking()
                working_mesh = dfx.mesh.refine(working_mesh, facets2ref)

        cprint("\n", save_output)
    
    if compute_exact_error:
        results_error = {"L2 error": [], "H10 error": []}
        extra_ref  = 6
        ref_degree = 3
        cprint(f"Solver: phiFEM. Method: {ref_method}. Compute exact errors.", save_output)

        FEM_dir_list = [subdir if subdir!="output_phiFEM" else "output_FEM" for subdir in output_dir.split(sep=os.sep)]
        FEM_dir = os.path.join("/", *(FEM_dir_list))
        with XDMFFile(MPI.COMM_WORLD, os.path.join(FEM_dir, "conforming_mesh.xdmf"), "r") as fi:
            reference_mesh = fi.read_mesh(name="Grid")
        
        for j in range(extra_ref):
            reference_mesh.topology.create_entities(reference_mesh.topology.dim - 1)
            reference_mesh = dfx.mesh.refine(reference_mesh)

        CGfElement = element("CG", reference_mesh.topology.cell_name(), ref_degree)
        reference_space = dfx.fem.functionspace(reference_mesh, CGfElement)

        if expression_u_exact is not None:
            u_exact_ref = u_exact.interpolate(reference_space)
        else:
            # Parametrization of the PETSc solver
            options = PETSc.Options()
            options["ksp_type"] = "cg"
            options["pc_type"] = "hypre"
            options["ksp_rtol"] = 1e-7
            options["pc_hypre_type"] = "boomeramg"
            petsc_solver = PETSc.KSP().create(reference_mesh.comm)
            petsc_solver.setFromOptions()

            FEM_solver = FEMSolver(reference_mesh, CGfElement, petsc_solver, num_step=i)
        
            dbc = dfx.fem.Function(FEM_solver.FE_space)
            facets = dfx.mesh.locate_entities_boundary(
                                    reference_mesh,
                                    1,
                                    lambda x: np.ones(x.shape[1], dtype=bool))
            dofs = dfx.fem.locate_dofs_topological(FEM_solver.FE_space, 1, facets)
            bcs = [dfx.fem.dirichletbc(dbc, dofs)]
            FEM_solver.set_data(f, bcs=bcs)
            num_dofs = FEM_solver.set_variational_formulation()
            FEM_solver.assemble()
            FEM_solver.solve()
            u_exact_ref = FEM_solver.solution

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

            if save_output:
                results_saver.save_function(L2_error_0,
                                            f"L2_error_{str(i).zfill(2)}")
                results_saver.save_function(H10_error_0,
                                            f"H10_error_{str(i).zfill(2)}")

        if save_output:
            results_saver.add_new_values(results_error, prnt=True)

def poisson_dirichlet_FEM(cl,
                          max_it,
                          expression_levelset,
                          source_dir,
                          expression_rhs=None,
                          expression_u_exact=None,
                          quadrature_degree=4,
                          ref_method="uniform",
                          geom_vertices=None,
                          save_output=True):
    output_dir = os.path.join(source_dir, "output_phiFEM", ref_method)
    compute_exact_error = expression_u_exact is not None

    if save_output:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    phi = Levelset(expression_levelset)

    if expression_u_exact is None:
        assert expression_rhs is not None, "At least expression_rhs or expression_u_exact must not be None."
        f = ContinuousFunction(expression_rhs)
    else:
        cprint("expression_u_exact passed, we compute the RHS from it.")
        u_exact = ExactSolution(expression_u_exact)
        u_exact.compute_negative_laplacian()
        f = u_exact.nlap

    """
    Generate conforming mesh
    """
    _ = mesh2d_from_levelset(cl,
                             phi,
                             geom_vertices=geom_vertices,
                             output_dir=output_dir)
    
    with XDMFFile(MPI.COMM_WORLD, os.path.join(output_dir, "conforming_mesh.xdmf"), "r") as fi:
        conforming_mesh = fi.read_mesh(name="Grid")

    if compute_exact_error:
        data = ["dofs", "H10 estimator", "L2 error", "H10 error"]
    else:
        data = ["dofs", "H10 estimator", "L2 estimator"]

    if save_output:
        results_saver = ResultsSaver(output_dir, data)

    for i in range(max_it):
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Creation FE space and data interpolation", save_output)
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
        
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Data interpolation.", save_output)
        dbc = dfx.fem.Function(FEM_solver.FE_space)
        facets = dfx.mesh.locate_entities_boundary(
                                conforming_mesh,
                                1,
                                lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = dfx.fem.locate_dofs_topological(FEM_solver.FE_space, 1, facets)
        bcs = [dfx.fem.dirichletbc(dbc, dofs)]
        FEM_solver.set_data(f, bcs=bcs)
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Variational formulation set up.", save_output)
        num_dofs = FEM_solver.set_variational_formulation()
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Linear system assembly.", save_output)
        FEM_solver.assemble()
        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Solve.", save_output)
        FEM_solver.solve()
        uh = FEM_solver.solution

        cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: a posteriori error estimation.", save_output)
        FEM_solver.estimate_residual()
        eta_h_H10 = FEM_solver.eta_h_H10
        H10_est = np.sqrt(sum(eta_h_H10.x.array[:]))
        eta_h_L2 = FEM_solver.eta_h_L2
        L2_est = np.sqrt(sum(eta_h_L2.x.array[:]))

        if compute_exact_error:
            cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: compute exact errors.", save_output)
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

            if save_output:
                results_saver.save_function(L2_error_0,
                                            f"L2_error_{str(i).zfill(2)}")
                results_saver.save_function(H10_error_0,
                                            f"H10_error_{str(i).zfill(2)}")

        # Save results
        if save_output:
            results_saver.save_function(eta_h_H10, f"eta_h_H10_{str(i).zfill(2)}")
            results_saver.save_function(eta_h_L2,  f"eta_h_L2_{str(i).zfill(2)}")
            results_saver.save_function(uh,        f"uh_{str(i).zfill(2)}")

            if compute_exact_error:
                results_saver.save_values([num_dofs, H10_est, L2_est, L2_error, H10_error], prnt=True)
            else:
                results_saver.save_values([num_dofs, H10_est, L2_est], prnt=True)

        # Marking
        if i < max_it - 1:
            cprint(f"Solver: FEM. Method: {ref_method}. Iteration n° {str(i).zfill(2)}: Mesh refinement.", save_output)

            # Uniform refinement (Omega_h only)
            if ref_method == "uniform":
                conforming_mesh = dfx.mesh.refine(conforming_mesh)

            # Adaptive refinement
            if ref_method == "adaptive":
                facets2ref = FEM_solver.marking()
                conforming_mesh = dfx.mesh.refine(conforming_mesh, facets2ref)

        cprint("\n", save_output)