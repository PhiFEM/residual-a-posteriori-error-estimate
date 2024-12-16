
from   basix.ufl import element
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import numpy as np
import os
from   petsc4py import PETSc

from phiFEM.src.solver import PhiFEMSolver, FEMSolver
from phiFEM.src.continuous_functions import Levelset, ExactSolution, ContinuousFunction
from phiFEM.src.saver import ResultsSaver
from phiFEM.src.mesh_scripts import mesh2d_from_levelset

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
        rhs = ContinuousFunction(expression_rhs)
    else:
        print("expression_u_exact passed, we compute the RHS from it.")
        u_exact = ExactSolution(expression_u_exact)
        u_exact.compute_negative_laplacian()
        rhs = u_exact.nlap

    """
    Create initial mesh
    """
    print("Create initial mesh.")

    nx = int(np.abs(bg_mesh_corners[1][0] - bg_mesh_corners[0][0]) * np.sqrt(2.) / cl)
    ny = int(np.abs(bg_mesh_corners[1][1] - bg_mesh_corners[0][1]) * np.sqrt(2.) / cl)
    bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bg_mesh_corners, [nx, ny])

    with XDMFFile(bg_mesh.comm, "./bg_mesh.xdmf", "w") as of:
        of.write_mesh(bg_mesh)

    if save_output:
        results_saver = ResultsSaver(output_dir)

    working_mesh = bg_mesh
    for i in range(max_it):
        CG1Element = element("Lagrange", working_mesh.topology.cell_name(), 1)

        # Parametrization of the PETSc solver
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = PETSc.KSP().create(working_mesh.comm)
        petsc_solver.setFromOptions()

        phiFEM_solver = PhiFEMSolver(working_mesh, CG1Element, petsc_solver, num_step=i, ref_strat=ref_method, save_output=save_output)
        phiFEM_solver.set_data(rhs, phi)
        phiFEM_solver.compute_tags(padding=True, plot=False)
        v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation()
        results_saver.add_new_value("dofs", num_dofs)
        phiFEM_solver.assemble()
        phiFEM_solver.solve()
        uh = phiFEM_solver.solution

        working_mesh = phiFEM_solver.submesh

        V = dfx.fem.functionspace(working_mesh, CG1Element)
        phiV = phi.interpolate(V)
        phiFEM_solver.estimate_residual()
        eta_h_H10 = phiFEM_solver.eta_h_H10
        results_saver.add_new_value("H10 estimator", np.sqrt(sum(eta_h_H10.x.array[:])))
        eta_h_L2 = phiFEM_solver.eta_h_L2
        results_saver.add_new_value("L2 estimator", np.sqrt(sum(eta_h_L2.x.array[:])))

        # Save results
        if save_output:
            results_saver.save_function(eta_h_H10,    f"eta_h_H10_{str(i).zfill(2)}")
            results_saver.save_function(eta_h_L2,     f"eta_h_L2_{str(i).zfill(2)}")
            results_saver.save_function(phiV,         f"phi_V_{str(i).zfill(2)}")
            results_saver.save_function(v0,           f"v0_{str(i).zfill(2)}")
            results_saver.save_function(uh,           f"uh_{str(i).zfill(2)}")
            results_saver.save_mesh    (working_mesh, f"mesh_{str(i).zfill(2)}")

        if compute_exact_error:
            phiFEM_solver.compute_exact_error(results_saver,
                                              expression_u_exact=expression_u_exact,
                                              save_output=save_output)

        # Marking
        if i < max_it - 1:
            # Uniform refinement (Omega_h only)
            if ref_method == "uniform":
                working_mesh = dfx.mesh.refine(working_mesh)

            # Adaptive refinement
            if ref_method in ["H10", "L2"]:
                facets2ref = phiFEM_solver.marking()
                working_mesh = dfx.mesh.refine(working_mesh, facets2ref)

        
        results_saver.save_values("results.csv")

        if save_output:
            print("\n")
    
def poisson_dirichlet_FEM(cl,
                          max_it,
                          expression_levelset,
                          source_dir,
                          expression_rhs=None,
                          expression_u_exact=None,
                          quadrature_degree=4,
                          ref_method="uniform",
                          geom_vertices=None,
                          save_output=True,
                          compute_exact_error=False):
    output_dir = os.path.join(source_dir, "output_FEM", ref_method)

    if save_output:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    phi = Levelset(expression_levelset)

    if expression_u_exact is None:
        assert expression_rhs is not None, "At least expression_rhs or expression_u_exact must not be None."
        rhs = ContinuousFunction(expression_rhs)
    else:
        print("expression_u_exact passed, we compute the RHS from it.")
        u_exact = ExactSolution(expression_u_exact)
        u_exact.compute_negative_laplacian()
        rhs = u_exact.nlap

    """
    Generate conforming mesh
    """
    _ = mesh2d_from_levelset(cl,
                             phi,
                             geom_vertices=geom_vertices,
                             output_dir=output_dir)

    with XDMFFile(MPI.COMM_WORLD, os.path.join(output_dir, "conforming_mesh.xdmf"), "r") as fi:
        conforming_mesh = fi.read_mesh()

    if save_output:
        results_saver = ResultsSaver(output_dir)
    
    for i in range(max_it):
        CG1Element = element("Lagrange", conforming_mesh.topology.cell_name(), 1)

        # Parametrization of the PETSc solver
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = PETSc.KSP().create(conforming_mesh.comm)
        petsc_solver.setFromOptions()

        FEM_solver = FEMSolver(conforming_mesh, CG1Element, petsc_solver, num_step=i, ref_strat=ref_method, save_output=save_output)
        
        dbc = dfx.fem.Function(FEM_solver.FE_space)
        facets = dfx.mesh.locate_entities_boundary(
                                conforming_mesh,
                                1,
                                lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = dfx.fem.locate_dofs_topological(FEM_solver.FE_space, 1, facets)
        bcs = [dfx.fem.dirichletbc(dbc, dofs)]
        FEM_solver.set_data(rhs, bcs=bcs)
        num_dofs = FEM_solver.set_variational_formulation()
        results_saver.add_new_value("dofs", num_dofs)
        FEM_solver.assemble()
        FEM_solver.solve()
        uh = FEM_solver.solution

        FEM_solver.estimate_residual()
        eta_h_H10 = FEM_solver.eta_h_H10
        results_saver.add_new_value("H10 estimator", np.sqrt(sum(eta_h_H10.x.array[:])))
        eta_h_L2 = FEM_solver.eta_h_L2
        results_saver.add_new_value("L2 estimator", np.sqrt(sum(eta_h_L2.x.array[:])))

        # Save results
        if save_output:
            results_saver.save_function(eta_h_H10,       f"eta_h_H10_{str(i).zfill(2)}")
            results_saver.save_function(eta_h_L2,        f"eta_h_L2_{str(i).zfill(2)}")
            results_saver.save_function(uh,              f"uh_{str(i).zfill(2)}")
            results_saver.save_mesh    (conforming_mesh, f"mesh_{str(i).zfill(2)}")

        if compute_exact_error:
            FEM_solver.compute_exact_error(results_saver,
                                           expression_u_exact=expression_u_exact,
                                           save_output=save_output)

        # Marking
        if i < max_it - 1:
            # Uniform refinement (Omega_h only)
            if ref_method == "uniform":
                conforming_mesh = dfx.mesh.refine(conforming_mesh)

            # Adaptive refinement
            if ref_method in ["H10", "L2"]:
                facets2ref = FEM_solver.marking()
                conforming_mesh = dfx.mesh.refine(conforming_mesh, facets2ref)

        if save_output:
            with XDMFFile(MPI.COMM_WORLD, os.path.join(output_dir, "conforming_mesh.xdmf"), "w") as of:
                of.write_mesh(conforming_mesh)
            
            results_saver.save_values("results.csv")
        
        print("\n")