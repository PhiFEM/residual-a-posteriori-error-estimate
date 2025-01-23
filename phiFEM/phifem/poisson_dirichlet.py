from   basix.ufl import element
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import numpy as np
import numpy.typing as npt
import os
from   os import PathLike
from   petsc4py.PETSc import Options, KSP
from   typing import Tuple

from phiFEM.phifem.solver import PhiFEMSolver, FEMSolver
from phiFEM.phifem.continuous_functions import Levelset, ExactSolution, ContinuousFunction
from phiFEM.phifem.mesh_scripts import mesh2d_from_levelset
from phiFEM.phifem.saver import ResultsSaver
from phiFEM.phifem.utils import assemble_and_save_residual

PathStr = PathLike[str] | str
NDArrayTuple = Tuple[npt.NDArray[np.float64]]
NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

def poisson_dirichlet_phiFEM(cl: float,
                             max_it: int,
                             expression_levelset: NDArrayFunction,
                             source_dir: PathStr,
                             expression_rhs: NDArrayFunction | None = None,
                             expression_u_exact: NDArrayFunction | None = None,
                             bbox_vertices: npt.NDArray[np.float64] = np.array([[-1., 1.],
                                                                                [-1., 1.]]),
                             quadrature_degree: int = 4,
                             sigma_D: float = 1.,
                             save_output: bool = True,
                             ref_method: str = "uniform",
                             compute_exact_error: bool = False) -> None:
    """ Main loop of phiFEM solve and refinement for a Poisson-Dirichlet test case.

    Args:
        cl: float, the characteristic length of the initial mesh.
        max_it: int, the maximum number of iterations.
        expression_levelset: method, the expression of the levelset.
        source_dir: Path object or str, the name of the test case source directory.
        expression_rhs: (optional) method, the expression of the right-hand side term of the PDE (force term) (if None, it is computed from expression_u_exact).
        expression_u_exact: (optional) method, the expression of the exact solution (if None and compute_exact_error=True, a reference solution is computed on a finer mesh).
        bbox_vertices: (optional) (2,2) ndarray, the coordinates of vertices of the background mesh.
        quadrature_degree: (optional) int, the degree of quadrature.
        sigma_D: (optional) float, the phiFEM stabilization coefficient.
        save_output: (optional) bool, if True, saves the functions, meshes and values on the disk.
        ref_method: (optional) str, specify the refinement method (three choices: uniform for uniform refinement, H10 for adaptive refinement based on the H10 residual estimator, L2 for adaptive refinement based on the L2 residual estimator).
        compute_exact_error: (optional) bool, if True compute the exact error on a finer reference mesh.
    """
    
    output_dir = os.path.join(source_dir, "output_phiFEM", ref_method)

    if save_output:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    phi = Levelset(expression_levelset)

    rhs: ContinuousFunction
    if expression_rhs is not None:
        rhs = ContinuousFunction(expression_rhs)
    elif expression_u_exact is not None:
        if expression_rhs is None:
            print("We compute the RHS from expression_u_exact using JAX.")
        u_exact = ExactSolution(expression_u_exact)
        u_exact.compute_negative_laplacian()
        if u_exact.nlap is None:
            raise TypeError("u_exact.nlap is None.")
        rhs = u_exact.nlap
    else:
        raise ValueError("poisson_dirichlet_phiFEM need expression_rhs or expression_u_exact not to be None.")

    """
    Create initial mesh
    """
    print("Create initial mesh.")
    nx = int(np.abs(bbox_vertices[0, 1] - bbox_vertices[0, 0]) * np.sqrt(2.) / cl)
    ny = int(np.abs(bbox_vertices[1, 1] - bbox_vertices[1, 0]) * np.sqrt(2.) / cl)
    bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, bbox_vertices.T, [nx, ny])

    with XDMFFile(bg_mesh.comm, "./bg_mesh.xdmf", "w") as of:
        of.write_mesh(bg_mesh)

    if save_output:
        results_saver = ResultsSaver(output_dir)

    working_mesh = bg_mesh
    for i in range(max_it):
        CG1Element = element("Lagrange", working_mesh.topology.cell_name(), 1)
        CG2Element = element("Lagrange", working_mesh.topology.cell_name(), 2)

        # Parametrization of the PETSc solver
        options = Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = KSP().create(working_mesh.comm)
        petsc_solver.setFromOptions()

        phiFEM_solver = PhiFEMSolver(working_mesh,
                                     CG1Element,
                                     petsc_solver,
                                     levelset_element=CG2Element,
                                     num_step=i,
                                     ref_strat=ref_method,
                                     save_output=save_output)
        phiFEM_solver.set_source_term(rhs)
        phiFEM_solver.set_levelset(phi)
        phiFEM_solver.compute_tags(padding=True, plot=False)
        v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation()
        results_saver.add_new_value("dofs", num_dofs)
        phiFEM_solver.assemble()
        phiFEM_solver.solve()
        uh = phiFEM_solver.get_solution()
        wh = phiFEM_solver.get_solution_wh()

        if phiFEM_solver.submesh is None:
            raise TypeError("phiFEM_solver.submesh is None.")
        
        working_mesh = phiFEM_solver.submesh

        V = dfx.fem.functionspace(working_mesh, CG1Element)
        phiV = phi.interpolate(V)
        h10_residuals, l2_residuals, correction_function = phiFEM_solver.estimate_residual()
        eta_h_H10 = phiFEM_solver.get_eta_h_H10()
        results_saver.add_new_value("H10 estimator", np.sqrt(sum(eta_h_H10.x.array[:])))
        eta_h_L2 = phiFEM_solver.get_eta_h_L2()
        results_saver.add_new_value("L2 estimator", np.sqrt(sum(eta_h_L2.x.array[:])))

        # Save results
        if save_output:
            for dict_res, norm in zip([h10_residuals, l2_residuals], ["H10", "L2"]):
                for key, res_letter in zip(dict_res.keys(), ["T", "E", "G", "Eb"]):
                    eta = dict_res[key]
                    if eta is not None:
                        res_name = "eta_" + res_letter + "_" + norm
                        assemble_and_save_residual(working_mesh, results_saver, eta, res_name, i)

            results_saver.save_function(eta_h_H10,           f"eta_h_H10_{str(i).zfill(2)}")
            results_saver.save_function(eta_h_L2,            f"eta_h_L2_{str(i).zfill(2)}")
            results_saver.save_function(phiV,                f"phi_V_{str(i).zfill(2)}")
            results_saver.save_function(v0,                  f"v0_{str(i).zfill(2)}")
            results_saver.save_function(uh,                  f"uh_{str(i).zfill(2)}")
            results_saver.save_function(wh,                  f"wh_{str(i).zfill(2)}")
            results_saver.save_function(correction_function, f"boundary_correction_{str(i).zfill(2)}")
            results_saver.save_mesh    (working_mesh,        f"mesh_{str(i).zfill(2)}")

        if compute_exact_error:
            solution_degree = phiFEM_solver.solution.function_space.element.basix_element.degree
            levelset_degree = phiFEM_solver.levelset_space.element.basix_element.degree
            ref_degree = solution_degree + levelset_degree + 1
            phiFEM_solver.compute_exact_error(results_saver,
                                              ref_degree= ref_degree,
                                              expression_u_exact=expression_u_exact,
                                              save_output=save_output)
            phiFEM_solver.compute_efficiency_coef(results_saver, norm="H10")
            phiFEM_solver.compute_efficiency_coef(results_saver, norm="L2")

        # Marking
        if i < max_it - 1:
            # Uniform refinement (Omega_h only)
            if ref_method == "uniform":
                working_mesh = dfx.mesh.refine(working_mesh)

            # Adaptive refinement
            if ref_method in ["H10", "L2"]:
                facets2ref = phiFEM_solver.marking()
                working_mesh = dfx.mesh.refine(working_mesh, facets2ref)

        if save_output:
            results_saver.save_values("results.csv")
            print("\n")
    
def poisson_dirichlet_FEM(cl: float,
                          max_it: int,
                          expression_levelset: NDArrayFunction,
                          source_dir: PathStr,
                          expression_rhs: NDArrayFunction | None = None,
                          expression_u_exact: NDArrayFunction | None = None,
                          quadrature_degree: int = 4,
                          save_output: bool = True,
                          ref_method: str = "uniform",
                          compute_exact_error: bool = False,
                          bbox_vertices: npt.NDArray[np.float64] = np.array([[-1., 1.], [-1., 1.]]),
                          geom_vertices: npt.NDArray[np.float64] | None = None,
                          remesh_boundary: bool = False) -> None:
    """ Main loop of FEM solve and refinement for a Poisson-Dirichlet test case.

    Args:
        cl: float, the characteristic length of the initial mesh.
        max_it: int, the maximum number of iterations.
        source_dir: Path object or str, the name of the test case source directory.
        expression_rhs: (optional) method, the expression of the right-hand side term of the PDE (force term) (if None, it is computed from expression_u_exact).
        expression_u_exact: (optional) method, the expression of the exact solution (if None and compute_exact_error=True, a reference solution is computed on a finer mesh).
        quadrature_degree: (optional) int, the degree of quadrature.
        save_output: (optional) bool, if True, saves the functions, meshes and values on the disk.
        ref_method: (optional) str, specify the refinement method (three choices: uniform for uniform refinement, H10 for adaptive refinement based on the H10 residual estimator, L2 for adaptive refinement based on the L2 residual estimator).
        compute_exact_error: (optional) bool, if True compute the exact error on a finer reference mesh.
        geom_vertices: (optional) (N, 2) ndarray, vertices of the exact domain.
        remesh_boundary: if True, recompute the boundary vertices at each refinement step.
    """
    output_dir = os.path.join(source_dir, "output_FEM", ref_method)

    if save_output:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    phi = Levelset(expression_levelset)

    rhs: ContinuousFunction
    if expression_rhs is not None:
        rhs = ContinuousFunction(expression_rhs)
    elif expression_u_exact is not None:
        print("expression_u_exact passed, we compute the RHS from it.")
        u_exact = ExactSolution(expression_u_exact)
        u_exact.compute_negative_laplacian()
        if u_exact.nlap is None:
            raise TypeError("u_exact.nlap is None.")
        rhs = u_exact.nlap
    else:
        raise ValueError("poisson_dirichlet_FEM need expression_rhs or expression_u_exact not to be None.")

    """
    Generate conforming mesh
    """
    _ = mesh2d_from_levelset(cl,
                             phi,
                             bbox=bbox_vertices,
                             geom_vertices=geom_vertices,
                             output_dir=output_dir)

    with XDMFFile(MPI.COMM_WORLD, os.path.join(output_dir, "conforming_mesh.xdmf"), "r") as fi:
        conforming_mesh = fi.read_mesh()

    if save_output:
        results_saver = ResultsSaver(output_dir)
    
    for i in range(max_it):
        CG1Element = element("Lagrange", conforming_mesh.topology.cell_name(), 1)

        # Parametrization of the PETSc solver
        options = Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"
        petsc_solver = KSP().create(conforming_mesh.comm)
        petsc_solver.setFromOptions()

        FEM_solver = FEMSolver(conforming_mesh, CG1Element, petsc_solver, num_step=i, ref_strat=ref_method, save_output=save_output)
        
        FEM_solver.set_source_term(rhs)
        dbc = dfx.fem.Function(FEM_solver.FE_space)
        facets = dfx.mesh.locate_entities_boundary(
                                conforming_mesh,
                                1,
                                lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = dfx.fem.locate_dofs_topological(FEM_solver.FE_space, 1, facets)
        bcs = [dfx.fem.dirichletbc(dbc, dofs)]
        FEM_solver.set_boundary_conditions(bcs)
        num_dofs = FEM_solver.set_variational_formulation()
        results_saver.add_new_value("dofs", num_dofs)
        FEM_solver.assemble()
        FEM_solver.solve()
        uh = FEM_solver.get_solution()

        h10_residuals, l2_residuals = FEM_solver.estimate_residual()
        eta_h_H10 = FEM_solver.get_eta_h_H10()
        results_saver.add_new_value("H10 estimator", np.sqrt(sum(eta_h_H10.x.array[:])))
        eta_h_L2 = FEM_solver.get_eta_h_L2()
        results_saver.add_new_value("L2 estimator", np.sqrt(sum(eta_h_L2.x.array[:])))

        # Save results
        if save_output:
            for dict_res, norm in zip([h10_residuals, l2_residuals], ["H10", "L2"]):
                for key, res_letter in zip(dict_res.keys(), ["T", "E", "G", "Eb"]):
                    eta = dict_res[key]
                    if eta is not None:
                        res_name = "eta_" + res_letter + "_" + norm
                        assemble_and_save_residual(conforming_mesh, results_saver, eta, res_name, i)

            results_saver.save_function(eta_h_H10,       f"eta_h_H10_{str(i).zfill(2)}")
            results_saver.save_function(eta_h_L2,        f"eta_h_L2_{str(i).zfill(2)}")
            results_saver.save_function(uh,              f"uh_{str(i).zfill(2)}")
            results_saver.save_mesh    (conforming_mesh, f"mesh_{str(i).zfill(2)}")

        if compute_exact_error:
            FEM_solver.compute_exact_error(results_saver,
                                           ref_degree=2,
                                           expression_u_exact=expression_u_exact,
                                           save_output=save_output,
                                           save_exact_solution=True)
            FEM_solver.compute_efficiency_coef(results_saver, norm="H10")
            FEM_solver.compute_efficiency_coef(results_saver, norm="L2")

        if i < max_it - 1:
            # Uniform refinement (Omega_h only)
            if ref_method == "uniform":
                conforming_mesh = dfx.mesh.refine(conforming_mesh)

            # Adaptive refinement
            if ref_method in ["H10", "L2"]:
                # Marking
                facets2ref = FEM_solver.marking()
                conforming_mesh = dfx.mesh.refine(conforming_mesh, facets2ref)

            if remesh_boundary:
                vertices_coordinates = conforming_mesh.geometry.x
                boundary_vertices = dfx.mesh.locate_entities_boundary(conforming_mesh,
                                                                      0,
                                                                      lambda x: np.full(x.shape[1], True, dtype=bool))
                boundary_vertices_coordinates = vertices_coordinates[boundary_vertices].T
                _ = mesh2d_from_levelset(1.,
                                        phi,
                                        output_dir=output_dir,
                                        interior_vertices=boundary_vertices_coordinates)
        if save_output:
            with XDMFFile(MPI.COMM_WORLD, os.path.join(output_dir, "conforming_mesh.xdmf"), "w") as of:
                of.write_mesh(conforming_mesh)
            
            results_saver.save_values("results.csv")
        
        print("\n")