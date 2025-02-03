from   basix.ufl import element
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.io import XDMFFile
from   dolfinx.mesh import Mesh
from   mpi4py import MPI
import numpy as np
import numpy.typing as npt
from   numpy.typing import NDArray
import os
from   os import PathLike
from   petsc4py.PETSc import Options, KSP # type: ignore[attr-defined]
from   typing import Tuple, Any

from phiFEM.phifem.solver import PhiFEMSolver, FEMSolver
from phiFEM.phifem.continuous_functions import Levelset, ExactSolution, ContinuousFunction
from phiFEM.phifem.mesh_scripts import mesh2d_from_levelset
from phiFEM.phifem.saver import ResultsSaver
from phiFEM.phifem.utils import assemble_and_save_residual

PathStr = PathLike[str] | str
NDArrayTuple = Tuple[npt.NDArray[np.float64]]
NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

class PhiFEMRefinementLoop:
    def __init__(self,
                 initial_mesh_size: float,
                 iteration_number: int,
                 refinement_method: str,
                 expression_levelset: NDArrayFunction,
                 stabilization_parameter: float,
                 source_dir: PathStr):
        
        if refinement_method not in ["uniform", "H10", "L2"]:
            raise ValueError("refinement_method must be 'uniform', 'H10' or 'L2'.")

        self.bbox: NDArray                        = np.array([[-1.0, 1.0],
                                                              [-1.0, 1.0]])
        self.boundary_detection_degree: int       = 1
        self.box_mode: bool                       = False
        self.exact_error: bool                    = False
        self.exact_solution: ExactSolution | None = None
        self.finite_element_degree: int           = 1
        self.initial_mesh_size: float             = initial_mesh_size
        self.initial_bg_mesh: Mesh | None         = None
        self.iteration_number: int                = iteration_number
        self.levelset: Levelset                   = Levelset(expression_levelset)
        self.levelset_degree: int                 = 1
        self.marking_parameter: float             = 0.3
        self.quadrature_degree: int | None        = None
        self.rhs: ContinuousFunction | None       = None
        self.refinement_method: str               = refinement_method
        self.results_saver: ResultsSaver          = ResultsSaver(os.path.join(source_dir,
                                                                              "output_phiFEM",
                                                                              refinement_method))
        self.source_dir: PathStr                  = source_dir
        self.stabilization_parameter: float       = stabilization_parameter
        self.use_fine_space: bool                 = False

    def set_parameters(self, parameters: dict[str, Any], expressions: dict[str, NDArrayFunction]):
        self.bbox                      = np.asarray(parameters["bbox"])
        self.boundary_detection_degree = parameters["boundary_detection_degree"]
        self.box_mode                  = parameters["box_mode"] 
        self.exact_error               = parameters["exact_error"]
        self.exact_solution            = ExactSolution(expressions["expression_u_exact"])
        self.finite_element_degree     = parameters["finite_element_degree"]
        self.levelset_degree           = parameters["levelset_degree"]
        self.marking_parameter         = parameters["marking_parameter"]
        self.quadrature_degree         = parameters["quadrature_degree"]
        self.use_fine_space            = parameters["use_fine_space"]
        self.save_output               = parameters["save_output"]

        if expressions["expression_rhs"] is not None:
            self.rhs = ContinuousFunction(expressions["expression_rhs"])
        else:
            self.rhs = None

    def set_bbox(self, bbox: NDArray):
        self.bbox = bbox
    
    def set_boundary_detection_degree(self, detection_degree: int):
        self.boundary_detection_degree = detection_degree
    
    def set_box_mode(self, box_mode: bool):
        self.box_mode = box_mode
    
    def set_exact_error_on(self, exact_error: bool):
        self.exact_error = exact_error
    
    def set_exact_solution(self, expression_exact_solution: NDArrayFunction):
        self.exact_solution = ExactSolution(expression_exact_solution)
    
    def set_finite_element_degree(self, finite_element_degree: int):
        self.finite_element_degree = finite_element_degree
    
    def set_levelset_degree(self, levelset_degree: int):
        self.levelset_degree = levelset_degree
    
    def set_marking_parameter(self, marking_parameter: float):
        self.marking_parameter = marking_parameter
    
    def set_quadrature_degree(self, quadrature_degree: int):
        self.quadrature_degree = quadrature_degree
    
    def set_use_fine_space(self, use_fine_space: bool):
        self.use_fine_space = use_fine_space

    def set_results_saver(self, output_dir: PathStr):
        self.results_saver = ResultsSaver(output_dir)
    
    def set_rhs(self, expression_rhs: NDArrayFunction):
        self.rhs = ContinuousFunction(expression_rhs)

    def set_save_output(self, save_output: bool):
        self.save_output = save_output
    
    def set_stabilization_parameter(self, stabilization_parameter: float):
        self.stabilization_parameter = stabilization_parameter
    
    def create_initial_bg_mesh(self):
        """Create the initial background mesh
        """
        print("Create initial mesh.")
        nx = int(np.abs(self.bbox[0, 1] - self.bbox[0, 0]) * np.sqrt(2.) / self.initial_mesh_size)
        ny = int(np.abs(self.bbox[1, 1] - self.bbox[1, 0]) * np.sqrt(2.) / self.initial_mesh_size)
        self.initial_bg_mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, self.bbox.T, [nx, ny])

        if self.results_saver is not None:
            self.results_saver.save_mesh(self.initial_bg_mesh, "initial_bg_mesh")
    
    def run(self):
        if self.rhs is None:
            self.rhs: ContinuousFunction
            if self.exact_solution is not None:
                self.exact_solution.compute_negative_laplacian()
                if self.exact_solution.nlap is None:
                    raise ValueError("exact_solution.nlap is None.")
                self.rhs = self.exact_solution.nlap
            else:
                raise ValueError("poisson_dirichlet_phiFEM need expression_rhs or expression_u_exact not to be None.")

        if self.initial_bg_mesh is None:
            raise ValueError("REFINEMENT_LOOP.initial_bg_mesh is None, did you forget to create the initial mesh ? (REFINEMENT_LOOP.create_initial_bg_mesh())")
        working_mesh = self.initial_bg_mesh
        for i in range(self.iteration_number):
            whElement        = element("Lagrange", working_mesh.topology.cell_name(), self.finite_element_degree)
            levelsetElement  = element("Lagrange", working_mesh.topology.cell_name(), self.levelset_degree)
            detectionElement = element("Lagrange", working_mesh.topology.cell_name(), self.boundary_detection_degree)

            # Parametrization of the PETSc solver
            options = Options()
            options["ksp_type"] = "cg"
            options["pc_type"] = "hypre"
            options["ksp_rtol"] = 1e-7
            options["pc_hypre_type"] = "boomeramg"
            petsc_solver = KSP().create(working_mesh.comm)
            petsc_solver.setFromOptions()

            phiFEM_solver = PhiFEMSolver(working_mesh,
                                         whElement,
                                         petsc_solver,
                                         levelset_element=levelsetElement,
                                         use_fine_space=self.use_fine_space,
                                         box_mode=self.box_mode,
                                         num_step=i,
                                         ref_strat=self.refinement_method,
                                         save_output=self.save_output)

            phiFEM_solver.set_source_term(self.rhs)
            phiFEM_solver.set_levelset(self.levelset)
            phiFEM_solver.compute_tags(detection_element=detectionElement, plot=False)
            v0, dx, dS, num_dofs = phiFEM_solver.set_variational_formulation(sigma=self.stabilization_parameter,
                                                                             quadrature_degree=self.quadrature_degree)
            phiFEM_solver.assemble()
            phiFEM_solver.solve()
            uh = phiFEM_solver.get_solution()
            wh = phiFEM_solver.get_solution_wh()

            if phiFEM_solver.submesh is None:
                raise TypeError("phiFEM_solver.submesh is None.")

            working_mesh = phiFEM_solver.submesh

            h10_residuals, l2_residuals, correction_function = phiFEM_solver.estimate_residual()
            eta_h_H10 = phiFEM_solver.get_eta_h_H10()
            self.results_saver.add_new_value("H10 estimator", np.sqrt(sum(eta_h_H10.x.array[:])))
            eta_h_L2 = phiFEM_solver.get_eta_h_L2()
            self.results_saver.add_new_value("L2 estimator", np.sqrt(sum(eta_h_L2.x.array[:])))

            CG1Element = element("Lagrange", working_mesh.topology.cell_name(), 1)
            V = dfx.fem.functionspace(working_mesh, CG1Element)
            phiV = self.levelset.interpolate(V)
            # Save results
            if self.save_output:
                for dict_res, norm in zip([h10_residuals, l2_residuals], ["H10", "L2"]):
                    for key, res_letter in zip(dict_res.keys(), ["T", "E", "G", "Eb"]):
                        eta = dict_res[key]
                        if eta is not None:
                            res_name = "eta_" + res_letter + "_" + norm
                            assemble_and_save_residual(working_mesh, self.results_saver, eta, res_name, i)

                self.results_saver.add_new_value("dofs", num_dofs)
                self.results_saver.save_function(eta_h_H10,           f"eta_h_H10_{str(i).zfill(2)}")
                self.results_saver.save_function(eta_h_L2,            f"eta_h_L2_{str(i).zfill(2)}")
                self.results_saver.save_function(phiV,                f"phi_V_{str(i).zfill(2)}")
                self.results_saver.save_function(v0,                  f"v0_{str(i).zfill(2)}")
                self.results_saver.save_function(uh,                  f"uh_{str(i).zfill(2)}")
                self.results_saver.save_function(wh,                  f"wh_{str(i).zfill(2)}")
                self.results_saver.save_function(correction_function, f"boundary_correction_{str(i).zfill(2)}")
                self.results_saver.save_mesh    (working_mesh,        f"mesh_{str(i).zfill(2)}")

            if self.exact_error:
                ref_degree = self.finite_element_degree + self.levelset_degree + 1
                expression_u_exact = self.exact_solution.expression
                phiFEM_solver.compute_exact_error(self.results_saver,
                                                  ref_degree= ref_degree,
                                                  expression_u_exact=expression_u_exact,
                                                  save_output=self.save_output)
                phiFEM_solver.compute_efficiency_coef(self.results_saver, norm="H10")
                phiFEM_solver.compute_efficiency_coef(self.results_saver, norm="L2")

            # Marking
            if i < self.iteration_number - 1:
                # Uniform refinement (Omega_h only)
                if self.refinement_method == "uniform":
                    working_mesh = dfx.mesh.refine(working_mesh)

                # Adaptive refinement
                if self.refinement_method in ["H10", "L2"]:
                    facets2ref = phiFEM_solver.marking()
                    working_mesh = dfx.mesh.refine(working_mesh, facets2ref)

            if self.save_output:
                self.results_saver.save_values("results.csv")
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