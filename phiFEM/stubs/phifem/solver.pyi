from   basix.ufl import element, _ElementBase
from   collections.abc import Callable
import dolfinx as dfx
from   dolfinx.mesh import Mesh, MeshTags
from   dolfinx.fem.petsc import assemble_matrix, assemble_vector
from   dolfinx.fem import Form, Function, FunctionSpace, DirichletBC
from   dolfinx.io import XDMFFile
from   mpi4py import MPI
import numpy as np
import numpy.typing as npt
import os
from   os import PathLike
from   petsc4py.PETSc import Options, KSP
from petsc4py.PETSc import Mat as PETSc_Mat
from petsc4py.PETSc import Vec as PETSc_Vec
from   typing import Any, Tuple
import ufl
from   ufl import inner, jump, grad, div, avg
from   ufl.classes import Measure

from phifem.compute_meshtags import tag_entities
from phifem.continuous_functions import ContinuousFunction, ExactSolution, Levelset
from phifem.mesh_scripts import compute_outward_normal
from phifem.saver import ResultsSaver

NDArrayTuple = Tuple[npt.NDArray[np.float64], ...]
NDArrayFunction = Callable[[NDArrayTuple], npt.NDArray[np.float64]]

class GenericSolver:
    """ Class representing a generic solver."""

    def __init__(self,
                 mesh: Mesh,
                 FE_element: _ElementBase,
                 PETSc_solver: KSP,
                 ref_strat: str = "uniform",
                 num_step: int = 0,
                 save_output: bool = True) -> None: ...
    
    def set_source_term(self, source_term: ContinuousFunction) -> None: ...
    
    def assemble(self) -> None: ...
    
    def print(self, str2print: str) -> None: ...
    
    def solve(self) -> None: ...
    
    def compute_exact_error(self,
                            results_saver: ResultsSaver,
                            expression_u_exact: NDArrayFunction | None = None,
                            save_output: bool = True,
                            extra_ref: int = 1,
                            ref_degree: int = 2,
                            padding: float = 1.e-14,
                            reference_mesh_path: PathLike | None = None): ...

    def marking(self, theta: float = 0.3) -> npt.NDArray[np.float64]: ...

class PhiFEMSolver(GenericSolver):

    def __init__(self,
                 bg_mesh: Mesh,
                 FE_element: _ElementBase,
                 PETSc_solver: KSP,
                 ref_strat: str = "uniform",
                 levelset_element: _ElementBase | None = None,
                 num_step: int = 0,
                 save_output: bool = True) -> None: ...

    def _compute_normal(self, mesh: Mesh) -> Function: ...

    def _transfer_cells_tags(self, cmap: npt.NDArray[Any]) -> None: ...

    def set_levelset(self, levelset: Levelset) -> None: ...

    def compute_tags(self, padding: bool = False, plot: bool = False) -> None: ...
    
    def set_variational_formulation(self,
                                    sigma: float = 1.,
                                    quadrature_degree: int | None = None) -> Tuple[Function, Measure, Measure, int]: ...
    
    def solve(self) -> None: ...
    
    def estimate_residual(self,
                          V0: FunctionSpace | None = None,
                          quadrature_degree: int | None = None,
                          boundary_term: bool = False) -> None: ...

class FEMSolver(GenericSolver):

    def __init__(self,
                 mesh: Mesh,
                 FE_element: _ElementBase,
                 PETSc_solver: KSP,
                 ref_strat: str = "uniform",
                 num_step: int = 0,
                 save_output: bool = True) -> None: ...
    
    def set_boundary_conditions(self,
                                bcs: list[DirichletBC]) -> None: ...
    
    def set_variational_formulation(self, quadrature_degree: int | None = None) -> int: ...

    def estimate_residual(self,
                          V0: FunctionSpace | None = None,
                          quadrature_degree: int | None = None) -> None: ...