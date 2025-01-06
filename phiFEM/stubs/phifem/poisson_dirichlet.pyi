from   collections.abc import Callable
import numpy as np
import numpy.typing as npt
from   os import PathLike
from typing import Tuple


NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

def poisson_dirichlet_phiFEM(cl: float,
                             max_it: int,
                             expression_levelset: NDArrayFunction,
                             source_dir: PathLike,
                             expression_rhs: NDArrayFunction | None = None,
                             expression_u_exact: NDArrayFunction | None = None,
                             bg_mesh_corners: npt.NDArray[np.float64] = np.array([[0., 0.],
                                                                                  [1., 1.]]),
                             quadrature_degree: int = 4,
                             sigma_D: float = 1.,
                             save_output: bool = True,
                             ref_method: str = "uniform",
                             compute_exact_error: bool = False) -> None: ...
    
def poisson_dirichlet_FEM(cl: float,
                          max_it: int,
                          expression_levelset: NDArrayFunction,
                          source_dir: PathLike,
                          expression_rhs: NDArrayFunction | None = None,
                          expression_u_exact: NDArrayFunction | None = None,
                          quadrature_degree: int = 4,
                          save_output: bool = True,
                          ref_method: str = "uniform",
                          compute_exact_error: bool = False,
                          geom_vertices: npt.NDArray[np.float64] | None = None): ...