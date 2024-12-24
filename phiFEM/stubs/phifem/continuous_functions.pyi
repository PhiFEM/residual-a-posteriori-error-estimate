from basix.ufl import _ElementBase
from collections.abc import Callable
import dolfinx as dfx
import numpy as np
import numpy.typing as npt
from typing import Tuple

NDArrayTuple = Tuple[npt.NDArray[np.float64], ...]
NDArrayFunction = Callable[[NDArrayTuple], npt.NDArray[np.float64]]

class ContinuousFunction:
    """ Class to represent a continuous (in the sense of non-discrete) function."""

    def __init__(self, expression: NDArrayFunction_2D) -> None: ...
    
    def __call__(self, *args: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    
    def interpolate(self, FE_space: dfx.fem.FunctionSpace) -> dict[_ElementBase, dfx.fem.Function]: ...
            
class Levelset(ContinuousFunction):

    def exterior(self, t: float, padding: float =0.) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]: ...
    
    def interior(self, t: float, padding: float =0.) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]: ...
    
class ExactSolution(ContinuousFunction):

    def compute_negative_laplacian(self) -> None: ...