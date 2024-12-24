from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from typing import Tuple

NDArrayTuple = Tuple[npt.NDArray[np.float64], ...]
NDArrayFunction = Callable[[NDArrayTuple], npt.NDArray[np.float64]]


def print_fct(func: NDArrayFunction) -> None: ...
def negative_laplacian(func: NDArrayFunction) -> NDArrayFunction: ...