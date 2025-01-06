from collections.abc import Callable
import numpy as np
import numpy.typing as npt


NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def print_fct(func: NDArrayFunction) -> None: ...
def negative_laplacian(func: NDArrayFunction) -> NDArrayFunction: ...