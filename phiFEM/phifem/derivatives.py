from collections.abc import Callable
import inspect
import jax
import numpy as np
import numpy.typing as npt
from typing import Tuple

NDArrayTuple = Tuple[npt.NDArray[np.float64], ...]
NDArrayFunction = Callable[[NDArrayTuple], npt.NDArray[np.float64]]

def negative_laplacian(func: NDArrayFunction) -> NDArrayFunction:
    """Compute the negative laplacian of a (numpy) function of two or three arguments.

    Args:
        func: the method to compute the negative laplacian of.
    
    Returns:
        nlap: the negative laplacian as a lambda function. 
    """
    num_args: int = len(inspect.signature(func).parameters)

    inaxes: int | tuple[int, int] | tuple[int, int, int]
    if num_args==1:
        inaxes = (0)
    elif num_args == 2:
        inaxes = (0,0)
    elif num_args == 3:
        inaxes = (0,0,0) # Vectorize column by column
    else:
        raise ValueError("Computation of negative laplacian is not implemented for functions of more than 3 variables.")

    # Computation of the 2nd order derivatives
    second_derivatives = []
    for argnum in range(num_args):
        dfunc  = jax.grad(func,  argnums=argnum)
        d2func = jax.grad(dfunc, argnums=argnum)
        vec_d2func = jax.vmap(d2func, in_axes=inaxes)
        second_derivatives.append(vec_d2func)

    def nlap(*args: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        val: npt.NDArray[np.float64]
        val = - sum([vec_d2func(*args) for vec_d2func in second_derivatives])
        return val
    
    return nlap