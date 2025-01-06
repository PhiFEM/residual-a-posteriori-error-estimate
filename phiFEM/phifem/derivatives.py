from collections.abc import Callable
import jax
import jax.numpy as jnp
from mypy_extensions import VarArg
import numpy as np
import numpy.typing as npt


NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | Callable[[VarArg(npt.NDArray[np.float64])], npt.NDArray[np.float64]]

def negative_laplacian(func: NDArrayFunction) -> NDArrayFunction:
    """Compute the negative laplacian of a (numpy) function of two or three arguments.

    Args:
        func: the method to compute the negative laplacian of.
    
    Returns:
        nlap: the negative laplacian function. 
    """
    def dummy(x, y, z):
        X = jnp.vstack((x, y, z))
        return func(X)[0]
    
    # Computation of the 2nd order derivatives
    second_derivatives = []
    for argnum in range(3):
        dfunc  = jax.grad(dummy, argnums=argnum)
        d2func = jax.grad(dfunc, argnums=argnum)
        vec_d2func = jax.vmap(d2func, in_axes=(0,0,0))
        second_derivatives.append(vec_d2func)

    def nlap(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        shape = X.shape
        val: npt.NDArray[np.float64]
        if shape[0] == 1:
            val = - sum([vec_d2func(X[0, :], np.zeros_like(X[0, :]), np.zeros_like(X[0, :])) for vec_d2func in second_derivatives])
        elif shape[0] == 2:
            val = - sum([vec_d2func(X[0, :], X[1, :], np.zeros_like(X[0, :])) for vec_d2func in second_derivatives])
        elif shape[0] == 3:
            val = - sum([vec_d2func(X[0, :], X[1, :], X[2, :]) for vec_d2func in second_derivatives])
        else:
            raise ValueError("The input (coordinates) dimension must be 3 at most.")
        return val
    
    return nlap