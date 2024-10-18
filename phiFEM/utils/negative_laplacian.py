import jax
import jax.numpy as jnp
import numpy as np

def negative_laplacian(func):
    """Compute the negative laplacian of a (numpy) function of two or three arguments.

    Args:
        func (method): the method to compute the negative laplacian of.
    
    Returns:
        nlap (lambda): the negative laplacian as a lambda function. 
    """
    num_args = func.__code__.co_argcount

    D3 = num_args == 3
    if D3:
        inaxes = (0,0,0) # Vectorize column by column
    else:
        inaxes = (0,0)

    # Computation of the 2nd order derivatives
    dfuncdx   = jax.grad(func,    argnums=0)
    d2funcdx2 = jax.grad(dfuncdx, argnums=0)
    dfuncdy   = jax.grad(func,    argnums=1)
    d2funcdy2 = jax.grad(dfuncdy, argnums=1)

    # Vectorization (important otherwise your computer's gonna blow up)
    vec_d2funcdx2 = jax.vmap(d2funcdx2, in_axes=inaxes)
    vec_d2funcdy2 = jax.vmap(d2funcdy2, in_axes=inaxes)

    if D3:
        dfuncdz   = jax.grad(func, argnums=2)
        d2funcdz2 = jax.grad(dfuncdz, argnums=2)

        vec_d2funcdz2 = jax.vmap(d2funcdz2, in_axes=inaxes)
    
        def nlap(x):
            return - vec_d2funcdx2(x[0], x[1], x[2]) - vec_d2funcdy2(x[0], x[1], x[2]) - vec_d2funcdz2(x[0], x[1], x[2])
    else:
        def nlap(x):
            return - vec_d2funcdx2(x[0], x[1]) - vec_d2funcdy2(x[0], x[1])
    return nlap