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

    if num_args == 3:
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

    if num_args == 3:
        dfuncdz   = jax.grad(func, argnums=2)
        d2funcdz2 = jax.grad(dfuncdz, argnums=2)

        vec_d2funcdz2 = jax.vmap(d2funcdz2, in_axes=inaxes)
    
        def nlap(x):
            return - vec_d2funcdx2(x[0], x[1], x[2]) - vec_d2funcdy2(x[0], x[1], x[2]) - vec_d2funcdz2(x[0], x[1], x[2])
    else:
        def nlap(x):
            return - vec_d2funcdx2(x[0], x[1]) - vec_d2funcdy2(x[0], x[1])
    return nlap

def compute_gradient(func):
    """Compute the gradient of a numpy function.

    Args:
        func (method): the method to compute the gradient of.
    
    Returns:
        grad (lambda): the gradient as a lambda function. 
    """
    num_args = func.__code__.co_argcount

    if num_args == 3:
        inaxes = (0,0,0) # Vectorize column by column
    else:
        inaxes = (0,0)
    
    dfuncdx = jax.grad(func, argnums=0)
    dfuncdy = jax.grad(func, argnums=1)
    vec_dfuncdx = jax.vmap(dfuncdx, in_axes=inaxes)
    vec_dfuncdy = jax.vmap(dfuncdy, in_axes=inaxes)

    if num_args == 3:
        dfuncdz = jax.grad(func, argnums=2)
        vec_dfuncdz = jax.vmap(dfuncdz, in_axes=inaxes)

        def grad(x):
            return [vec_dfuncdx(x[0], x[1], x[2]),
                    vec_dfuncdy(x[0], x[1], x[2]),
                    vec_dfuncdz(x[0], x[1], x[2])]
    else:
        def grad(x):
            return [vec_dfuncdx(x[0], x[1]),
                    vec_dfuncdy(x[0], x[1])]
    return grad