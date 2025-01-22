import argparse
import jax.numpy as jnp
import numpy as np
import os
from phiFEM.phifem.poisson_dirichlet import poisson_dirichlet_phiFEM, poisson_dirichlet_FEM

parent_dir = os.path.dirname(__file__)

tilt_angle = np.pi/6.
def rotation(angle, x):
    if x.shape[0] == 3:
        R = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                       [jnp.sin(angle),  jnp.cos(angle), 0],
                       [             0,               0, 1]])
    elif x.shape[0] == 2:
        R = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                       [jnp.sin(angle),  jnp.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(jnp.asarray(x))

def tilted_square(x):
    def fct(x):
        return jnp.sum(jnp.abs(rotation(-tilt_angle + jnp.pi/4., x)), axis=0)
    return fct(x) - np.sqrt(2.)/2.

def expression_levelset(x):
    def func(x):
        
    val = -jnp.cos(jnp.pi * rotation(-tilt_angle, x)[0, :]) * \
           jnp.cos(jnp.pi * rotation(-tilt_angle, x)[1, :])
    val = val.at[np.where(tilted_square(x) > 0.)].set(1.)
    return val

def expression_u_exact(x):
    return jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[0, :]) * \
           jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[1, :])

# Not required since jax will compute the negative laplacian of u_exact automatically but we add it since we know the analytical expression :)
def expression_rhs(x):
    return 8. * jnp.pi**2 * jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[0, :]) * \
                            jnp.sin(2. * jnp.pi * rotation(-tilt_angle, x)[1, :])

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Sine-sine test case.",
                                     description="Run iterations of FEM or phiFEM with uniform or adaptive refinement.")

    parser.add_argument("solver", 
                        type=str,
                        choices=["FEM", "phiFEM"],
                        help="Finite element solver.")
    parser.add_argument("char_length",
                        type=float,
                        help="Size of the initial mesh.")
    parser.add_argument("num_it",
                        type=int,
                        help="Number of refinement iterations.")
    parser.add_argument("ref_mode",
                        type=str,
                        choices=["uniform", "H10", "L2"],
                        help="Refinement strategy.")
    parser.add_argument("--exact_error",
                        default=False,
                        action='store_true',
                        help="Compute the exact errors.")

    args = parser.parse_args()
    solver               = args.solver
    cl                   = args.char_length
    num_it               = args.num_it
    ref_method           = args.ref_mode
    compute_exact_errors = args.exact_error

    if solver=="phiFEM":
        poisson_dirichlet_phiFEM(cl,
                                 num_it,
                                 expression_levelset,
                                 parent_dir,
                                 expression_u_exact=expression_u_exact,
                                 bbox_vertices=np.array([[-1., 1.],
                                                         [-1., 1.]]),
                                 ref_method=ref_method,
                                 compute_exact_error=compute_exact_errors)
    
    if solver=="FEM":
        point_1 = rotation(tilt_angle,
                            np.array([-0.5, -0.5]))
        point_2 = rotation(tilt_angle,
                            np.array([0.5,  -0.5]))
        point_3 = rotation(tilt_angle,
                            np.array([0.5,  0.5]))
        point_4 = rotation(tilt_angle,
                            np.array([-0.5, 0.5]))

        geom_vertices = np.vstack([point_1, point_2, point_3, point_4]).T
        poisson_dirichlet_FEM(cl,
                              num_it,
                              expression_levelset,
                              parent_dir,
                              expression_rhs=expression_rhs,
                              quadrature_degree=4,
                              ref_method=ref_method,
                              geom_vertices=geom_vertices,
                              compute_exact_error=compute_exact_errors)