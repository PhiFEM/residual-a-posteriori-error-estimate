import argparse
import jax.numpy as jnp
import numpy as np
import os
from phiFEM.src.poisson_dirichlet import poisson_dirichlet_phiFEM, poisson_dirichlet_FEM

parent_dir = os.path.dirname(__file__)

tilt_angle = np.pi/6.
def rotation(angle, x):
    R = jnp.array([[ jnp.cos(angle), jnp.sin(angle)],
                   [-jnp.sin(angle), jnp.cos(angle)]])
    return R.dot(jnp.asarray(x))

def expression_levelset(x, y):
    def fct(x, y):
        return jnp.sum(jnp.abs(rotation(tilt_angle - jnp.pi/4., [x, y])), axis=0)
    return fct(x, y) - np.sqrt(2.)/2.

def expression_u_exact(x, y):
    return jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[0]) * \
           jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[1])

# Not required since jax will compute the negative laplacian of u_exact automatically but we add it since we know the analytical expression :)
def expression_rhs(x, y):
    return 8. * jnp.pi * jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[0]) * \
                         jnp.sin(2. * jnp.pi * rotation(tilt_angle, [x, y])[1])

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Sine-sine test case.",
                                     description="Run iterations of FEM or phiFEM with uniform or adaptive refinement.")

    parser.add_argument("solver", type=str, choices=["FEM", "phiFEM"], help="Finite element solver.")
    parser.add_argument("char_length", type=float, help="Size of the initial mesh.")
    parser.add_argument("num_it", type=int, help="Number of refinement iterations.")
    parser.add_argument("ref_mode", type=str, choices=["uniform", "H10", "L2"], help="Refinement strategy.")
    parser.add_argument("--exact_error", default=False, action='store_true', help="Compute the exact errors.")
    args = parser.parse_args()
    solver = args.solver
    cl = args.char_length
    num_it = args.num_it
    ref_method = args.ref_mode
    compute_exact_errors = args.exact_error

    if solver=="phiFEM":
        poisson_dirichlet_phiFEM(cl,
                                 num_it,
                                 expression_levelset,
                                 parent_dir,
                                 expression_rhs=expression_rhs,
                                 expression_u_exact=expression_u_exact,
                                 bg_mesh_corners=[np.array([-1., -1.]),
                                                  np.array([1., 1.])],
                                 ref_method=ref_method,
                                 compute_exact_error=compute_exact_errors)
    
    if solver=="FEM":
        point_1 = rotation(- tilt_angle - np.pi/4.,
                            np.array([0., -1.]) * np.sqrt(2.)/2.)
        point_2 = rotation(- tilt_angle - np.pi/4.,
                            np.array([1.,  0.]) * np.sqrt(2.)/2.)
        point_3 = rotation(- tilt_angle - np.pi/4.,
                            np.array([0.,  1.]) * np.sqrt(2.)/2.)
        point_4 = rotation(- tilt_angle - np.pi/4.,
                            np.array([-1., 0.]) * np.sqrt(2.)/2.)

        geom_vertices = np.vstack([point_1, point_2, point_3, point_4])
        poisson_dirichlet_FEM(cl,
                              num_it,
                              expression_levelset,
                              parent_dir,
                              expression_rhs=expression_rhs,
                              expression_u_exact=expression_u_exact,
                              quadrature_degree=4,
                              ref_method=ref_method,
                              geom_vertices=geom_vertices,
                              compute_exact_error=compute_exact_errors)