import argparse
import numpy as np
import os
from phiFEM.phifem.poisson_dirichlet import poisson_dirichlet_phiFEM, poisson_dirichlet_FEM

parent_dir = os.path.split(os.path.abspath(__file__))[0]

tilt_angle = np.pi/6.
shift = np.array([np.pi/32., np.pi/32.])

def rotation(angle, x):
    # Rotation matrix
    R = np.array([[ np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    rotated = R.dot(np.array(x))
    return rotated

def line(x, y, a, b, c):
    rotated = rotation(tilt_angle, [x, y])
    return a*rotated[0] + b*rotated[1] + np.full_like(x, c)

def expression_levelset(x):
    x_shift = x[0, :] - np.full_like(x[0, :], shift[0])
    y_shift = x[1, :] - np.full_like(x[1, :], shift[1])

    line_1 = line(x_shift, y_shift, -1.,  0.,   0.)
    line_2 = line(x_shift, y_shift,  0., -1.,   0.)
    line_3 = line(x_shift, y_shift,  1.,  0., -0.5)
    line_4 = line(x_shift, y_shift,  0.,  1., -0.5)
    line_5 = line(x_shift, y_shift,  0., -1., -0.5)
    line_6 = line(x_shift, y_shift, -1.,  0., -0.5)

    reentrant_corner = np.minimum(line_1, line_2)
    top_right_corner = np.maximum(line_3, line_4)
    corner           = np.maximum(reentrant_corner, top_right_corner)
    horizontal_leg   = np.maximum(corner, line_5)
    vertical_leg     = np.maximum(horizontal_leg, line_6)
    return vertical_leg

def expression_rhs(x):
    return np.ones_like(x[0, :])

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="L-shaped test case.",
                                     description="Run iterations of FEM or phiFEM with uniform or adaptive refinement.")

    parser.add_argument("solver", type=str, choices=["FEM", "phiFEM"])
    parser.add_argument("char_length", type=float)
    parser.add_argument("num_it", type=int)
    parser.add_argument("ref_mode", type=str, choices=["uniform", "H10", "L2"])
    parser.add_argument("--exact_error", default=False, action='store_true', help="Compute the exact errors.")
    args = parser.parse_args()
    solver = args.solver
    cl = args.char_length
    num_it = args.num_it
    ref_method = args.ref_mode
    compute_exact_errors = args.exact_error

    output_dir = os.path.join(parent_dir, "output_" + solver, ref_method)

    if solver=="phiFEM":
        poisson_dirichlet_phiFEM(cl,
                                 num_it,
                                 expression_levelset,
                                 parent_dir,
                                 expression_rhs=expression_rhs,
                                 bg_mesh_corners=[np.array([-1., -1.]),
                                                  np.array([ 1.,  1.])],
                                 ref_method=ref_method,
                                 compute_exact_error=compute_exact_errors)
    
    if solver=="FEM":
        point_1 = rotation(- tilt_angle, np.array([0.,  0.]) * 0.5) + shift
        point_2 = rotation(- tilt_angle, np.array([0., -1.]) * 0.5) + shift
        point_3 = rotation(- tilt_angle, np.array([1., -1.]) * 0.5) + shift
        point_4 = rotation(- tilt_angle, np.array([1.,  1.]) * 0.5) + shift
        point_5 = rotation(- tilt_angle, np.array([-1., 1.]) * 0.5) + shift
        point_6 = rotation(- tilt_angle, np.array([-1., 0.]) * 0.5) + shift

        geom_vertices = np.vstack([point_1, point_2, point_3, point_4, point_5, point_6]).T
        poisson_dirichlet_FEM(cl,
                              num_it,
                              expression_levelset,
                              parent_dir,
                              expression_rhs=expression_rhs,
                              quadrature_degree=4,
                              ref_method=ref_method,
                              geom_vertices=geom_vertices,
                              compute_exact_error=compute_exact_errors)