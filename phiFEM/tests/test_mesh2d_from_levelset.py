import itertools
import numpy as np
import pytest
from phiFEM.src.mesh_scripts import mesh2d_from_levelset
from phiFEM.src.continuous_functions import Levelset

"""
Data_n° = ("Fct formula", function of var x, y, geom_vertices (np.array shape (N,2)), bbox (np.array shape (2,2)))
"""
# Tilted square
def expression_1(x, y):
    return np.abs(x) + np.abs(y) - np.ones_like(x)
geom_vertices_1 = np.array([[1.,  0.], [0.,  1.], [-1., 0.], [0., -1.]])
bbox_1 = np.array([[-1.1, 1.1],
                   [-1.1, 1.1]])

# Unit circle
def expression_2(x, y):
    return x**2 + y**2 - np.ones_like(x)
bbox_2 = np.array([[-1.1, 1.1],
                   [-1.1, 1.1]])

# Tilted L-shaped
def expression_3(x, y):
    x = x - np.full_like(x, np.pi/32.)
    y = y - np.full_like(y, np.pi/32.)

    angle = np.pi/6. + np.pi/2.
    R = np.array([[ np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    rotated = R.dot(np.array([x, y]))

    reetrant_corner = np.minimum(-1. * rotated[0],
                                 -1. * rotated[1])
    top_right_corner = np.maximum(rotated[0] - np.full_like(x, 0.5),
                                  rotated[1] - np.full_like(x, 0.5))
    corner = np.maximum(reetrant_corner,
                        top_right_corner)
    horizontal_leg = np.maximum(corner,
                                -rotated[1] - np.full_like(x, 0.5))
    vertical_leg = np.maximum(horizontal_leg,
                              -rotated[0] - np.full_like(x, 0.5))
    return vertical_leg
geom_vertices_3 = np.array([[0.,    0.],
                            [0.,  -0.5],
                            [0.5, -0.5],
                            [0.5,  0.5],
                            [-0.5, 0.5],
                            [-0.5,  0.]])
bbox_3 = np.array([[-0.9, 0.3],
                   [-0.7, 0.8]])

data_1  = ("|x|+|y|-1",
           expression_1,
           geom_vertices_1,
           bbox_1)
data_2  = ("x^2+y^2-1",
           expression_2,
           None,
           bbox_2)
data_3  = ("l-shaped",
           expression_3,
           geom_vertices_3,
           bbox_3)

testdata = [data_1, data_2, data_3]

@pytest.mark.parametrize("lc,data", itertools.product([0.1, 0.01, 0.005], testdata))
def test_mesh2d_from_levelset(lc, data):
    data_name     = data[0]
    levelset      = Levelset(data[1])
    geom_vertices = data[2]
    bbox          = data[3]

    boundary_vertices = mesh2d_from_levelset(lc,
                                             levelset,
                                             level=0.,
                                             bbox=bbox,
                                             geom_vertices=geom_vertices,
                                             output_dir=None)

    try:
        # This test assumes that the levelset behaves like the distance to the boundary.
        err_max_to_boundary = np.max(np.abs(levelset(boundary_vertices[:, 0], boundary_vertices[:, 1])))
        assert np.isclose(err_max_to_boundary, 0.)
    except AssertionError:
        print(f"Error data {data_name}. Maximum error to the boundary:", err_max_to_boundary)