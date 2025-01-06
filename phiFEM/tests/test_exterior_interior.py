import matplotlib.pyplot as plt
import numpy as np
import os
from   phiFEM.phifem.continuous_functions import Levelset
import pytest

parent_dir = os.path.dirname(__file__)

def _plot_benchmarks(x, values_exterior, values_interior, data_name, expression_levelset):
    plt.figure()
    plt.title("Exterior")
    nx = 1000
    ny = 1000
    xs = np.linspace(-2., 2., nx)
    ys = np.linspace(-2., 2., ny)

    xx, yy = np.meshgrid(xs, ys)
    xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
    yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
    xxx = np.zeros((2, len(xx_rs)))
    xxx[0, :] = xx_rs
    xxx[1, :] = yy_rs
    zz_rs = expression_levelset(xxx)
    zz = zz_rs.reshape(xx.shape)

    plt.contour(xx, yy, zz, [0.], linewidths=0.5)
    colors = np.where(values_exterior, 'blue', 'red')

    plt.scatter(x[0, :], x[1, :], c=colors, label=['True', 'False'], s=5)

    # Add a legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='True')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False')
    plt.legend(handles=[blue_patch, red_patch])

    plt.savefig("exterior_" + data_name + ".svg", bbox_inches="tight")
    plt.figure()
    plt.title("Interior")
    nx = 1000
    ny = 1000
    xs = np.linspace(-2., 2., nx)
    ys = np.linspace(-2., 2., ny)

    xx, yy = np.meshgrid(xs, ys)
    xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
    yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
    xxx = np.zeros((2, len(xx_rs)))
    xxx[0, :] = xx_rs
    xxx[1, :] = yy_rs
    zz_rs = expression_levelset(xxx)
    zz = zz_rs.reshape(xx.shape)

    plt.contour(xx, yy, zz, [0.], linewidths=0.5)
    colors = np.where(values_interior, 'blue', 'red')

    plt.scatter(x[0, :], x[1, :], c=colors, label=['True', 'False'], s=5)

    # Add a legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='True')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False')
    plt.legend(handles=[blue_patch, red_patch])
    plt.savefig("interior_" + data_name + ".svg", bbox_inches="tight")

tilt_angle = np.pi/6.
def rotation(angle, x):
    if x.shape[0] == 3:
        R = np.array([[ np.cos(angle), np.sin(angle), 0],
                      [-np.sin(angle), np.cos(angle), 0],
                      [             0,             0, 1]])
    elif x.shape[0] == 2:
        R = np.array([[ np.cos(angle), np.sin(angle)],
                      [-np.sin(angle), np.cos(angle)]])
    else:
        raise ValueError("Incompatible argument dimension.")
    return R.dot(np.asarray(x))

def expression_circle(x):
    return np.square(x[0, :]) + np.square(x[1, :]) - np.ones_like(x[0, :])

def expression_tilted_square(x):
    def fct(x):
        return np.sum(np.abs(rotation(tilt_angle - np.pi/4., x)), axis=0)
    return fct(x) - np.sqrt(2.)/2.

"""
Data_n° = ("Data name", "mesh name", levelset object)
"""
data_1 = ("circle",        Levelset(expression_circle))
data_2 = ("tilted_square", Levelset(expression_tilted_square))

testdata = [data_1, data_2]

@pytest.mark.parametrize("data_name, levelset", testdata)
def test_exterior_interior(data_name, levelset, save_as_benchmark=False):
    nx = 100
    ny = 100
    xs = np.linspace(-2., 2., nx)
    ys = np.linspace(-2., 2., ny)
    xx, yy = np.meshgrid(xs, ys)
    xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
    yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
    x = np.zeros((2, len(xx_rs)))
    x[0, :] = xx_rs
    x[1, :] = yy_rs

    exterior = levelset.exterior(0.)
    interior = levelset.interior(0.)

    values_exterior = exterior(x)
    values_interior = interior(x)

    int_bench_name = "interior_" + data_name
    ext_bench_name = "exterior_" + data_name
    if save_as_benchmark:
        _plot_benchmarks(x, values_exterior, values_interior, data_name, levelset.expression)
        np.savetxt(os.path.join(parent_dir, "tests_data", ext_bench_name + ".csv"), values_exterior, delimiter=" ", newline="\n")
        np.savetxt(os.path.join(parent_dir, "tests_data", int_bench_name + ".csv"), values_interior, delimiter=" ", newline="\n")
    else:
        try:
            values_ext_bench = np.loadtxt(os.path.join(parent_dir, "tests_data", ext_bench_name + ".csv"), delimiter=" ")
        except FileNotFoundError:
            raise FileNotFoundError(f"{ext_bench_name}.csv not found, have you generated the benchmark ?")
        try:
            values_int_bench = np.loadtxt(os.path.join(parent_dir, "tests_data", int_bench_name + ".csv"), delimiter=" ")
        except FileNotFoundError:
            raise FileNotFoundError(f"{int_bench_name}.csv not found, have you generated the benchmark ?")
    
        assert np.all(values_exterior == values_ext_bench)
        assert np.all(values_interior == values_int_bench)


if __name__=="__main__":
    print("Circle")
    data_name = "circle"

    levelset = Levelset(expression_circle)
    test_exterior_interior(data_name, levelset, save_as_benchmark=True)

    print("Tilted square")
    data_name = "tilted_square"
        
    levelset = Levelset(expression_tilted_square)

    test_exterior_interior(data_name, levelset, save_as_benchmark=True)
