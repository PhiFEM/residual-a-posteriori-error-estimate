import basix as bx
import dolfinx as dfx
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
import numpy as np
import pytest
from utils.compute_meshtags import tag_entities
from utils.classes import Levelset
import os


"""
Dara_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("Circle radius 1", "disk", Levelset(lambda x: x[0]**2 + x[1]**2 - 0.125), "celltags_1", "facettags_1")

testdata = [data_1]

parent_dir = os.path.dirname(__file__)

@pytest.mark.parametrize("data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name", testdata)
def test_compute_meshtags(data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name):
    cells_benchmark = np.loadtxt(os.path.join(parent_dir, "tests_data", cells_benchmark_name + ".csv"), delimiter=" ")
    facets_benchmark = np.loadtxt(os.path.join(parent_dir, "tests_data", facets_benchmark_name + ".csv"), delimiter=" ")

    with XDMFFile(MPI.COMM_WORLD, os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf"), "r") as fi:
        mesh = fi.read_mesh(name="Grid")
    
    # Test computation of cells tags
    cells_tags = tag_entities(mesh,
                              levelset,
                              2)

    assert np.all(cells_tags.indices == cells_benchmark[0,:])
    assert np.all(cells_tags.values  == cells_benchmark[1,:])

    # Test computation of facets tags when cells tags are provided
    facets_tags = tag_entities(mesh,
                               levelset,
                               1,
                               cells_tags=cells_tags)

    assert np.all(facets_tags.indices == facets_benchmark[0,:])
    assert np.all(facets_tags.values  == facets_benchmark[1,:])

    # Test computation of facets tags when cells tags are not provided
    facets_tags = tag_entities(mesh,
                               levelset,
                               1)

    assert np.all(facets_tags.indices == facets_benchmark[0,:])
    assert np.all(facets_tags.values  == facets_benchmark[1,:])

if __name__=="__main__":
    # For debugging purpose only
    from utils.mesh_plot import plot_mesh_tags
    import matplotlib.pyplot as plt

    with XDMFFile(MPI.COMM_WORLD, os.path.join(parent_dir, "tests_data", "disk.xdmf"), "r") as fi:
        mesh = fi.read_mesh(name="Grid")
    
    levelset = Levelset(lambda x: x[0]**2 + x[1]**2 - 0.125)

    cells_tags = tag_entities(mesh,
                              levelset,
                              2)

    facets_tags = tag_entities(mesh,
                               levelset,
                               1,
                               cells_tags=cells_tags)

    figure, ax = plt.subplots()
    plot_mesh_tags(mesh, facets_tags, ax=ax)
    circle = plt.Circle((0,0), np.sqrt(0.125), color="black", fill=False, lw=1)
    ax.add_patch(circle)
    plt.savefig("./test_facets.png")
    figure, ax = plt.subplots()
    plot_mesh_tags(mesh, cells_tags, ax=ax)
    circle = plt.Circle((0,0), np.sqrt(0.125), color="black", fill=False, lw=1)
    ax.add_patch(circle)
    plt.savefig("./test_cells.png")
