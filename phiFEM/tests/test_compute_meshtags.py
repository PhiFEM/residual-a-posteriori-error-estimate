import basix as bx
import dolfinx as dfx
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
import numpy as np
import pytest
from utils.compute_meshtags import tag_entities
from utils.classes import Levelset
import os
# from utils.mesh_plot import plot_mesh_tags
# import matplotlib.pyplot as plt

"""
Dara_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("Circle radius 1", "disk", Levelset(lambda x: x[0]**2 + x[1]**2 - 0.125), "celltags_1", "facettags_1")

testdata = [data_1]

@pytest.mark.parametrize("data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name", testdata)
def test_compute_meshtags(data_name, mesh_name, levelset, cells_benchmark_name, facets_benchmark_name):
    cells_benchmark = np.loadtxt(os.path.abspath(os.path.join("tests", "tests_data", cells_benchmark_name + ".csv")), delimiter=" ")
    facets_benchmark = np.loadtxt(os.path.abspath(os.path.join("tests", "tests_data", facets_benchmark_name + ".csv")), delimiter=" ")

    with XDMFFile(MPI.COMM_WORLD, os.path.abspath(os.path.join("tests", "tests_data", mesh_name + ".xdmf")), "r") as fi:
        mesh = fi.read_mesh(name="Grid")
    
    cells_tags = tag_entities(mesh,
                              levelset,
                              2,
                              tag_interior=1,
                              tag_boundary=2)

    facets_tags = tag_entities(mesh,
                               levelset,
                               1,
                               tag_interior=1,
                               tag_boundary=2)
    
    # figure, ax = plt.subplots()
    # plot_mesh_tags(mesh, facets_tags, ax=ax)
    # plt.savefig("./test_facets.png")
    # figure, ax = plt.subplots()
    # plot_mesh_tags(mesh, cells_tags, ax=ax)
    # plt.savefig("./test_cells.png")

    assert np.all(cells_tags.indices == cells_benchmark[0,:])
    assert np.all(cells_tags.values  == cells_benchmark[1,:])
    assert np.all(facets_tags.indices == facets_benchmark[0,:])
    assert np.all(facets_tags.values  == facets_benchmark[1,:])