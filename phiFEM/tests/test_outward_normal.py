import basix as bx
import dolfinx as dfx
from dolfinx.io import XDMFFile, gmshio
import dolfinx.plot as plot
from dolfinx.mesh import compute_midpoints
import matplotlib.pyplot as plt
from mpi4py import MPI
import meshio
import numpy as np
import pygmsh as pg
import pytest
import os
import ufl

from utils.compute_meshtags import tag_entities
from utils.classes import Levelset
from utils.mesh_scripts import (msh2xdmf_conversion_2D,
                                plot_mesh_tags,
                                reshape_facets_map,
                                compute_outward_normal)

parent_dir = os.path.dirname(__file__)

def create_mesh(mesh_path, lcar):
    with pg.geo.Geometry() as geom:
        # Points
        p1 = geom.add_point([0.,   0., 0.], lcar)
        p2 = geom.add_point([0.5,  0., 0.], lcar)
        p3 = geom.add_point([-0.5, 0., 0.], lcar)

        # Lines
        c1 = geom.add_circle_arc(p2, p1, p3)
        c2 = geom.add_circle_arc(p3, p1, p2)

        # Suface
        lloop = geom.add_curve_loop([c1, c2])
        surf = geom.add_plane_surface(lloop)

        mesh = geom.generate_mesh()

        mesh.points = mesh.points[:, :2]

        for cell in mesh.cells:
            if cell.type == 'triangle':
                triangle_cells = cell.data

        meshio.write(os.path.join(parent_dir, "tests_data", "disk.xdmf"),     
                     meshio.Mesh(points = mesh.points,
                                 cells={"triangle": triangle_cells}))

def rotation(angle, x):
    return (np.cos(angle)*x[0] + np.sin(angle)*x[1], -np.sin(angle)*x[0] + np.cos(angle)*x[1])

"""
Dara_n° = ("Data name", "mesh name", levelset object, "cells benchmark name", "facets benchmark name")
"""
data_1 = ("Circle radius 1", "disk", Levelset(lambda x, y: x**2 + y**2 - 0.125))
data_2 = ("Square", "square", Levelset(lambda x, y: np.exp(-1./(np.abs(rotation(np.pi/6. - np.pi/4., [x, y])[0])+np.abs(rotation(np.pi/6. - np.pi/4., [x, y])[1]))) - np.exp(-0.75)))

testdata = [data_1, data_2]

@pytest.mark.parametrize("data_name, mesh_name, levelset", testdata)
def test_outward_normal(data_name, mesh_name, levelset):
    mesh_path = os.path.join(parent_dir, "tests_data", mesh_name + ".xdmf")

    if not os.path.isfile(mesh_path):
        print(f"{mesh_path} not found, we create it.")
        create_mesh(mesh_path, 0.1)
    
    gdim = 2
    with XDMFFile(MPI.COMM_WORLD, os.path.join(parent_dir, "tests_data", "disk.xdmf"), "r") as fi:
        mesh = fi.read_mesh(name="Grid")
    
    cdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1

    cells_tags = tag_entities(mesh, levelset, cdim)
    facets_tags = tag_entities(mesh, levelset, fdim, cells_tags=cells_tags)
    w0 = compute_outward_normal(mesh, [cells_tags, facets_tags], levelset)

    W0 = w0.function_space

    mesh.topology.create_connectivity(fdim, 0)
    f2v_connect = mesh.topology.connectivity(fdim, 0)

    f2v_map = np.reshape(f2v_connect.array, (-1, 2))

    num_facets = mesh.topology.index_map(fdim).size_global
    points = mesh.geometry.x

    f2c_connect = mesh.topology.connectivity(fdim, cdim)
    f2c_map = reshape_facets_map(f2c_connect)
    mask = np.where(facets_tags.values == 4)
    f2c_map[mask]

    inner_prods = []
    for facet in facets_tags.indices[mask]:
        neighbor_inside_cell = np.intersect1d(f2c_map[facet], cells_tags.indices[np.where(cells_tags.values == 2)])
        dof_0 = dfx.fem.locate_dofs_topological(W0.sub(0), cdim, neighbor_inside_cell)
        dof_1 = dfx.fem.locate_dofs_topological(W0.sub(1), cdim, neighbor_inside_cell)
        verts = f2v_map[facet]
        vec_facet = [points[verts][0][0] - points[verts][1][0], points[verts][0][1] - points[verts][1][1]]
        val_normal = [w0.sub(0).x.array[dof_0][0], w0.sub(1).x.array[dof_1][0]]
        inner_pdct = np.inner(vec_facet, val_normal)
        
        # Check that the gradient from the levelset is orthogonal to the boundary facet
        assert np.isclose(inner_pdct, 0.)

        coords = mesh.geometry.x
        cell_vertices = mesh.topology.connectivity(cdim, 0).links(neighbor_inside_cell[0])
        cell_midpoint = coords[cell_vertices].mean(axis=0)
        facet_vertices = mesh.topology.connectivity(fdim, 0).links(facet)
        facet_midpoint = coords[facet_vertices].mean(axis=0)

        vec_midpoints = facet_midpoint - cell_midpoint

        # Check that the gradient from the levelset is pointing outward of Omega_h        
        assert np.greater(np.inner(val_normal, vec_midpoints[:-1]), 0.)

        # Check that the gradient is normalized
        norm = np.sqrt(np.inner(val_normal, val_normal))
        assert np.isclose(norm, 1.)

if __name__=="__main__":
    test_outward_normal("0", "disk", Levelset(lambda x, y: x**2 + y**2 - 0.125))