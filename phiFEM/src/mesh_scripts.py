# Snippet stolen from https://github.com/multiphenics/multiphenicsx/blob/main/tutorials/07_understanding_restrictions/tutorial_understanding_restrictions.ipynb...
# ...and butchered so that we can pass a mesh_tags with more than 2 different tags.
# TODO: add more line styles for the moment it's not very colorblind friendly.

from basix.ufl import element
import dolfinx as dfx
import matplotlib as mpl
from matplotlib import colormaps as cm
import mpl_toolkits.axes_grid1
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import numpy as np
import os
import pygmsh
import ufl
from ufl import inner, grad
from lxml import etree

def mesh2d_from_levelset(lc, levelset, level=0., bbox=np.array([[-1., 1.], [-1., 1.]]), geom_vertices=None, output_dir=None, file_name="conforming_mesh"):
    """ Generate a 2D conforming mesh from a levelset function and saves it as an xdmf mesh.

    Args:
        lc:            characteristic length of the mesh.
        levelset:      levelset function (as a Levelset object).
        level:         the level of the isoline (default: 0.).
        bbox:          bounding box of the isoline (default: np.array([[-1., 1.], [-1., 1.]])).
        geom_vertices: specific vertices to be added to the isoline (e.g. vertices of the geometry.)
        output_dir:    directory path where the mesh is saved. If None the mesh is not saved.

    Returns:
        The coordinates of the boundary vertices.
    """

    x = np.arange(bbox[0,0], bbox[0,1], step=lc/np.sqrt(2.), dtype=np.float64)
    y = np.arange(bbox[1,0], bbox[1,1], step=lc/np.sqrt(2.), dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    X_flat, Y_flat = X.flatten(), Y.flatten()
    Z_flat = levelset(X_flat, Y_flat)
    Z = np.reshape(Z_flat, X.shape)

    c = plt.contour(X, Y, Z, [level])
    boundary_vertices = c.get_paths()[0].vertices
    if geom_vertices is not None:
        boundary_vertices = np.vstack(geom_vertices)
    
    with pygmsh.geo.Geometry() as geom:
        # The boundary vertices are correctly ordered by matplotlib.
        geom.add_polygon(boundary_vertices, mesh_size=lc)
        # http://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
        # algorithm=9 for structured mesh (packing of parallelograms)
        mesh = geom.generate_mesh(dim=2, algorithm=6)

    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangular_cells = [("triangle", cell_block.data)]

    if output_dir is not None:
        meshio.write_points_cells(os.path.join(output_dir, f"{file_name}.xdmf"), mesh.points, triangular_cells)
    
        # meshio and dolfinx use incompatible Grid names ("Grid" for meshio and "mesh" for dolfinx)
        # the lines below change the Grid name from "Grid" to "mesh" to ensure the compatibility between meshio and dolfinx.
        tree = etree.parse(os.path.join(output_dir, f"{file_name}.xdmf"))
        root = tree.getroot()

        for grid in root.findall(".//Grid"):
            grid.set("Name", "mesh")
        
        tree.write(os.path.join(output_dir, f"{file_name}.xdmf"), pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return boundary_vertices

def compute_outward_normal(mesh, levelset):
    # This function is used to define the unit outward pointing normal to Gamma_h
    CG1Element = element("Lagrange", mesh.topology.cell_name(), 1)
    V = dfx.fem.functionspace(mesh, CG1Element)
    DG0VecElement = element("DG", mesh.topology.cell_name(), 0, shape=(mesh.topology.dim,))
    W0 = dfx.fem.functionspace(mesh, DG0VecElement)
    ext = dfx.fem.Function(V)
    ext.interpolate(levelset.exterior(0.))

    # Compute the unit outwards normal, but the scaling might create NaN where grad(ext) = 0
    normal_Omega_h = grad(ext) / (ufl.sqrt(inner(grad(ext), grad(ext))))

    # In order to remove the eventual NaNs, we interpolate into a vector functions space and enforce the values of the gradient to 0. in the cells that are not cut
    w0 = dfx.fem.Function(W0)
    w0.sub(0).interpolate(dfx.fem.Expression(normal_Omega_h[0], W0.sub(0).element.interpolation_points()))
    w0.sub(1).interpolate(dfx.fem.Expression(normal_Omega_h[1], W0.sub(1).element.interpolation_points()))

    w0.sub(0).x.array[:] = np.nan_to_num(w0.sub(0).x.array, nan=0.0)
    w0.sub(1).x.array[:] = np.nan_to_num(w0.sub(1).x.array, nan=0.0)
    return w0

def reshape_facets_map(f2c_connect):
    f2c_array = f2c_connect.array
    num_cells_per_facet = np.diff(f2c_connect.offsets)
    max_cells_per_facet = num_cells_per_facet.max()
    f2c_map = -np.ones((len(f2c_connect.offsets) - 1, max_cells_per_facet), dtype=int)

    # Mask to select the boundary facets
    mask = np.where(num_cells_per_facet == 1)
    f2c_map[mask, 0] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    f2c_map[mask, 1] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    # Mask to select the interior facets
    mask = np.where(num_cells_per_facet == 2)
    f2c_map[mask, 0] = f2c_array[num_cells_per_facet.cumsum()[mask] - 2]
    f2c_map[mask, 1] = f2c_array[num_cells_per_facet.cumsum()[mask] - 1]
    return f2c_map

def msh2xdmf_conversion_2D(msh_file_path, cell_type, prune_z=False):
    mesh = meshio.read(msh_file_path)
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    triangle_mesh = meshio.Mesh(points=points,
                                cells={cell_type: cells},
                                cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    path, name_ext = os.path.split(msh_file_path)
    name = os.path.splitext(name_ext)[0]
    meshio.write(os.path.join(path, name, ".xdmf"), triangle_mesh)

def plot_mesh_tags(mesh, mesh_tags, ax = None, display_indices=False, expression_levelset=None):
    """Plot a mesh tags object on the provied (or, if None, the current) axes object."""
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    points = mesh.geometry.x
    cmap = [cm.get_cmap('tab10')(i / 10.) for i in range(10)]
    cmap_bounds = np.arange(10)
    norm = mpl.colors.BoundaryNorm(cmap_bounds, len(cmap))
    assert mesh_tags.dim in (mesh.topology.dim, mesh.topology.dim - 1)
    cells_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cells_map.size_local + cells_map.num_ghosts
    if mesh_tags.dim == mesh.topology.dim:
        cells = mesh.geometry.dofmap
        tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
        cell_colors = np.zeros((cells.shape[0], ))
        if display_indices:
            tdim = mesh.topology.dim
            connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
            vertex_map = {
                topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
                    connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
            }
        for c in range(num_cells):
            if c in mesh_tags.indices:
                cell_colors[c] = mesh_tags.values[np.where(mesh_tags.indices == c)][0]
                if display_indices:
                    vertices = [vertex_map[v] for v in connectivity_cells_to_vertices.links(c)]
                    midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
                    ax.text(midpoint[0], midpoint[1], f"{c}", horizontalalignment="center", verticalalignment="center", fontsize=6)
            else:
                cell_colors[c] = 0
        # cell_colors[mesh_tags.indices[mesh_tags.values != 0]] = 1
        mappable: mpl.collections.Collection = ax.tripcolor(
            tria, cell_colors, edgecolor="k", cmap="tab10", norm=norm)
    elif mesh_tags.dim == mesh.topology.dim - 1:
        tdim = mesh.topology.dim
        connectivity_cells_to_facets = mesh.topology.connectivity(tdim, tdim - 1)
        connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
        connectivity_facets_to_vertices = mesh.topology.connectivity(tdim - 1, 0)
        vertex_map = {
            topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
                connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
        }
        lines = list()
        lines_colors_as_int = list()
        lines_colors_as_str = list()
        lines_linestyles = list()
        for c in range(num_cells):
            facets = connectivity_cells_to_facets.links(c)
            for f in facets:
                if f in mesh_tags.indices:
                    value_f = mesh_tags.values[np.where(mesh_tags.indices == f)][0]
                else:
                    value_f = 0
                vertices = [vertex_map[v] for v in connectivity_facets_to_vertices.links(f)]
                lines.append(points[vertices][:, :2])
                lines_colors_as_int.append(value_f)
                lines_colors_as_str.append(cmap[value_f])
                lines_linestyles.append("solid")
                if display_indices:
                    midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
                    ax.text(midpoint[0], midpoint[1], f"{f}", horizontalalignment="center", verticalalignment="center", fontsize=6)
        mappable: mpl.collections.Collection = mpl.collections.LineCollection(  # type: ignore[no-redef]
            lines, cmap="tab10", norm=norm, colors=lines_colors_as_str, linestyles=lines_linestyles, linewidth=0.2)
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(mappable)
        ax.autoscale()
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mappable, cax=cax, boundaries=cmap_bounds, ticks=cmap_bounds)

    if expression_levelset is not None:
        x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
        y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
        nx = 1000
        ny = 1000
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)

        xx, yy = np.meshgrid(xs, ys)
        xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
        yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
        zz_rs = expression_levelset(xx_rs, yy_rs)
        zz = zz_rs.reshape(xx.shape)

        ax.contour(xx, yy, zz, [0.], linewidths=0.2)
    return ax

def compute_facets_to_refine(mesh, facets_tags):
    cdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1
    Gamma_h_facets = facets_tags.indices[np.where(facets_tags.values == 4)]

    mesh.topology.create_connectivity(fdim, cdim)
    mesh.topology.create_connectivity(cdim, fdim)

    f2c_connect = mesh.topology.connectivity(fdim, cdim)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)

    num_facets_per_cell = len(c2f_connect.links(0))

    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
    f2c_map = reshape_facets_map(f2c_connect)
    cells_connected_to_Gamma_h = np.unique(np.argsort(np.ndarray.flatten(f2c_map[Gamma_h_facets])))
    facets_connected_to_Gamma_h = np.unique(np.argsort(np.ndarray.flatten(c2f_map[cells_connected_to_Gamma_h])))
    facets_Omega_h = facets_tags.indices[np.where(np.logical_or(facets_tags.values == 1, facets_tags.values == 2, facets_tags.values == 4))]
    return np.union1d(facets_Omega_h, facets_connected_to_Gamma_h)