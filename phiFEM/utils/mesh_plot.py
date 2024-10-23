# Snippet stolen from https://github.com/multiphenics/multiphenicsx/blob/main/tutorials/07_understanding_restrictions/tutorial_understanding_restrictions.ipynb...
# ...and butchered so that we can pass a mesh_tags with more than 2 different tags.
# TODO: add more line styles for the moment it's not very colorblind friendly.

import matplotlib as mpl
from matplotlib import colormaps as cm
import mpl_toolkits.axes_grid1
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

def plot_mesh_tags(mesh, mesh_tags, ax = None):
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
        for c in range(num_cells):
            if c in mesh_tags.indices:
                cell_colors[c] = mesh_tags.values[np.where(mesh_tags.indices == c)][0]
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
        linestyles = "solid"
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
        mappable: mpl.collections.Collection = mpl.collections.LineCollection(  # type: ignore[no-redef]
            lines, cmap="tab10", norm=norm, colors=lines_colors_as_str, linestyles=lines_linestyles)
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(mappable)
        ax.autoscale()
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mappable, cax=cax, boundaries=cmap_bounds, ticks=cmap_bounds)
    return ax