from   basix.ufl import element
from   collections.abc import Callable
from   contourpy import contour_generator
import dolfinx as dfx
from   dolfinx.cpp.graph import AdjacencyList_int32
from   dolfinx.mesh import Mesh, MeshTags
from   dolfinx.fem import Function
from   mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.collections as mpl_collections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import numpy as np
import numpy.typing as npt
import os
from   os import PathLike
import pygmsh
from   typing import cast
import ufl
from   ufl import inner, grad
from   phiFEM.phifem.utils import immutable
from   lxml import etree
from   phiFEM.phifem.continuous_functions import Levelset

PathStr = PathLike[str] | str

NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

@immutable("geom_vertices")
def mesh2d_from_levelset(lc: float,
                         levelset: Levelset,
                         level:float = 0.,
                         bbox: npt.NDArray[np.float64] = np.array([[-1., 1.], [-1., 1.]]),
                         geom_vertices: npt.NDArray[np.float64] | None = None,
                         output_dir: PathStr | None = None,
                         file_name: str ="conforming_mesh") -> npt.NDArray[np.float64]:
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

    # TODO: is there a way to combine geom_vertices and contour generated vertices ?
    boundary_vertices: npt.NDArray[np.float64]
    if geom_vertices is None:
        x = np.arange(bbox[0,0], bbox[1,0], step=lc/np.sqrt(2.), dtype=np.float64)
        y = np.arange(bbox[0,1], bbox[1,1], step=lc/np.sqrt(2.), dtype=np.float64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        X_flat, Y_flat = X.flatten(), Y.flatten()
        arr = np.vstack([X_flat, Y_flat])
        Z_flat = levelset(arr)
        Z = np.reshape(Z_flat, X.shape)
        cg = contour_generator(x=X, y=Y, z=Z, name="threaded")
        boundary_vertices = cast(npt.NDArray[np.float64], cg.lines(0.)[0]).T
    else:
        boundary_vertices = geom_vertices
    
    if boundary_vertices.shape[0] == 1:
        boundary_vertices = np.vstack((boundary_vertices, np.zeros_like(boundary_vertices), np.zeros_like(boundary_vertices)))
    elif boundary_vertices.shape[0] == 2:
        boundary_vertices = np.vstack((boundary_vertices, np.zeros_like(boundary_vertices[0, :])))
    elif boundary_vertices.shape[0] == 3:
        boundary_vertices = boundary_vertices
    else:
        raise ValueError("The geometry vertices must have at most 3 coordinates, not more.")
    
    with pygmsh.geo.Geometry() as geom:
        # The boundary vertices are correctly ordered by matplotlib.
        geom.add_polygon(boundary_vertices.T, mesh_size=lc)
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

def compute_outward_normal(mesh: Mesh, levelset: Levelset) -> Function:
    """ Compute the outward normal to Omega_h.

    Args:
        mesh: the mesh on which the levelset is discretized.
        levelset: the levelset defining Omega_h.
    
    Returns:
        w0: the vector field defining the outward normal.
    """
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

def reshape_facets_map(f2c_connect: AdjacencyList_int32) -> npt.NDArray[np.int32]:
    """ Reshape the facets-to-cells indices mapping.

    Args:
        f2c_connect: the facets-to-cells connectivity.
    
    Returns:
        The facets-to-cells mapping as a ndarray.
    """
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

# Snippet stolen from https://github.com/multiphenics/multiphenicsx/blob/main/tutorials/07_understanding_restrictions/tutorial_understanding_restrictions.ipynb...
# ...and butchered so that we can pass a mesh_tags with more than 2 different tags and display the cells and/or facets indices.
# TODO: add more line styles for the moment it's not very colorblind friendly.
# TODO: plot_mesh_tags is broken...
def plot_mesh_tags(
    mesh: Mesh,
    mesh_tags: MeshTags,
    ax: plt.Axes | None = None,
    display_indices: bool = False,
    expression_levelset: Callable[..., np.ndarray] | None = None
) -> plt.Axes:
    """Plot a mesh tags object on the provided (or, if None, the current) axes object.
    
    Args:
        mesh: the corresponding mesh.
        mesh_tags: the mesh tags.
        ax: (optional) the matplotlib axes.
        display_indices: (optional) boolean, if True displays the indices of the cells/facets.
        expression_levelset: (optional), if not None, display the contour line of the levelset.
    
    Returns:
        A matplotlib axis with the corresponding plot.
    """
    if ax is None:
        ax = plt.gca()  # type: ignore
    ax.set_aspect("equal")
    points = mesh.geometry.x

    # Get unique tags and create a custom colormap
    colors = [plt.cm.tab10(i / 10.) for i in range(10)]
    colors = colors[:5]
    cmap = mcolors.ListedColormap(colors) # type: ignore
    norm = mcolors.BoundaryNorm(np.arange(6) - 0.5, 5)

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
                cell_colors[c] = -1  # Handle cells without tags (optional)
        mappable: mpl_collections.Collection = ax.tripcolor(
            tria, cell_colors, edgecolor="k", cmap=cmap, norm=norm)
        tag_dict = {0: "No tag",
                    1: "Interior cells",
                    2: "Cut cells",
                    3: "Exterior cells",
                    4: "Padding cells"}
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
                    value_f = -1  # Handle facets without tags (optional)
                vertices = [vertex_map[v] for v in connectivity_facets_to_vertices.links(f)]
                lines_colors_as_int.append(value_f)
                lines_colors_as_str.append(cmap(value_f) if value_f != -1 else "gray")
                lines.append(points[vertices][:, :2])
                lines_linestyles.append("solid")
                if display_indices:
                    midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
                    ax.text(midpoint[0], midpoint[1], f"{f}", horizontalalignment="center", verticalalignment="center", fontsize=6)
        mappable: mpl_collections.Collection = mpl_collections.LineCollection(
            lines, cmap=cmap, norm=norm, colors=lines_colors_as_str, linestyles=lines_linestyles, linewidth=0.5)
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(mappable)
        ax.autoscale()
        tag_dict = {0: "No tag",
                    1: "Interior facets",
                    2: "Cut facets",
                    3: "Exterior facets",
                    4: "Gamma_h"}
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(mappable, cax=cax, boundaries=np.arange(6) - 0.5, ticks=np.arange(6))
    
    # Set colorbar labels
    colorbar.set_ticklabels([f"{tag_dict[key]} ({key})" for key in tag_dict.keys()])

    if expression_levelset is not None:
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        nx = 1000
        ny = 1000
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)

        xx, yy = np.meshgrid(xs, ys)
        xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
        yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
        zz_rs = expression_levelset(xx_rs, yy_rs)
        zz = zz_rs.reshape(xx.shape)

        ax.contour(xx, yy, zz, [0.], linewidths=0.5)
    
    return ax

# def plot_mesh_tags(mesh: Mesh,
#                    mesh_tags: MeshTags,
#                    ax: Axes | None = None,
#                    display_indices: bool = False,
#                    expression_levelset: NDArrayFunction | None = None) -> Axes:
#     """Plot a mesh tags object on the provied (or, if None, the current) axes object.
    
#     Args:
#         mesh: the corresponding mesh.
#         mesh_tags: the mesh tags.
#         ax: (optional) the matplotlib axes.
#         display_indices: (optional) boolean, if True displays the indices of the cells/facets.
#         expression_levelset: (optional), if not None, display the contour line of the levelset.
    
#     Returns:
#         A matplotlib axis with the corresponding plot.
#     """
#     if ax is None:
#         ax = plt.gca() # type: ignore
#     ax.set_aspect("equal")
#     unique_tags = np.unique(mesh_tags.values)
#     num_tags = len(unique_tags) + 1
#     points = mesh.geometry.x
#     cmap = [cm.get_cmap('Set1')(i / float(num_tags)) for i in range(num_tags)]
#     cmap_bounds = np.arange(num_tags + 1)
#     norm = mpl.colors.BoundaryNorm(cmap_bounds, len(cmap))
#     assert mesh_tags.dim in (mesh.topology.dim, mesh.topology.dim - 1)
#     cells_map = mesh.topology.index_map(mesh.topology.dim)
#     num_cells = cells_map.size_local + cells_map.num_ghosts
#     if mesh_tags.dim == mesh.topology.dim:
#         cells = mesh.geometry.dofmap
#         tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
#         cell_colors = np.zeros((cells.shape[0], ))
#         if display_indices:
#             tdim = mesh.topology.dim
#             connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
#             vertex_map = {
#                 topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
#                     connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
#             }
#         for c in range(num_cells):
#             if c in mesh_tags.indices:
#                 cell_colors[c] = mesh_tags.values[np.where(mesh_tags.indices == c)][0]
#                 if display_indices:
#                     vertices = [vertex_map[v] for v in connectivity_cells_to_vertices.links(c)]
#                     midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
#                     ax.text(midpoint[0], midpoint[1], f"{c}", horizontalalignment="center", verticalalignment="center", fontsize=6)
#             else:
#                 cell_colors[c] = 0
#         # cell_colors[mesh_tags.indices[mesh_tags.values != 0]] = 1
#         mappable: mpl.collections.Collection = ax.tripcolor(
#             tria, cell_colors, edgecolor="k", cmap="Set1", norm=norm)
#         tag_dict = {0: "No tag",
#                     1: "Interior cells",
#                     2: "Cut cells",
#                     3: "Exterior cells",
#                     4: "Padding cells",
#                     5: ""}
#     elif mesh_tags.dim == mesh.topology.dim - 1:
#         tdim = mesh.topology.dim
#         connectivity_cells_to_facets = mesh.topology.connectivity(tdim, tdim - 1)
#         connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
#         connectivity_facets_to_vertices = mesh.topology.connectivity(tdim - 1, 0)
#         vertex_map = {
#             topology_index: geometry_index for c in range(num_cells) for (topology_index, geometry_index) in zip(
#                 connectivity_cells_to_vertices.links(c), mesh.geometry.dofmap[c])
#         }
#         lines = list()
#         lines_colors_as_int = list()
#         lines_colors_as_str = list()
#         lines_linestyles = list()
#         for c in range(num_cells):
#             facets = connectivity_cells_to_facets.links(c)
#             for f in facets:
#                 if f in mesh_tags.indices:
#                     value_f = mesh_tags.values[f]
#                 else:
#                     value_f = 0
#                 vertices = [vertex_map[v] for v in connectivity_facets_to_vertices.links(f)]
#                 lines_colors_as_int.append(value_f)
#                 lines_colors_as_str.append(cmap[value_f])
#                 lines.append(points[vertices][:, :2])
#                 lines_linestyles.append("solid")
#                 if display_indices:
#                     midpoint = np.sum(points[vertices], axis=0)/np.shape(points[vertices])[0]
#                     ax.text(midpoint[0], midpoint[1], f"{f}", horizontalalignment="center", verticalalignment="center", fontsize=6)
#         mappable: mpl.collections.Collection = mpl.collections.LineCollection(  # type: ignore[no-redef]
#             lines, cmap="Set1", norm=norm, colors=lines_colors_as_str, linestyles=lines_linestyles, linewidth=0.5)
#         mappable.set_array(np.array(lines_colors_as_int))
#         ax.add_collection(cast(Collection[Any], mappable))
#         ax.autoscale()
#         tag_dict = {0: "No tag",
#                     1: "Interior facets",
#                     2: "Cut facets",
#                     3: "Exterior facets",
#                     4: "Gamma_h",
#                     5: ""}
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(mappable, cax=cax, boundaries=np.arange(num_tags + 1) - 0.5, ticks=np.arange(num_tags))
#     colorbar = plt.colorbar(mappable, cax=cax, boundaries=cmap_bounds, ticks=cmap_bounds)

#     colorbar.set_ticklabels([f"{tag_dict[key]}" for key in tag_dict.keys()])

#     if expression_levelset is not None:
#         x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
#         y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
#         nx = 1000
#         ny = 1000
#         xs = np.linspace(x_min, x_max, nx)
#         ys = np.linspace(y_min, y_max, ny)

#         xx, yy = np.meshgrid(xs, ys)
#         xx_rs = xx.reshape(xx.shape[0] * xx.shape[1])
#         yy_rs = yy.reshape(yy.shape[0] * yy.shape[1])
#         zz_rs = expression_levelset(xx_rs, yy_rs)
#         zz = zz_rs.reshape(xx.shape)

#         ax.contour(xx, yy, zz, [0.], linewidths=0.2)
#     return ax