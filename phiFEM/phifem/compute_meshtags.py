from   basix.ufl import element
import dolfinx as dfx
from   dolfinx.fem import Function
from   dolfinx.fem.petsc import assemble_vector
from   dolfinx.mesh import Mesh, MeshTags
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import ufl # type: ignore[import-untyped]
from   ufl import inner

from phiFEM.phifem.mesh_scripts import plot_mesh_tags

def tag_cells(mesh: Mesh,
              discrete_levelset: Function,
              padding: bool = False,
              plot: bool = False) -> MeshTags:
    """Tag the mesh cells by computing detection = Σ f(dof)/Σ|f(dof)| for each cell.
         detection == 1  => the cell is stricly OUTSIDE {phi_h < 0} => we tag it as 3
         detection == -1 => the cell is stricly INSIDE  {phi_h < 0} => we tag it as 1
         otherwise       => the cell is CUT by Gamma_h              => we tag is as 2

    Args:
        mesh: the background mesh.
        discrete_levelset: the discretization of the levelset.
        padding: unused for the moment, TODO: implement the possiblity to add a padding.
        plot: if True plots the mesh with tags (can drastically slow the computation!).
    
    Returns:
        The cells tags as a MeshTags object.
    """
    # Create the custom quadrature rule.
    # The evaluation points are the dofs of the reference cell.
    # The weights are 1.
    quadrature_points: npt.NDArray[np.float64]
    levelset_degree = discrete_levelset.function_space.ufl_element().degree
    if mesh.topology.cell_name() == "triangle":
        xs = np.linspace(0., 1., levelset_degree + 1)
        xx, yy = np.meshgrid(xs, xs)
        x_coords = xx.reshape((1, xx.shape[0] * xx.shape[1]))
        y_coords = yy.reshape((1, yy.shape[0] * yy.shape[1]))
        points = np.vstack([x_coords, y_coords])
        quadrature_points = points[:,points[1,:] <= np.ones_like(points[0,:])-points[0,:]]
    elif mesh.topology.cell_name() == "tetrahedron":
        xs = np.linspace(0., 1., levelset_degree + 1)
        xx, yy, zz = np.meshgrid(xs, xs, xs)
        x_coords = xx.reshape((1, xx.shape[0] * xx.shape[1]))
        y_coords = yy.reshape((1, yy.shape[0] * yy.shape[1]))
        z_coords = zz.reshape((1, zz.shape[0] * zz.shape[1]))
        points = np.vstack([x_coords, y_coords, z_coords])
        quadrature_points = points[:,points[2,:] <= np.ones_like(points[0,:])-points[0,:]-points[1,:]]
    else:
        raise NotImplementedError("Mesh cell type not supported. Only supported types are: triangle, tetrahedron.")
    
    quadrature_weights = np.ones_like(quadrature_points[0,:])
    custom_rule = {"quadrature_rule":    "custom",
                   "quadrature_points":  quadrature_points.T,
                   "quadrature_weights": quadrature_weights}
    
    cells_detection_dx = ufl.Measure("dx",
                                     domain=mesh,
                                     metadata=custom_rule)
    
    # We localize at each cell via a DG0 test function.
    DG0Element = element("DG", mesh.topology.cell_name(), 0)
    V0 = dfx.fem.functionspace(mesh, DG0Element)
    v0 = ufl.TestFunction(V0)

    # Assemble the numerator of detection
    cells_detection_num = inner(discrete_levelset, v0) * cells_detection_dx
    cells_detection_num_form = dfx.fem.form(cells_detection_num)
    cells_detection_num_vec = assemble_vector(cells_detection_num_form)
    # Assemble the denominator of detection
    cells_detection_denom = inner(ufl.algebra.Abs(discrete_levelset), v0) * cells_detection_dx
    cells_detection_denom_form = dfx.fem.form(cells_detection_denom)
    cells_detection_denom_vec = assemble_vector(cells_detection_denom_form)

    # cells_detection_denom_vec is not supposed to be zero, this would mean that the levelset is zero at all dofs in a cell, which is not allowed.
    if np.any(np.isclose(cells_detection_denom_vec, 0.)):
        raise ValueError("The discrete levelset vanishes on at least one entire mesh cell.")
    
    cells_detection_vec = cells_detection_num_vec.array/cells_detection_denom_vec.array
    detection = dfx.fem.Function(V0)
    detection.x.array[:] = cells_detection_vec
    from dolfinx.io import XDMFFile
    with XDMFFile(mesh.comm, "./detection.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(detection)
    
    exterior_indices = np.where(cells_detection_vec == 1.)[0]
    interior_indices = np.where(cells_detection_vec == -1.)[0]
    cut_indices      = np.where(np.logical_and(cells_detection_vec > -1., 
                                               cells_detection_vec < 1.))[0]
    
    if len(interior_indices) == 0:
        raise ValueError("No interior cells (1)!")
    if len(cut_indices) == 0:
        raise ValueError("No cut cells (2)!")

    # Create the meshtags from the indices.
    indices = np.hstack([exterior_indices,
                         interior_indices,
                         cut_indices]).astype(np.int32)
    exterior_marker = np.full_like(exterior_indices, 3).astype(np.int32)
    interior_marker = np.full_like(interior_indices, 1).astype(np.int32)
    cut_marker      = np.full_like(cut_indices,      2).astype(np.int32)
    markers = np.hstack([exterior_marker,
                         interior_marker,
                         cut_marker]).astype(np.int32)
    sorted_indices = np.argsort(indices)

    cells_tags = dfx.mesh.meshtags(mesh,
                                   mesh.topology.dim,
                                   indices[sorted_indices],
                                   markers[sorted_indices])

    if plot:
        figure, ax = plt.subplots()
        plot_mesh_tags(mesh, cells_tags, ax=ax, display_indices=False)
        plt.savefig("./cells_tags.svg", format="svg", dpi=2400, bbox_inches="tight")
    
    return cells_tags

def tag_facets(mesh: Mesh,
               discrete_levelset: Function,
               cells_tags: MeshTags,
               plot: bool = False) -> MeshTags:
    """Tag the mesh facets.

    Args:
        mesh: the background mesh.
        discrete_levelset: the discretization of the levelset.
        cells_tags: the MeshTags object containing cells tags.
        plot: if True plots the mesh with tags (can drastically slow the computation!).
    
    Returns:
        The facets tags as a MeshTags object.
    """
    cdim = mesh.topology.dim
    fdim = cdim - 1
    # Create the cell to facet connectivity and reshape it into an array s.t. c2f_map[cell_index] = [facets of this cell index]
    mesh.topology.create_connectivity(cdim, fdim)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    num_facets_per_cell = len(c2f_connect.links(0))
    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))

    # Get tagged cells
    interior_cells = cells_tags.find(1)
    cut_cells      = cells_tags.find(2)
    exterior_cells = cells_tags.find(3)
    
    # Facets shared by an interior cell and a cut cell
    interior_boundary_facets = np.intersect1d(c2f_map[interior_cells],
                                              c2f_map[cut_cells])
    # Facets shared by an exterior cell and a cut cell
    exterior_boundary_facets = np.intersect1d(c2f_map[exterior_cells],
                                              c2f_map[cut_cells])
    # Boundary facets ∂Ω_h
    real_boundary_facets = np.intersect1d(c2f_map[cut_cells], 
                                          dfx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.ones_like(x[1]).astype(bool)))
    boundary_facets = np.union1d(exterior_boundary_facets, real_boundary_facets)

    # Cut facets F_h^Γ
    cut_facets = np.setdiff1d(c2f_map[cut_cells],
                              np.union1d(exterior_boundary_facets, boundary_facets))

    # Interior facets 
    interior_facets = np.setdiff1d(c2f_map[interior_cells],
                                   interior_boundary_facets)
    # Exterior facets 
    exterior_facets = np.setdiff1d(c2f_map[exterior_cells],
                                   exterior_boundary_facets)
    
    # if len(interior_facets) == 0:
    #     raise ValueError("No interior facets (1)!")
    # if len(cut_facets) == 0:
    #     raise ValueError("No cut facets (2)!")
    # if len(boundary_facets) == 0:
    #     raise ValueError("No boundary facets (4)!")
    
    # Create the meshtags from the indices.
    indices = np.hstack([exterior_facets,
                         interior_facets,
                         cut_facets,
                         boundary_facets]).astype(np.int32)
    interior_marker = np.full_like(interior_facets, 1).astype(np.int32)
    cut_marker      = np.full_like(cut_facets,      2).astype(np.int32)
    exterior_marker = np.full_like(exterior_facets, 3).astype(np.int32)
    boundary_marker = np.full_like(boundary_facets, 4).astype(np.int32)
    markers = np.hstack([exterior_marker,
                         interior_marker,
                         cut_marker,
                         boundary_marker]).astype(np.int32)
    sorted_indices = np.argsort(indices)

    facets_tags = dfx.mesh.meshtags(mesh,
                                    fdim,
                                    indices[sorted_indices],
                                    markers[sorted_indices])
    
    if plot:
        figure, ax = plt.subplots()
        plot_mesh_tags(mesh, facets_tags, ax=ax, display_indices=False)
        plt.savefig("./facets_tags.svg", format="svg", dpi=2400, bbox_inches="tight")
    return facets_tags