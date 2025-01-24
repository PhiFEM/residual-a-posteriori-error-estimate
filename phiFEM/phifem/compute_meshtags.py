import dolfinx as dfx
from dolfinx.mesh import Mesh, MeshTags
from dolfinx import cpp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt 
from phiFEM.phifem.mesh_scripts import plot_mesh_tags
from phiFEM.phifem.continuous_functions import Levelset
from typing import Any, Tuple

# TODO: Modify to use in parallel

def _select_entities(mesh: Mesh,
                     levelset: Levelset,
                     edim: int,
                     padding: bool =False) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]] | Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """ Compute the list of entities strictly inside Omega_h and the list of entities having a non-empty intersection with Gamma_h.

    Args:
        mesh: the background mesh.
        levelset: a Levelset object
        edim: the dimension of the entities.
        padding: (optional) bool, select padding entities to increase the chances of the submesh to contain the exact boundary.
    
    Returns:
        Two lists of entities indices.
    """
    mesh.topology.create_connectivity(edim, edim)
    # List all entities of the mesh
    entities: npt.NDArray[np.int32] = np.arange(mesh.topology.index_map(edim).size_global, dtype = np.int32) # TODO: change this line to allow parallel computing

    # List entities that are stricly included in Omega_h
    list_interior_entities: npt.NDArray[Any] = dfx.mesh.locate_entities(mesh,
                                                                        edim,
                                                                        levelset.interior(0.))

    # List entities that are strictly excluded from Omega_h
    exterior_entities: npt.NDArray[Any] = dfx.mesh.locate_entities(mesh,
                                                                   edim,
                                                                   levelset.exterior(0.))

    # List entities having a non-empty intersection with Omega_h
    interior_entities: npt.NDArray[Any] = np.setdiff1d(entities, exterior_entities)
    # Cells case
    # List entities having a non-empty intersection with Gamma_h
    cut_entities: npt.NDArray[Any] = np.setdiff1d(interior_entities, list_interior_entities)

    padding_entities: npt.NDArray[Any] = np.asarray([])
    if padding:
        hmax_cutcells: float = max(cpp.mesh.h(mesh._cpp_object, edim, cut_entities))
        list_exterior_padding_entities = dfx.mesh.locate_entities(mesh,
                                                                  edim,
                                                                  levelset.exterior(0., padding=hmax_cutcells))
        padding_interior_entities = np.setdiff1d(entities, list_exterior_padding_entities)
        padding_entities = np.setdiff1d(padding_interior_entities, np.union1d(interior_entities, cut_entities))
        exterior_entities = np.setdiff1d(exterior_entities, padding_entities)

        return list_interior_entities, cut_entities, exterior_entities, padding_entities
    else:
        return list_interior_entities, cut_entities, exterior_entities

def tag_entities(mesh: Mesh,
                 levelset: Levelset,
                 edim: int,
                 cells_tags: MeshTags | None = None,
                 padding: bool = False,
                 plot: bool = False) -> MeshTags:
    """ Compute the entity tags for the interior (Omega_h) and the cut (set of cells having a non empty intersection with Gamma_h).
    Tag = 1: interior strict (Omega_h \ Omega_Gamma_h)
    Tag = 2: cut (Omega_Gamma_h)
    Tag = 3: exterior (entities strictly outside Omega_h)
    Tag = 4: facets: Gamma_h
             cells: padding cells.

    Args:
        mesh: the background mesh.
        levelset: a Levelset object.
        edim: dimension of the entities.
        cells_tags: the cells tags.

    Returns:
        A meshtags object.
    """

    # If cells_tags is not provided, we need to go through cells selection.
    cdim: int = mesh.topology.dim
    if cells_tags is None:
        if padding:
            interior_entities, cut_fronteer_entities, exterior_entities, padding_entities = _select_entities(mesh, levelset, cdim, padding=padding)
        else:
            interior_entities, cut_fronteer_entities, exterior_entities = _select_entities(mesh, levelset, cdim, padding=padding)
    else:
        interior_entities     = cells_tags.find(1)
        cut_fronteer_entities = cells_tags.find(2)
        exterior_entities     = cells_tags.find(3)
        padding_entities      = cells_tags.find(4)
    
    lists: list[npt.NDArray[np.int32] | list[int]]
    if edim == mesh.topology.dim - 1:
        # Get the indices of facets belonging to cells in interior_entities and cut_fronteer_entities
        mesh.topology.create_connectivity(cdim, edim)
        c2f_connect = mesh.topology.connectivity(cdim, edim)
        num_facets_per_cell = len(c2f_connect.links(0))
        # Note: here the reshape does not need any special treatment since all cells have the same number of facets.
        c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
        interior_boundary_facets = np.intersect1d(c2f_map[interior_entities],
                                                  c2f_map[cut_fronteer_entities])

        if padding:
            boundary_facets = np.intersect1d(c2f_map[cut_fronteer_entities], 
                                             np.union1d(c2f_map[exterior_entities], c2f_map[padding_entities]))
        else:
            boundary_facets = np.intersect1d(c2f_map[cut_fronteer_entities], 
                                             c2f_map[exterior_entities])

        interior_fronteer_facets, cut_facets, exterior_facets = _select_entities(mesh, levelset, edim)
        interior_facets = np.setdiff1d(interior_fronteer_facets, interior_boundary_facets)
        cut_facets = np.union1d(cut_facets, interior_boundary_facets)
        exterior_facets = np.setdiff1d(exterior_facets, boundary_facets)
        
        # Add the boundary facets to the proper lists of facets (shouldn't be necessary since the intersection of Gamma_h with the boundary of the box should be empty)
        background_boundary_facets = dfx.mesh.locate_entities_boundary(mesh,
                                                                       edim,
                                                                       lambda x: np.ones_like(x[1]).astype(bool))
        intersect_exterior_boundary_facets = np.intersect1d(c2f_map[cut_fronteer_entities], background_boundary_facets)
        exterior_facets = np.setdiff1d(exterior_facets, intersect_exterior_boundary_facets)
        boundary_facets = np.union1d(boundary_facets, intersect_exterior_boundary_facets)

        lists = [interior_facets,
                 cut_facets,
                 exterior_facets,
                 boundary_facets]

        assert len(interior_facets) > 0, "No interior facets (1) tagged!"
        assert len(cut_facets)      > 0, "No cut facets (2) tagged!"
        assert len(boundary_facets) > 0, "No boundary facets (4) tagged!"
    elif edim == mesh.topology.dim:
        if padding:
            lists = [interior_entities,
                     cut_fronteer_entities,
                     exterior_entities,
                     padding_entities]
        else:
            lists = [interior_entities,
                     cut_fronteer_entities,
                     exterior_entities]


        assert len(interior_entities)     > 0, "No interior cells (1) tagged!"
        assert len(cut_fronteer_entities) > 0, "No cut cells (2) tagged!"
        if padding:
            assert len(padding_entities)  > 0, "No padding cells (4) tagged!"

    list_markers = [np.full_like(l, i+1) for i, l in enumerate(lists)]
    entities_indices = np.hstack(lists).astype(np.int32)
    entities_markers = np.hstack(list_markers).astype(np.int32)

    sorted_indices = np.argsort(entities_indices)
 
    entities_tags = dfx.mesh.meshtags(mesh,
                                      edim,
                                      entities_indices[sorted_indices],
                                      entities_markers[sorted_indices])
    
    if plot:
        figure, ax = plt.subplots()
        plot_mesh_tags(mesh, entities_tags, ax=ax, display_indices=False, expression_levelset=levelset.expression)
        if edim == mesh.topology.dim:
            ename = "cells"
        else:
            ename = "facets"
        plt.savefig(f"./{ename}_tags.svg", format="svg", dpi=2400, bbox_inches="tight")
    return entities_tags