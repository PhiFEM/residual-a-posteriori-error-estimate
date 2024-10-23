import dolfinx as dfx
import numpy as np

# TODO: Modify to use in parallel

def _select_entities(mesh, levelset, edim):
    """ Compute the list of entities strictly inside Omega_h and the list of entities having a non-empty intersection with Gamma_h.

    Args:
        mesh: the background mesh.
        levelset: a Levelset object
        edim: the dimension of the entities.
    
    Returns:
        Two lists of entities indices.
    """
    mesh.topology.create_connectivity(edim, edim)
    # List all entities of the mesh
    entities = np.arange(mesh.topology.index_map(edim).size_global, dtype = np.int32) # TODO: change this line to allow parallel computing

    # List entities that are stricly included in Omega_h
    list_interior_entities = dfx.mesh.locate_entities(mesh,
                                                      edim,
                                                      levelset.interior(0.))

    # List entities that are strictly excluded from Omega_h
    exterior_entities = dfx.mesh.locate_entities(mesh,
                                                 edim,
                                                 levelset.exterior(0.))

    # List entities having a non-empty intersection with Omega_h
    interior_entities = np.setdiff1d(entities, exterior_entities)
    # Cells case
    # List entities having a non-empty intersection with Gamma_h
    cut_entities = np.setdiff1d(interior_entities, list_interior_entities)

    return list_interior_entities, cut_entities, exterior_entities

def tag_entities(mesh,
                 levelset,
                 edim,
                 cells_tags=None):
    """ Compute the entity tags for the interior (Omega_h) and the cut (set of cells having a non empty intersection with Gamma_h).
    Tag = 1: interior (Omega_h)
    Tag = 2: cut (Omega_Gamma_h)
    Tag = 3: exterior (entities strictly outside Omega_h)
    Tag = 4: Gamma_h (for facets only, edim < mesh.topology.dim)

    Args:
        mesh: the background mesh.
        levelset: a Levelset object.
        edim: dimension of the entities.
        cells_tags (Meshtags object): the cells tags (default: None).

    Returns:
        A meshtags object.
    """

    # If cells_tags is not provided, we need to go through cells selection.
    cdim = mesh.topology.dim
    if cells_tags is None:
        interior_entities, cut_fronteer_entities, exterior_entities = _select_entities(mesh, levelset, cdim)
    else:
        interior_entities     = cells_tags.indices[np.where(cells_tags.values == 1)]
        cut_fronteer_entities = cells_tags.indices[np.where(cells_tags.values == 2)]
        exterior_entities     = cells_tags.indices[np.where(cells_tags.values == 3)]

    if edim == mesh.topology.dim - 1:
        # Get the indices of facets belonging to cells in interior_entities and cut_fronteer_entities
        mesh.topology.create_connectivity(cdim, edim)
        c2f_connect = mesh.topology.connectivity(cdim, edim)
        num_facets_per_cell = len(c2f_connect.links(0))
        c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
        interior_boundary_facets = np.intersect1d(c2f_map[interior_entities],
                                                  c2f_map[cut_fronteer_entities])
        boundary_facets = np.intersect1d(c2f_map[cut_fronteer_entities], 
                                         c2f_map[exterior_entities])

        interior_fronteer_facets, cut_facets, exterior_facets = _select_entities(mesh, levelset, edim)
        interior_facets = np.setdiff1d(interior_fronteer_facets, interior_boundary_facets)
        cut_facets = np.union1d(cut_facets, interior_boundary_facets)
        exterior_facets = np.setdiff1d(exterior_facets, boundary_facets)
        
        # Add the boundary facets to the proper lists of facets
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
    elif edim == mesh.topology.dim:
        lists = [interior_entities,
                 cut_fronteer_entities,
                 exterior_entities]

    list_markers = [np.full_like(l, i+1) for i, l in enumerate(lists)]
    entities_indices = np.hstack(lists).astype(np.int32)
    entities_markers = np.hstack(list_markers).astype(np.int32)

    sorted_indices = np.argsort(entities_indices)

    entities_tags = dfx.mesh.meshtags(mesh,
                                      edim,
                                      entities_indices[sorted_indices],
                                      entities_markers[sorted_indices])
    return entities_tags