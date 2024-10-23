import dolfinx as dfx
import numpy as np

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
    boundary_entities = np.setdiff1d(interior_entities, list_interior_entities)

    return list_interior_entities, boundary_entities

def tag_entities(mesh,
                 levelset,
                 edim,
                 tag_interior=1,
                 tag_boundary=2):
    """ Compute the entity tags for the interior (Omega_h) and the boundary (set of cells having a non empty intersection with Gamma_h).

    Args:
        mesh: the background mesh.
        levelset: a Levelset object.
        edim: dimension of the entities.
        tag_interior (int): the tag used to label the entities of Omega_h (default: 1).
        tag_boundary (int): the tag used to label the entities with an non-empty intersection with Gamma_h (default: 2).

    Returns:
        A meshtags object.
    """

    # We need to go through cells selection even if edim < mesh.topology.dim
    cdim = mesh.topology.dim
    interior_entities, boundary_fronteer_entities = _select_entities(mesh, levelset, cdim)

    if edim == mesh.topology.dim - 1:
        # Get the indices of facets belonging to cells in interior_entities and boundary_fronteer_entities
        mesh.topology.create_connectivity(cdim, edim)
        c2f_connect = mesh.topology.connectivity(cdim, edim)
        num_facets_per_cell = len(c2f_connect.links(0))
        c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
        fronteer_facets = np.intersect1d(c2f_map[interior_entities], c2f_map[boundary_fronteer_entities])

        interior_fronteer_facets, boundary_facets = _select_entities(mesh, levelset, edim)
        interior_entities = np.setdiff1d(interior_fronteer_facets, fronteer_facets)
        boundary_fronteer_entities = np.union1d(boundary_facets, fronteer_facets)

    entities_indices = np.hstack([interior_entities, boundary_fronteer_entities]).astype(np.int32)
    entities_markers = np.hstack([np.full_like(interior_entities, tag_interior),
                                  np.full_like(boundary_fronteer_entities, tag_boundary)]).astype(np.int32)

    sorted_indices = np.argsort(entities_indices)

    entities_tags = dfx.mesh.meshtags(mesh,
                                      edim,
                                      entities_indices[sorted_indices],
                                      entities_markers[sorted_indices])
    return entities_tags