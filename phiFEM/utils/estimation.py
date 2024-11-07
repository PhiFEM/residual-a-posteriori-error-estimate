from basix.ufl import element
import dolfinx as dfx
import numpy as np
import ufl
from ufl import avg, div, grad, inner, jump

def estimate_residual(fV, phiFEM_solver, V0=None, quadrature_degree=None):
    if phiFEM_solver.submesh is not None:
        uh = phiFEM_solver.submesh_solution
        working_cells_tags = phiFEM_solver.submesh_cells_tags
        working_mesh = phiFEM_solver.submesh
    else:
        uh = phiFEM_solver.bg_mesh_solution
        working_cells_tags = phiFEM_solver.bg_mesh_cells_tags
        working_mesh = phiFEM_solver.bg_mesh

    if quadrature_degree is None:
        k = uh.function_space.element.basix_element.degree
        quadrature_degree_cells = max(0, k - 2)
        quadrature_degree_facets = max(0, k - 1)
    
    dx = ufl.Measure("dx",
                     domain=working_mesh,
                     subdomain_data=working_cells_tags,
                     metadata={"quadrature_degree": quadrature_degree_cells})
    dS = ufl.Measure("dS",
                     domain=working_mesh,
                     subdomain_data=phiFEM_solver.facets_tags,
                     metadata={"quadrature_degree": quadrature_degree_facets})

    n   = ufl.FacetNormal(working_mesh)
    h_T = ufl.CellDiameter(working_mesh)
    h_E = ufl.FacetArea(working_mesh)

    r = fV + div(grad(uh))
    J_h = jump(grad(uh), -n)

    if V0 is None:
        DG0Element = element("DG", working_mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(working_mesh, DG0Element)

    v0 = ufl.TestFunction(V0)

    # Interior residual
    eta_T = h_T**2 * inner(inner(r, r), v0) * (dx(1) + dx(2))

    # Facets residual
    eta_E = avg(h_E) * inner(inner(J_h, J_h), avg(v0)) * (dS(1) + dS(2))
    eta = eta_T + eta_E
    eta_form = dfx.fem.form(eta)

    eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
    eta_h = dfx.fem.Function(V0)
    eta_h.vector.setArray(eta_vec.array[:])
    return eta_h

def marking(eta_h, theta=0.3):
    mesh = eta_h.function_space.mesh
    cdim = mesh.topology.dim
    fdim = cdim - 1
    assert(mesh.comm.size == 1)
    theta = 0.3

    eta_global = sum(eta_h.x.array)
    cutoff = theta * eta_global

    sorted_cells = np.argsort(eta_h.x.array)[::-1]
    rolling_sum = 0.0
    for j, e in enumerate(eta_h.x.array[sorted_cells]):
        rolling_sum += e
        if rolling_sum > cutoff:
            breakpoint = j
            break

    refine_cells = sorted_cells[0:breakpoint + 1]
    indices = np.array(np.sort(refine_cells), dtype=np.int32)
    c2f_connect = mesh.topology.connectivity(cdim, fdim)
    num_facets_per_cell = len(c2f_connect.links(0))
    c2f_map = np.reshape(c2f_connect.array, (-1, num_facets_per_cell))
    facets_indices = np.unique(np.sort(c2f_map[indices]))
    return facets_indices