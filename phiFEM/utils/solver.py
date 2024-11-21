from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
import numpy as np
import ufl
from ufl import inner, jump, grad, div, avg

from utils.compute_meshtags import tag_entities
from utils.mesh_scripts import plot_mesh_tags, compute_outward_normal

import matplotlib.pyplot as plt

# TODO: keep the connectivities as class objects to avoid unnecessary multiple computations.
class PhiFEMSolver:
    def __init__(self, bg_mesh, FE_element, PETSc_solver, levelset_element=None, num_step=0):
        self.petsc_solver = PETSc_solver
        self.i = num_step
        self.bg_mesh = bg_mesh
        self.FE_element = FE_element
        if levelset_element is None:
            self.levelset_element = FE_element
        else:
            self.levelset_element = levelset_element

        self.submesh            = None
        self.submesh_cells_tags = None
        self.bg_mesh_cells_tags = None
        self.facets_tags        = None
        self.rhs                = None
        self.levelset           = None
        self.levelset_space     = None
        self.submesh_solution   = None
        self.bg_mesh_solution   = None
        self.FE_space           = None
        self.eta_h              = None
        self.v0                 = None
    
    def _compute_normal(self, mesh):
        return compute_outward_normal(mesh, [self.bg_mesh_cells_tags, self.facets_tags], self.levelset)
    
    def _transfer_cells_tags(self, cmap):
        cdim = self.submesh.topology.dim

        # TODO: change this line to allow parallel computing
        bg_mesh_interior = self.bg_mesh_cells_tags.indices[np.where(self.bg_mesh_cells_tags.values == 1)]
        bg_cells_omega_h_gamma = self.bg_mesh_cells_tags.indices[np.where(self.bg_mesh_cells_tags.values == 2)]
        bg_cells_padding = self.bg_mesh_cells_tags.indices[np.where(self.bg_mesh_cells_tags.values == 4)]

        mask_interior = np.in1d(cmap, bg_mesh_interior)
        mask_omega_h_gamma = np.in1d(cmap, bg_cells_omega_h_gamma)
        mask_padding = np.in1d(cmap, bg_cells_padding)
        submesh_interior = np.where(mask_interior)[0]
        submesh_cells_omega_h_gamma = np.where(mask_omega_h_gamma)[0]
        submesh_cells_padding = np.where(mask_padding)[0]
        list_cells = [submesh_interior,
                      submesh_cells_omega_h_gamma,
                      submesh_cells_padding]
        list_markers = [np.full_like(submesh_interior, 1),
                        np.full_like(submesh_cells_omega_h_gamma, 2),
                        np.full_like(submesh_cells_padding, 4)]
        cells_indices = np.hstack(list_cells).astype(np.int32)
        cells_markers = np.hstack(list_markers).astype(np.int32)
        sorted_indices = np.argsort(cells_indices)

        self.submesh_cells_tags = dfx.mesh.meshtags(self.submesh,
                                                    cdim,
                                                    cells_indices[sorted_indices],
                                                    cells_markers[sorted_indices])

    def set_data(self, source_term, levelset):
        self.rhs = source_term
        self.levelset = levelset

    def mesh2mesh_interpolation(self, origin_mesh_fct, dest_mesh_fct):
        # scatter_forward has to do with ghost cells in parallel (see: https://fenicsproject.discourse.group/t/the-usage-of-the-functions-of-dolfinx/13214/4)
        origin_mesh_fct.x.scatter_forward()
        # See: https://github.com/FEniCS/dolfinx/blob/e4439ccca81b976d11c6f606d9c612afcf010a31/python/test/unit/fem/test_interpolation.py#L790
        mesh1_2_mesh2_nmm_data = dfx.fem.create_nonmatching_meshes_interpolation_data(
                                      origin_mesh_fct.function_space.mesh._cpp_object,
                                      origin_mesh_fct.function_space.element,
                                      origin_mesh_fct.function_space.mesh._cpp_object)
        dest_mesh_fct.interpolate(origin_mesh_fct, nmm_interpolation_data=mesh1_2_mesh2_nmm_data)
        dest_mesh_fct.x.scatter_forward()
        return dest_mesh_fct

    def compute_tags(self, padding=False, plot=False):
        working_mesh = self.bg_mesh
        # Tag cells of the background mesh. Used to tag the facets and/or create the submesh.
        self.bg_mesh_cells_tags = tag_entities(self.bg_mesh, self.levelset, self.bg_mesh.topology.dim, padding=padding)
        working_cells_tags = self.bg_mesh_cells_tags
        # Create the submesh and transfer the cells tags from the bg mesh to the submesh.
        # Tag the facets of the submesh.
        omega_h_cells = self.bg_mesh_cells_tags.indices[np.where(np.logical_or(np.logical_or(self.bg_mesh_cells_tags.values == 1, self.bg_mesh_cells_tags.values == 2), self.bg_mesh_cells_tags.values == 4))]
        self.submesh, c_map, v_map, n_map = dfx.mesh.create_submesh(self.bg_mesh, self.bg_mesh.topology.dim, omega_h_cells)
        self._transfer_cells_tags(c_map)
        working_cells_tags = self.submesh_cells_tags
        working_mesh = self.submesh

        self.facets_tags = tag_entities(working_mesh, self.levelset, working_mesh.topology.dim - 1, cells_tags=working_cells_tags)

        if plot:
            figure, ax = plt.subplots()
            plot_mesh_tags(working_mesh, self.facets_tags, ax=ax, display_indices=False)
            plt.savefig(f"./output/test_facets_{str(self.i).zfill(2)}.svg", format="svg", dpi=2400)
            figure, ax = plt.subplots()
            plot_mesh_tags(working_mesh, working_cells_tags, ax=ax, display_indices=False)
            plt.savefig(f"./output/test_cells_{str(self.i).zfill(2)}.svg", format="svg", dpi=2400)
    
    def set_variational_formulation(self,
                                    sigma=1.,
                                    quadrature_degree=None):

        if quadrature_degree is None:
            quadrature_degree = 2 * (self.levelset_element.basix_element.degree + 1)

        # If the submesh hasn't been created, we work directly on the background mesh.
        # We need to set-up a "boolean" DG0 function to represent Omega_h and define a custom outward pointing normal to partial Omega_h.
        if self.submesh is None:
            working_mesh = self.bg_mesh
            cells_tags = self.bg_mesh_cells_tags
        else:
            working_mesh = self.submesh
            cells_tags = self.submesh_cells_tags
        
        Omega_h_n = self._compute_normal(working_mesh)
        
        # Set up a DG function with values 1. in Omega_h and 0. outside
        DG0Element = element("DG", working_mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(working_mesh, DG0Element)
        v0 = dfx.fem.Function(V0)
        v0.vector.set(0.)
        interior_cells = cells_tags.indices[np.where(cells_tags.values == 1)]
        cut_cells      = cells_tags.indices[np.where(cells_tags.values == 2)]
        Omega_h_cells = np.union1d(interior_cells, cut_cells)
        # We do not need the dofs here since cells and DG0 dofs share the same indices in dolfinx
        v0.x.array[Omega_h_cells] = 1.

        with XDMFFile(working_mesh.comm, "output/v0.xdmf", "w") as of:
            of.write_mesh(working_mesh)
            of.write_function(v0)
        
        self.FE_space = dfx.fem.functionspace(working_mesh, self.FE_element)
        self.levelset_space = dfx.fem.functionspace(working_mesh, self.levelset_element)

        phi_h = self.levelset.interpolate(self.levelset_space)
        num_dofs = sum(np.where(phi_h.x.array < 0., 1, 0))

        f_h = self.rhs.interpolate(self.FE_space)

        h = ufl.CellDiameter(working_mesh)
        n = ufl.FacetNormal(working_mesh)

        w = ufl.TrialFunction(self.FE_space)
        v = ufl.TestFunction(self.FE_space)

        dx = ufl.Measure("dx",
                         domain=working_mesh,
                         subdomain_data=cells_tags,
                         metadata={"quadrature_degree": quadrature_degree})
        dS = ufl.Measure("dS",
                         domain=working_mesh,
                         subdomain_data=self.facets_tags,
                         metadata={"quadrature_degree": quadrature_degree})
        
        dpartial_omega_h = dS(4)

        """
        Bilinear form
        """
        stiffness = inner(grad(phi_h * w), grad(phi_h * v))
        boundary = inner(2. * avg(inner(grad(phi_h * w), Omega_h_n) * v0),
                         2. * avg(phi_h * v * v0))

        penalization_facets = sigma * avg(h) * inner(jump(grad(phi_h * w), n),
                                                     jump(grad(phi_h * v), n))
        penalization_cells = sigma * h**2 * inner(div(grad(phi_h * w)),
                                                  div(grad(phi_h * v)))

        a = stiffness             * (dx(1) + dx(2)) \
            - boundary            * dpartial_omega_h \
            + penalization_facets * dS(2) \
            + penalization_cells  * dx(2)
        
        a += inner(w, v) * v0 * (dx(3) + dx(4))

        """
        Linear form
        """
        rhs = inner(f_h, phi_h * v)
        penalization_rhs = sigma * h**2 * inner(f_h, div(grad(phi_h * v)))

        L = rhs                 * (dx(1) + dx(2))\
            - penalization_rhs  * dx(2)

        self.bilinear_form = dfx.fem.form(a)
        self.linear_form = dfx.fem.form(L)
        self.v0 = v0
        return v0, dx, dS, num_dofs
    
    def assemble(self):
        self.A = assemble_matrix(self.bilinear_form)
        self.A.assemble()
        self.b = assemble_vector(self.linear_form)

    def solve(self):
        wh = dfx.fem.Function(self.FE_space)
        self.petsc_solver.setOperators(self.A)
        self.petsc_solver.solve(self.b, wh.vector)
        phi_h = self.levelset.interpolate(self.FE_space)
        self.solution = dfx.fem.Function(self.FE_space)
        self.solution.x.array[:] = wh.x.array * phi_h.x.array
    
    def estimate_residual(self, V0=None, quadrature_degree=None, boundary_term=False):
        uh = self.solution
        working_cells_tags = self.submesh_cells_tags
        working_mesh = self.submesh

        if quadrature_degree is None:
            k = uh.function_space.element.basix_element.degree
            quadrature_degree_cells  = max(0, k - 2)
            quadrature_degree_facets = max(0, k - 1)
        
        dx = ufl.Measure("dx",
                        domain=working_mesh,
                        subdomain_data=working_cells_tags,
                        metadata={"quadrature_degree": quadrature_degree_cells})
        dS = ufl.Measure("dS",
                        domain=working_mesh,
                        subdomain_data=self.facets_tags,
                        metadata={"quadrature_degree": quadrature_degree_facets})
        ds = ufl.Measure("ds",
                        domain=working_mesh,
                        metadata={"quadrature_degree": quadrature_degree_facets})

        n   = ufl.FacetNormal(working_mesh)
        h_T = ufl.CellDiameter(working_mesh)
        h_E = ufl.FacetArea(working_mesh)

        f_h = self.rhs.interpolate(self.FE_space)

        r = f_h + div(grad(uh))
        J_h = jump(grad(uh), -n)

        if V0 is None:
            DG0Element = element("DG", working_mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(working_mesh, DG0Element)

        w0 = ufl.TestFunction(V0)

        # Interior residual
        eta_T = h_T**2 * inner(inner(r, r), w0) * self.v0 * (dx(1) + dx(2))

        # Facets residual
        eta_E = avg(h_E) * inner(inner(J_h, J_h) * avg(self.v0), avg(w0)) * (dS(1) + dS(2))

        eta = eta_T + eta_E

        if boundary_term:
            eta_boundary = h_E * inner(inner(grad(uh), n), inner(grad(uh), n)) * w0 * self.v0 * ds
            eta += eta_boundary
        
        eta_form = dfx.fem.form(eta)

        eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
        eta_h = dfx.fem.Function(V0)
        eta_h.vector.setArray(eta_vec.array[:])
        self.eta_h = eta_h
    
    def marking(self, eta_h=None, theta=0.3):
        if eta_h is None:
            eta_h = self.eta_h
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


class FEMSolver:
    def __init__(self, mesh, FE_element, PETSc_solver, num_step=0):
        self.mesh          = mesh
        self.FE_space      = dfx.fem.functionspace(mesh, FE_element)
        self.petsc_solver  = PETSc_solver
        self.i             = num_step
        self.rhs           = None
        self.bcs           = None
        self.bilinear_form = None
        self.linear_form   = None
        self.A             = None
        self.b             = None
        self.solution      = None
        self.eta_h         = None
    
    def set_data(self, source_term, bcs=None):
        self.rhs = source_term
        self.bcs = bcs
    
    def set_variational_formulation(self, quadrature_degree=None):
        if quadrature_degree is None:
            quadrature_degree = 2 * (self.FE_space.element.basix_element.degree + 1)
        
        f_h = self.rhs.interpolate(self.FE_space)

        num_dofs = len(f_h.x.array[:])
        u = ufl.TrialFunction(self.FE_space)
        v = ufl.TestFunction(self.FE_space)

        dx = ufl.Measure("dx",
                         domain=self.mesh,
                         metadata={"quadrature_degree": quadrature_degree})
        
        """
        Bilinear form
        """
        a = inner(grad(u), grad(v)) * dx

        """
        Linear form
        """
        L = inner(f_h, v) * dx

        self.bilinear_form = dfx.fem.form(a)
        self.linear_form = dfx.fem.form(L)
        return num_dofs

    def assemble(self):
        self.A = assemble_matrix(self.bilinear_form, bcs=self.bcs)
        self.A.assemble()
        self.b = assemble_vector(self.linear_form)
        dfx.fem.apply_lifting(self.b, [self.bilinear_form], [self.bcs])
        dfx.fem.set_bc(self.b, self.bcs)

    def solve(self):
        self.solution = dfx.fem.Function(self.FE_space)
        self.petsc_solver.setOperators(self.A)
        self.petsc_solver.solve(self.b, self.solution.vector)
    
    def estimate_residual(self, V0=None, quadrature_degree=None):
        if quadrature_degree is None:
            k = self.solution.function_space.element.basix_element.degree
            quadrature_degree_cells  = max(0, k - 2)
            quadrature_degree_facets = max(0, k - 1)
        
        dx = ufl.Measure("dx",
                         domain=self.mesh,
                         metadata={"quadrature_degree": quadrature_degree_cells})
        dS = ufl.Measure("dS",
                         domain=self.mesh,
                         metadata={"quadrature_degree": quadrature_degree_facets})

        n   = ufl.FacetNormal(self.mesh)
        h_T = ufl.CellDiameter(self.mesh)
        h_E = ufl.FacetArea(self.mesh)

        f_h = self.rhs.interpolate(self.FE_space)

        r = f_h + div(grad(self.solution))
        J_h = jump(grad(self.solution), -n)

        if V0 is None:
            DG0Element = element("DG", self.mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(self.mesh, DG0Element)

        v0 = ufl.TestFunction(V0)

        # Interior residual
        eta_T = h_T**2 * inner(inner(r, r), v0) * dx

        # Facets residual
        eta_E = avg(h_E) * inner(inner(J_h, J_h), avg(v0)) * dS

        eta = eta_T + eta_E

        eta_form = dfx.fem.form(eta)

        eta_vec = dfx.fem.petsc.assemble_vector(eta_form)
        eta_h = dfx.fem.Function(V0)
        eta_h.vector.setArray(eta_vec.array[:])
        self.eta_h = eta_h
    
    def marking(self, eta_h=None, theta=0.3):
        if eta_h is None:
            eta_h = self.eta_h
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