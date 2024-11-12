from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
import numpy as np
import os
import pandas as pd
import ufl
from ufl import inner, jump, grad, div, avg

from utils.derivatives import negative_laplacian, compute_gradient
from utils.compute_meshtags import tag_entities
from utils.mesh_scripts import plot_mesh_tags, compute_outward_normal

import matplotlib.pyplot as plt

class ContinuousFunction:
    def __init__(self, expression):
        self.expression = expression
        self.interpolated = None
    
    def __call__(self, x, y):
        return self.expression(x, y)
    
    def dolfinx_call(self, x):
        return self(x[0], x[1])
    
    def interpolate(self, FE_space):
        self.interpolated = dfx.fem.Function(FE_space)
        self.interpolated.interpolate(self.dolfinx_call)

class Levelset(ContinuousFunction):
    def exterior(self, t):
        """ Compute a lambda function determining if the point x is outside the domain defined by the isoline of level t.
        
        Args:
            t (float): level of the isoline.
        
        Return:
            lambda function taking a tuple of coordinates and returning a boolean 
        """
        return lambda x: self(x[0], x[1]) > t
    
    def interior(self, t):
        """ Compute a lambda function determining if the point x is inside the domain defined by the isoline of level t.
        
        Args:
            t (float): level of the isoline.
        
        Return:
            lambda function taking a tuple of coordinates and returning a boolean 
        """
        return lambda x: self(x[0], x[1]) < t
    
    def gradient(self):
        def func(x, y):
            return self.__call__(x, y) # Dirty workaround because compute_gradient looks for the number of arguments in order to determine the dimension and "self" messes up the count.
        return compute_gradient(func)

class ExactSolution(ContinuousFunction):
    def compute_negative_laplacian(self):
        def func(x, y):
            return self.__call__(x, y) # Dirty workaround because negative_laplacian looks for the number of arguments in order to determine the dimension and "self" messes up the count.
        comp_nlap = negative_laplacian(func)
        self.nlap = ContinuousFunction(lambda x, y: comp_nlap([x, y]))

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
        self.submesh_solution   = None
        self.bg_mesh_solution   = None
        self.FE_space           = None
    
    def _compute_normal(self, mesh):
        return compute_outward_normal(mesh, [self.bg_mesh_cells_tags, self.facets_tags], self.levelset)
    
    def _transfer_cells_tags(self, cmap):
        cdim = self.submesh.topology.dim
        # TODO: change this line to allow parallel computing
        bg_mesh_interior = self.bg_mesh_cells_tags.indices[np.where(self.bg_mesh_cells_tags.values == 1)]
        bg_cells_omega_h_gamma = self.bg_mesh_cells_tags.indices[np.where(self.bg_mesh_cells_tags.values == 2)]
        mask_interior = np.in1d(cmap, bg_mesh_interior)
        mask_omega_h_gamma = np.in1d(cmap, bg_cells_omega_h_gamma)
        submesh_interior = np.where(mask_interior)[0]
        submesh_cells_omega_h_gamma = np.where(mask_omega_h_gamma)[0]
        list_cells = [submesh_interior, submesh_cells_omega_h_gamma]
        list_markers = [np.full_like(l, i+1) for i, l in enumerate(list_cells)]
        cells_indices = np.hstack(list_cells).astype(np.int32)
        cells_markers = np.hstack(list_markers).astype(np.int32)
        sorted_indices = np.argsort(cells_indices)

        self.submesh_cells_tags = dfx.mesh.meshtags(self.submesh,
                                                    cdim,
                                                    cells_indices[sorted_indices],
                                                    cells_markers[sorted_indices])

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

    def set_data(self, source_term, levelset):
        self.rhs = source_term
        self.levelset = levelset
    
    def compute_tags(self, create_submesh=False, plot=False):
        working_mesh = self.bg_mesh
        # Tag cells of the background mesh. Used to tag the facets and/or create the submesh.
        self.bg_mesh_cells_tags = tag_entities(self.bg_mesh, self.levelset, self.bg_mesh.topology.dim)
        working_cells_tags = self.bg_mesh_cells_tags
        # Create the submesh and transfer the cells tags from the bg mesh to the submesh.
        # Tag the facets of the submesh.
        if create_submesh:
            omega_h_cells = self.bg_mesh_cells_tags.indices[np.where(np.logical_or(self.bg_mesh_cells_tags.values == 1, self.bg_mesh_cells_tags.values == 2))]
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
        v0 = None

        if quadrature_degree is None:
            quadrature_degree = 2 * (self.levelset_element.basix_element.degree + 1)

        # If the submesh hasn't been created, we work directly on the background mesh.
        # We need to set-up a "boolean" DG0 function to represent Omega_h and define a custom outward pointing normal to partial Omega_h.
        if self.submesh is None:
            working_mesh = self.bg_mesh
            cells_tags = self.bg_mesh_cells_tags
            
            # Set up a DG function with values 1. in Omega_h and 0. outside
            DG0Element = element("DG", working_mesh.topology.cell_name(), 0)
            V0 = dfx.fem.functionspace(working_mesh, DG0Element)
            v0 = dfx.fem.Function(V0)
            v0.vector.set(0.)
            interior_cells = cells_tags.indices[np.where(self.bg_mesh_cells_tags.values == 1)]
            cut_cells      = cells_tags.indices[np.where(self.bg_mesh_cells_tags.values == 2)]
            Omega_h_cells = np.union1d(interior_cells, cut_cells)
            # We do not need the dofs here since cells and DG0 dofs share the same indices in dolfinx
            v0.x.array[Omega_h_cells] = 1.

            Omega_h_n = self._compute_normal(working_mesh)
        else:
            working_mesh = self.submesh
            dpartial_omega_h = ufl.Measure("ds",
                                           domain=working_mesh,
                                           metadata={"quadrature_degree": quadrature_degree})
            cells_tags = self.submesh_cells_tags
        
        self.FE_space = dfx.fem.functionspace(working_mesh, self.FE_element)
        levelset_space = dfx.fem.functionspace(working_mesh, self.levelset_element)

        self.levelset.interpolate(levelset_space)
        phi_h = self.levelset.interpolated
        num_dofs = sum(np.where(phi_h.x.array < 0., 1, 0))

        self.rhs.interpolate(self.FE_space)
        f_h = self.rhs.interpolated

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
        
        if self.submesh is None:
            dpartial_omega_h = dS(4)

        """
        Bilinear form
        """
        stiffness = inner(grad(phi_h * w), grad(phi_h * v))
        if self.submesh is None:
            boundary = inner(2. * avg(inner(grad(phi_h * w), Omega_h_n) * v0),
                            2. * avg(phi_h * v * v0))
        else:
            boundary = inner(inner(grad(phi_h * w), n), phi_h * v)

        penalization_facets = sigma * avg(h) * inner(jump(grad(phi_h * w), n),
                                                     jump(grad(phi_h * v), n))
        penalization_cells = sigma * h**2 * inner(div(grad(phi_h * w)),
                                                  div(grad(phi_h * v)))

        a = stiffness             * (dx(1) + dx(2)) \
            - boundary            * dpartial_omega_h \
            + penalization_facets * dS(2) \
            + penalization_cells  * dx(2)
        
        if self.submesh is None:
            # This term is useless (always zero, everybody hate it)
            # but PETSc complains if I don't include it
            a += inner(w, v) * v0 * dx(3)

        """
        Linear form
        """
        rhs = inner(f_h, phi_h * v)
        penalization_rhs = sigma * h**2 * inner(f_h, div(grad(phi_h * v)))

        L = rhs                 * (dx(1) + dx(2))\
            - penalization_rhs  * dx(2)

        self.bilinear_form = dfx.fem.form(a)
        self.linear_form = dfx.fem.form(L)

        return v0, dx, dS, num_dofs
    
    def assemble(self):
        self.A = assemble_matrix(self.bilinear_form)
        self.A.assemble()
        self.b = assemble_vector(self.linear_form)

    def solve(self):
        self.solution = dfx.fem.Function(self.FE_space)
        self.petsc_solver.setOperators(self.A)
        self.petsc_solver.solve(self.b, self.solution.vector)
    
    def compute_submesh_solution(self):
        assert self.submesh is not None, "No submesh has been created."
        # If a submesh has been created, phi_h lives on the submesh
        phi_h = self.levelset.interpolated
        uh = dfx.fem.Function(self.FE_space)
        uh.x.array[:] = self.solution.x.array * phi_h.x.array
        self.submesh_solution = uh
        return uh
    
    def compute_bg_mesh_solution(self):
        if self.submesh is None:
            phi_h = self.levelset.interpolated
            uh_bg = dfx.fem.Function(self.FE_space)
            uh_bg.x.array[:] = self.solution.x.array * phi_h.x.array
        else:
            V_bg = dfx.fem.functionspace(self.bg_mesh, self.FE_element)
            phi_bg = self.levelset.interpolated
            solution_bg = dfx.fem.Function(V_bg)
            self.mesh2mesh_interpolation(self.solution, solution_bg)
            uh_bg = dfx.fem.Function(V_bg)
            uh_bg.x.array[:] = solution_bg.x.array * phi_bg.x.array
        self.bg_mesh_solution = uh_bg
        return uh_bg

class ResultsSaver:
    def __init__(self, output_path, data_keys):
        self.output_path = output_path
        self.results = {key: [] for key in data_keys}

        if not os.path.isdir(output_path):
	        print(f"{output_path} directory not found, we create it.")
	        os.mkdir(os.path.join(".", output_path))

        output_functions_path = os.path.join(output_path, "functions/")
        if not os.path.isdir(output_functions_path):
	        print(f"{output_functions_path} directory not found, we create it.")
	        os.mkdir(output_functions_path)
        
    def save_values(self, values, prnt=False):
        for key, val in zip(self.results.keys(), values):
            self.results[key].append(val)
        
        self.dataframe = pd.DataFrame(self.results)
        self.dataframe.to_csv(os.path.join(self.output_path, "results.csv"))
        if prnt:
            print(self.dataframe)
    
    def save_function(self, function, file_name):
        mesh = function.function_space.mesh
        with XDMFFile(mesh.comm, os.path.join(self.output_path, "functions",  file_name + ".xdmf"), "w") as of:
            of.write_mesh(mesh)
            of.write_function(function)