from basix.ufl import element
import dolfinx as dfx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import numpy as np
import ufl
from ufl import inner, dot, jump, grad, div, avg

from utils.derivatives import negative_laplacian, compute_gradient
from utils.compute_meshtags import tag_entities
from utils.mesh_scripts import plot_mesh_tags, compute_outward_normal

import matplotlib.pyplot as plt 
class ContinuousFunction:
    def __init__(self, expression):
        self.expression = expression
    
    def __call__(self, x, y):
        return self.expression(x, y)
    
    def dolfinx_call(self, x):
        return self(x[0], x[1])

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
        func = lambda x, y: self.__call__(x, y) # Dirty workaround because compute_gradient looks for the number of arguments in order to determine the dimension and "self" messes up the count.
        return compute_gradient(func)

class ExactSolution(ContinuousFunction):
    def negative_laplacian(self):
        func = lambda x, y: self.__call__(x, y) # Dirty workaround because negative_laplacian looks for the number of arguments in order to determine the dimension and "self" messes up the count.
        return negative_laplacian(func)

# TODO: keep the connectivities as class objects to avoid unnecessary multiple computations.
class PhiFEMSolver:
    def __init__(self, PETSc_solver, num_step):
        self.petsc_solver = PETSc_solver
        self.i = num_step
    
    def _compute_normal(self, mesh):
        return compute_outward_normal(mesh, [self.cells_tags, self.facets_tags], self.levelset)
    
    def set_data(self, discrete_rhs, levelset):
        self.rhs = discrete_rhs
        self.levelset = levelset
    
    def compute_tags(self, mesh, plot=False):
        self.cells_tags  = tag_entities(mesh, self.levelset, mesh.topology.dim)
        self.facets_tags = tag_entities(mesh, self.levelset, mesh.topology.dim - 1, cells_tags=self.cells_tags)

        if plot:
            figure, ax = plt.subplots()
            plot_mesh_tags(mesh, self.facets_tags, ax=ax, display_indices=False)
            plt.savefig("./output/test_facets.png", dpi=1200)
            figure, ax = plt.subplots()
            plot_mesh_tags(mesh, self.cells_tags, ax=ax, display_indices=False)
            plt.savefig("./output/test_cells.png", dpi=1200)

    def set_variational_formulation(self, FE_space, sigma=1., quadrature_degree=None, levelset_space=None):
        mesh = FE_space.mesh
        if quadrature_degree is None:
            quadrature_degree = 2 * (FE_space.element.basix_element.degree + 1)

        phi = self.levelset

        if levelset_space is None:
            phi_disc = dfx.fem.Function(FE_space)
        else:
            phi_disc = dfx.fem.Function(levelset_space)
        phi_disc.interpolate(phi.dolfinx_call)

        f = self.rhs

        # Set up a DG function with values 1. in Omega_h and 0. outside
        DG0Element = element("DG", mesh.topology.cell_name(), 0)
        V0 = dfx.fem.functionspace(mesh, DG0Element)
        v0 = dfx.fem.Function(V0)
        v0.vector.set(0.)
        interior_cells = self.cells_tags.indices[np.where(self.cells_tags.values == 1)]
        cut_cells      = self.cells_tags.indices[np.where(self.cells_tags.values == 2)]
        Omega_h_cells = np.union1d(interior_cells, cut_cells)
        # Omega_h_cells = self.cells_tags.indices[np.where(self.cells_tags.values == 1)]
        # We do not need the dofs here since cells and DG0 dofs share the same indices in dolfinx
        v0.x.array[Omega_h_cells] = 1.

        with dfx.io.XDMFFile(mesh.comm, f"output/v0_{str(self.i).zfill(2)}.xdmf", "w") as of:
            of.write_mesh(mesh)
            of.write_function(v0)

        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)

        w = ufl.TrialFunction(FE_space)
        v = ufl.TestFunction(FE_space)

        Omega_h_n = self._compute_normal(mesh)

        dx = ufl.Measure("dx",
                         domain=mesh,
                         subdomain_data=self.cells_tags,
                         metadata={"quadrature_degree": quadrature_degree})

        dS = ufl.Measure("dS",
                         domain=mesh,
                         subdomain_data=self.facets_tags,
                         metadata={"quadrature_degree": quadrature_degree})
        
        """
        Bilinear form
        """
        stiffness = inner(grad(phi_disc * w), grad(phi_disc * v)) * dx(1) \
                  + inner(grad(phi_disc * w), grad(phi_disc * v)) * dx(2)
        boundary = inner(2. * avg(inner(grad(phi_disc * w), Omega_h_n) * v0),
                         2. * avg(phi_disc * v * v0)) * dS(4)
        penalization_facets = sigma * avg(h) * inner(jump(grad(phi_disc * w), n),
                                                     jump(grad(phi_disc * v), n)) * dS(2)
        penalization_cells = sigma * h**2 * inner(div(grad(phi_disc * w)),
                                                  div(grad(phi_disc * v))) * dx(2)
        # This term is useless (always zero, everybody hate it)
        # but PETSc complains if I don't include it
        useless = inner(w, v) * v0 * dx(3)

        a = stiffness \
            - boundary \
            + penalization_facets \
            + penalization_cells \
            + useless
        
        # names = ["stiffness", "boundary", "penalization_facets", "penalization_cells", "useless"]
        # terms = [ stiffness,   boundary,   penalization_facets,   penalization_cells,   useless ]
        # for name, term in zip(names, terms):
        #     print(name)
        #     form = dfx.fem.form(term)
        #     Mat = assemble_matrix(form)
        #     Mat.assemble()
        #     B = Mat.convert("dense")
        #     C = B.getDenseArray()
        #     print(np.max(np.abs(C)))
        # assert 1==2

        """
        Linear form
        """
        rhs = inner(f, phi_disc * v) * dx(1) \
            + inner(f, phi_disc * v) * dx(2)
        penalization_rhs = sigma * h**2 * inner(f, div(grad(phi_disc * v))) * dx(2)

        L = rhs \
            - penalization_rhs

        self.bilinear_form = dfx.fem.form(a)
        self.linear_form = dfx.fem.form(L)
        return v0
    
    def assembly(self):
        self.A = assemble_matrix(self.bilinear_form)
        self.A.assemble()
        self.b = assemble_vector(self.linear_form)
    
    def solve(self, wh):
        self.petsc_solver.setOperators(self.A)
        self.petsc_solver.solve(self.b, wh.vector)