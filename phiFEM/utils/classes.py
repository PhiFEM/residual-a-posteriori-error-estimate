from utils.derivatives import negative_laplacian, compute_gradient

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
        func = lambda x, y: self.__call__(x, y) # Dirty workaround because negative_laplacian looks for the number of arguments in order to determine the dimension and "self" messes up the count.
        return compute_gradient(func)

class ExactSolution(ContinuousFunction):
    def negative_laplacian(self):
        func = lambda x, y: self.__call__(x, y) # Dirty workaround because negative_laplacian looks for the number of arguments in order to determine the dimension and "self" messes up the count.
        return negative_laplacian(func)
    
class PhiFEMSolver:
    def __init__(self, PETSc_solver):
        self.petsc_solver = PETSc_solver
    
    def set_variational_formulation(FE_space, levelset, mesh_tags, quadrature_degree=None):
        if quadrature_degree is None:
            quadrature_degree = 2 * (FE_space.element.basix_element.degree + 1)
        
        w = ufl.TrialFunction(FE_space)
        v = ufl.TestFunction(FE_space)

        h = ufl.CellDiameter(FE_space.mesh)
        n = ufl.FacetNormal(FE_space.mesh)

        dx = ufl.Measure("dx",
                         domain=mesh,
                         subdomain_data=mesh_tags[0],
                         metadata={"quadrature_degree": quadrature_degree})

        dS = ufl.Measure("dS",
                        domain=mesh,
                        subdomain_data=mesh_tags[1],
                        metadata={"quadrature_degree": quadrature_degree})

