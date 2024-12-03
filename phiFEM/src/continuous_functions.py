import dolfinx as dfx
from phiFEM.src.derivatives import negative_laplacian, compute_gradient


class ContinuousFunction:
    def __init__(self, expression):
        self.expression = expression
        self.interpolations = {}
    
    def __call__(self, x, y):
        return self.expression(x, y)
    
    def dolfinx_call(self, x):
        return self(x[0], x[1])
    
    def interpolate(self, FE_space):
        element = FE_space.element
        if element not in self.interpolations.keys():
            interpolation = dfx.fem.Function(FE_space)
            interpolation.interpolate(self.dolfinx_call)
            self.interpolations[element] = interpolation
        return self.interpolations[element]
            
class Levelset(ContinuousFunction):
    def exterior(self, t, padding=0.):
        """ Compute a lambda function determining if the point x is outside the domain defined by the isoline of level t.
        
        Args:
            t (float): level of the isoline.
            padding (float): padding parameter.
        
        Return:
            lambda function taking a tuple of coordinates and returning a boolean 
        """
        return lambda x: self(x[0], x[1]) > t + padding
    
    def interior(self, t, padding=0.):
        """ Compute a lambda function determining if the point x is inside the domain defined by the isoline of level t.
        
        Args:
            t (float): level of the isoline.
            padding (float): padding parameter.
        
        Return:
            lambda function taking a tuple of coordinates and returning a boolean 
        """
        return lambda x: self(x[0], x[1]) < t - padding
    
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