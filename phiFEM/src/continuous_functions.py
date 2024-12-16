import dolfinx as dfx
from phiFEM.src.derivatives import negative_laplacian, compute_gradient


class ContinuousFunction:
    """ Class to represent a continuous (in the sense of non-discrete) function."""

    def __init__(self, expression):
        """ Intialize a continuous function.

        Args:
            expression: a method giving the expression of the continuous function.
        """

        self.expression = expression
        self.interpolations = {}
    
    def __call__(self, x, y):
        """ Call the continuous function for computation.

        Args:
            x: array, x coordinates of the points in which the function is computed.
            y: array, y coordinates of the points in which the function is computed.
        
        Returns:
            The array of values of the function.
        """
        return self.expression(x, y)
    
    def dolfinx_call(self, x):
        """ Call the continuous function for computation, but using a single (vectorial) input.

        Args:
            x: ndarray, the coordinates of the points in which the function is computed.

        Returns:
            The array of values of the function.
        """
        return self(x[0], x[1])
    
    def interpolate(self, FE_space):
        """ Interpolate the function onto a finite element space.
        A dict is created in order to remember previous interpolations and save computational time.

        Args:
            FE_space: a finite element space in which the function will be interpolated.

        Returns:
            The dict of interpolations, with a new entry if needed.
        """
        element = FE_space.element
        if element not in self.interpolations.keys():
            interpolation = dfx.fem.Function(FE_space)
            interpolation.interpolate(self.dolfinx_call)
            self.interpolations[element] = interpolation
        return self.interpolations[element]
            
class Levelset(ContinuousFunction):
    """ Class to represent a levelset function as a continuous function."""

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
    
class ExactSolution(ContinuousFunction):
    """ Class to represent the exact solution of the PDE as a continuous function."""
    def compute_negative_laplacian(self):
        """ Compute the negative laplacian of the function."""
        def func(x, y):
            return self.__call__(x, y) # Dirty workaround because negative_laplacian looks for the number of arguments in order to determine the dimension and "self" messes up the count.
        comp_nlap = negative_laplacian(func)
        self.nlap = ContinuousFunction(lambda x, y: comp_nlap([x, y]))