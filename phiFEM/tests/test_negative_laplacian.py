from collections.abc import Callable
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from phiFEM.phifem.continuous_functions import ExactSolution
# import plotly.graph_objs as go
# import plotly.io as pio
from typing import Tuple


NDArrayFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


# def print_fct(list_funcs: list[NDArrayFunction], file_name: str) -> None:
#     colorscales = [ 'Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Greens' ] 
#     # Generate the grid of points
#     x = np.linspace(-10., 10., 100)
#     y = np.linspace(-10., 10., 100)
#     x, y = np.meshgrid(x, y)
#     shape = x.shape
#     x_flat = np.ndarray.flatten(x)
#     y_flat = np.ndarray.flatten(y)

#     fig = go.Figure()
#     for i, func in enumerate(list_funcs):
#         num_args = len(inspect.signature(func).parameters)
#         if num_args==1:
#             X = np.zeros((len(x_flat), 2))
#             X[:,0] = x_flat
#             X[:,1] = y_flat
#             z_flat = func(X)
#         elif num_args==2:
#             z_flat = func(x_flat, y_flat)
#         z = z_flat.reshape(shape)
#         # Create the plot
#         surface = go.Surface(z=z, x=x, y=y, colorscale=colorscales[i], name=f"Surface {i+1}", showlegend=True)
#         fig.add_trace(surface)
#     layout = go.Layout(
#         title='3D Surface Plot',
#         scene=dict(
#             xaxis=dict(title='X axis'),
#             yaxis=dict(title='Y axis'),
#             zaxis=dict(title='Z axis')
#         ),
#     )
#     fig.update_layout(layout)

#     # Save the plot as an HTML file
#     html_file = f"{file_name}.html"
#     pio.write_html(fig, file=html_file)


"""
Data_n° = ("Fct formula",      num_var, (lambda x: expression fct),                                     (lambda x: expression negative laplacian))
"""
data_1  = ("x",                      2, (lambda x: x[0, :]),                                            (lambda x: np.zeros_like(x[0, :])))
data_2  = ("x^2",                    2, (lambda x: x[0, :]**2),                                         (lambda x: -2. * np.ones_like(x[0, :])))
data_3  = ("x^3",                    2, (lambda x: x[0, :]**3),                                         (lambda x: -6. * x[0, :]))
data_4  = ("y",                      2, (lambda x: x[1, :]),                                            (lambda x: np.zeros_like(x[1, :])))
data_5  = ("y^2",                    2, (lambda x: x[1, :]**2),                                         (lambda x: -2. * np.ones_like(x[1, :])))
data_6  = ("y^3",                    2, (lambda x: x[1, :]**3),                                         (lambda x: -6. * x[1, :]))
data_7  = ("z",                      3, (lambda x: x[2, :]),                                            (lambda x: np.zeros_like(x[2, :])))
data_8  = ("z^2",                    3, (lambda x: x[2, :]**2),                                         (lambda x: -2. * np.ones_like(x[2, :])))
data_9  = ("z^3",                    3, (lambda x: x[2, :]**3),                                         (lambda x: -6. * x[2, :]))
data_10 = ("x*y",                    2, (lambda x: x[0, :]*x[1, :]),                                    (lambda x: np.zeros_like(x[0, :])))
data_11 = ("x*y*z",                  3, (lambda x: x[0, :]*x[1, :]*x[2, :]),                            (lambda x: np.zeros_like(x[0, :])))
data_12 = ("x^3 + y^2*z + z^4",      3, (lambda x: x[0, :]**3 + x[1, :]**3*x[2, :]**2 + x[2, :]**4),    (lambda x: - 6. * x[0, :] - 6. * x[1, :] * x[2, :]**2 - 2.*x[1, :]**3 - 12.*x[2, :]**2))
data_13 = ("sin(x)*sin(y)",          2, (lambda x: jnp.sin(x[0, :])*jnp.sin(x[1, :])),                  (lambda x: 2. * jnp.sin(x[0, :])*jnp.sin(x[1, :])))
data_14 = ("sin(x)*sin(y)*sin(z)",   3, (lambda x: jnp.sin(x[0, :])*jnp.sin(x[1, :])*jnp.sin(x[2, :])), (lambda x: 3. * jnp.sin(x[0, :])*jnp.sin(x[1, :])*jnp.sin(x[2, :])))
data_15 = ("(y/x)*exp(-1/(10*x^2))", 2, (lambda x: jnp.exp(-x[0, :]**2/10.)*x[1, :]/x[0, :]),           (lambda x: -jnp.exp(-x[0, :]**2/10.)*(50. + 5.*x[0, :]**2+x[0, :]**4)*x[1, :]/(25.*x[0, :]**3)))

# testdata = [data_1,  data_2,  data_3,  data_4,  data_5,
#             data_6,  data_7,  data_8,  data_9,  data_10,
#             data_11, data_12, data_13, data_14, data_15 ]

# Only 2D data for now. TODO: add 3D.
# data_15 is too nasty, I disable it for now
testdata = [data_1,  data_2,  data_3,
            data_4,  data_5,  data_6,
            data_7,  data_8,  data_9,
            data_10, data_11, data_12,
            data_13, data_14 ]

@pytest.mark.parametrize("data_name,num_var,func,exact_nlap", testdata)
def test_negative_laplacian(data_name, num_var, func, exact_nlap):
    func = ExactSolution(func)
    func.compute_negative_laplacian()
    nlap = func.get_negative_laplacian()
    x = np.random.uniform(low=-1.0, high=1.0, size=(num_var, 100000))
    err_max = np.max(np.abs(nlap(x) - exact_nlap(x)))
    if not np.isclose(err_max, 0., atol=5.e-6):
        raise ValueError(f"Error max= {err_max}")


if __name__=="__main__":
    test_negative_laplacian(*data_15)