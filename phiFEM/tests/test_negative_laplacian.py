import jax.numpy as jnp
import numpy as np
import pytest
from utils.derivatives import negative_laplacian

"""
Data_n° = ("Fct formula",      num_var, (lambda x: expression fct),                         (lambda x: expression negative laplacian))
"""
data_1  = ("x",                      2, (lambda x,y:   x),                                (lambda x: np.zeros_like(x[0])))
data_2  = ("x^2",                    2, (lambda x,y:   x**2),                             (lambda x: -2. * np.ones_like(x[0])))
data_3  = ("x^3",                    2, (lambda x,y:   x**3),                             (lambda x: -6. * x[0]))
data_4  = ("y",                      2, (lambda x,y:   y),                                (lambda x: np.zeros_like(x[1])))
data_5  = ("y^2",                    2, (lambda x,y:   y**2),                             (lambda x: -2. * np.ones_like(x[1])))
data_6  = ("y^3",                    2, (lambda x,y:   y**3),                             (lambda x: -6. * x[1]))
data_7  = ("z",                      3, (lambda x,y,z: z),                                (lambda x: np.zeros_like(x[2])))
data_8  = ("z^2",                    3, (lambda x,y,z: z**2),                             (lambda x: -2. * np.ones_like(x[2])))
data_9  = ("z^3",                    3, (lambda x,y,z: z**3),                             (lambda x: -6. * x[2]))
data_10 = ("x*y",                    2, (lambda x,y:   x*y),                              (lambda x: np.zeros_like(x[0])))
data_11 = ("x*y*z",                  3, (lambda x,y,z: x*y*z),                            (lambda x: np.zeros_like(x[0])))
data_12 = ("x^3 + y^2*z + z^4",      3, (lambda x,y,z: x**3 + y**3*z**2 + z**4),          (lambda x: - 6. * x[0] - 6. * x[1] * x[2]**2 - 2.*x[1]**3 - 12.*x[2]**2))
data_13 = ("sin(x)*sin(y)",          2, (lambda x,y:   jnp.sin(x)*jnp.sin(y)),            (lambda x: 2. * jnp.sin(x[0])*jnp.sin(x[1])))
data_14 = ("sin(x)*sin(y)*sin(z)",   3, (lambda x,y,z: jnp.sin(x)*jnp.sin(y)*jnp.sin(z)), (lambda x: 3. * jnp.sin(x[0])*jnp.sin(x[1])*jnp.sin(x[2])))
data_15 = ("(y/x)*exp(-1/(10*x^2))", 2, (lambda x,y:   jnp.exp(-x**2/10.)*y/x),           (lambda x: -jnp.exp(-x[0]**2/10.)*(50. + 5.*x[0]**2+x[0]**4)*x[1]/(25.*x[0]**3)))

testdata = [data_1,  data_2,  data_3,  data_4,  data_5,
            data_6,  data_7,  data_8,  data_9,  data_10,
            data_11, data_12, data_13, data_14, data_15 ]
# testdata = [data_2]

@pytest.mark.parametrize("data_name,num_var,func,exact_nlap", testdata)
def test_negative_laplacian(data_name, num_var, func, exact_nlap):
    x = np.random.uniform(low=-1.0, high=1.0, size=(100000, 3))

    nlap = negative_laplacian(func)
    try:
        err_max = np.max(np.abs(nlap(x) - exact_nlap(x)))
        assert np.isclose(err_max, 0.)
    except AssertionError:
        print("Erreur max: ", err_max)

    if num_var == 2:
        x = np.random.uniform(low=-1.0, high=1.0, size=(100000, 2))
        nlap = negative_laplacian(func)
        try:
            err_max = np.max(np.abs(nlap(x) - exact_nlap(x)))
            assert np.isclose(err_max, 0.)
        except AssertionError:
            print("Erreur max: ", err_max)