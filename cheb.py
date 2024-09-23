import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap, jacfwd
import scipy as sp
from scipy import fft, special
import numpy as np
import orthax as ox
import math as mt
import scipy.linalg as scla
import sklearn as sk
import sklearn.neighbors

'''
def gridpts(N, with_weights=False):
    # chebyshev nodes of second kind
    # see "A MATLAB Differentiation Matrix Suite" by Weideman and Reddy
    x = jnp.sin(np.pi * ((N - 1) - 2 * jnp.linspace(N - 1, 0, N)) / (2 * (N - 1)))  # W&R way
    if with_weights:
        weights = (jnp.pi / (N-1)) * jnp.sin( jnp.pi * jnp.arange(N) / (N-1) )**2
        return x[::-1], weights
    else:
        return x[::-1]
    '''

def gridpts(N, with_weights=False):
    x = jnp.cos(jnp.pi * (2 * jnp.arange(N) + 1) / (2*N))
    if with_weights:
        weights = (2/N) * np.ones_like(x)
        return x, weights
    else:
        return x

def collocate_D(N):
    M = 1
    DM = np.zeros((M, N, N))

    # n1 = (N/2); n2 = round(N/2.)     # indices used for flipping trick [Original]
    n1 = mt.floor(N / 2)
    n2 = mt.ceil(N / 2)  # indices used for flipping trick [Corrected]
    k = np.arange(N)  # compute theta vector
    th = k * np.pi / (N - 1)

    # Assemble the differentiation matrices
    T = np.tile(th / 2, (N, 1))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)  # trigonometric identity
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))  # flipping trick
    DX[range(N), range(N)] = 1.  # diagonals of D
    DX = DX.T

    C = scla.toeplitz((-1.) ** k)  # matrix with entries c(k)/c(j)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1. / DX  # Z contains entries 1/(x(k)-x(j))
    Z[range(N), range(N)] = 0.  # with zeros on the diagonal.

    D = np.eye(N)  # D contains differentiation matrices.
    for ell in range(M):
        D = (ell + 1) * Z * (C * np.tile(np.diag(D), (N, 1)).T - D)  # off-diagonals
        D[range(N), range(N)] = -np.sum(D, axis=1)  # negative sum trick
        DM[ell, :, :] = D  # store current D in DM
    return jnp.asarray(DM[0])

def a(x):
    return jnp.asarray(x)

def to_symmetric(gridpts):
    return 2*gridpts - 1

def to_unit(gridpts):
    return 0.5 + gridpts/2

def collocate_M(a):
    return jnp.diag(a)
def dcht(u):
    N = len(u)
    scale = jnp.ones((N,)) + jnp.concatenate((a([1]), a([0] * (N - 2)), a([1])))
    return sp.fft.dct(u / (N - 1), norm="backward", type=1) / scale
def idcht(u):
    N = len(u)
    scale = jnp.ones((N,)) + jnp.concatenate((a([1]), a([0] * (N - 2)), a([1])))
    return (N - 1) * sp.fft.idct(u * scale, norm="backward", type=1)

def eval_on_grid(func, N_grid, dim, flattened=True, use_grid=None):
    if use_grid is None:
        one_d_grid = gridpts(N_grid)
    else:
        one_d_grid = use_grid
        N_grid = len(one_d_grid)

    grid = jnp.stack(jnp.meshgrid(*dim*[one_d_grid]), axis=-1).reshape((-1, dim))
    func = vmap(func, in_axes=0)
    if flattened:
        return func(grid)
    else:
        return func(grid).reshape(dim*[N_grid])

class ChebInterpolator1D():
    def __init__(self, values_on_cheb_grid):
        self.coeffs = dcht(values_on_cheb_grid)

    def __call__(self, inputs):
        return ox.chebyshev.chebval(inputs, self.coeffs)
class ChebInterpolator2D():
    def __init__(self, grid, vals, N_grid):
        '''
        Using the given values on the grid, estimate the values of the function on a chebyshev grid via KDE
        Then interpolate the estimated values with chebyshev polynomials

        :param grid: an N by D matrix of gridpoints where the function values are observed
        :param values_on_grid: a N, matrix of function values
        '''
        self.N_grid = N_grid
        self.dim = 2

        K = sk.neighbors.KNeighborsRegressor(n_neighbors=8, weights='distance')
        K.fit(grid, vals)

        ax_gridpts = gridpts(N_grid)
        dim_gridpts = jnp.stack(np.meshgrid(*self.dim*[ax_gridpts]), axis=-1).reshape((-1, self.dim))

        self.ax_gridpts = ax_gridpts
        self.dim_gridpts = dim_gridpts
        vals_on_cheb_grid = K.predict(dim_gridpts)


        I = jnp.arange(N_grid**2)
        idcs_indicator = jnp.zeros((N_grid, N_grid, N_grid**self.dim)).at[I//N_grid, I%N_grid, I].set(1)
        lagrange_to_cheb = ox.chebyshev.chebgrid2d(ax_gridpts, ax_gridpts, idcs_indicator).reshape((N_grid**2, N_grid**2))

        self.cheb_coeffs = jnp.linalg.solve(lagrange_to_cheb, vals_on_cheb_grid.flatten()).reshape((N_grid, N_grid))

    def eval(self, target_points):
        return ox.chebyshev.chebval2d(target_points[:, 1], target_points[:, 0], self.cheb_coeffs)

'''
class ChebInterpolator():
    def __init__(self, values_on_cheb_grid):
        shape = values_on_cheb_grid.shape
        N_grid = shape[0]
        assert all([s == shape[0] for s in shape]), "The input values must have the same number of gridpoints in each dimension"
        self.N_grid = N_grid
        self.dim = len(shape)
        

        one_d_gridpts = gridpts(N_grid)
        all_gridpts = np.stack(np.meshgrid(*self.dim*[one_d_gridpts]), axis=-1).reshape((-1, self.dim))
        self.all_gridpts = all_gridpts

        mi = mp.MultiIndexSet.from_degree(spatial_dimension=self.dim,
                                          poly_degree=N_grid)
        lag_poly = mp.LagrangePolynomial(mi)

        coeffs_newton = mp.get_transformation(lag_poly, mp.NewtonPolynomial).transformation_operator.array_repr_full
        exponents = lag_poly.multi_index.exponents
        generating_points = lag_poly.grid.generating_points
        basis_transf = mp.utils.eval_newton_polynomials(gridpts, coeffs_newton, exponents, generating_points)

        coeffs_lagrange, _, _, _ = jnp.linalg.lstsq(basis_transf, values_on_cheb_grid.flatten())
        lag_poly = mp.LagrangePolynomial(mi, coeffs_lagrange)
        l2n = mp.get_transformation(lag_poly, mp.NewtonPolynomial)
        self.poly = l2n()

    def __call__(self, target_points):
        return self.poly(target_points)
'''

if __name__ == "__main__":

    key = jr.key(641)
    N = 50
    xn = gridpts(N)

    for m in range(0, N):
        uhat = jnp.zeros((N,)).at[m].set(1)
        tm = jnp.cos(m * jnp.arccos(xn))
        tm_ = idcht(uhat)
        print(f"Mode {m} absolute error {jnp.max(jnp.abs(tm - tm_))}")

    D = collocate_D(xn)
    u = jnp.cos(xn)
    upr = -jnp.sin(xn)
    upr_ = collocate_D(xn) @ u
    assert jnp.allclose(upr, upr_)
    u = jr.normal(key, shape=(N,))

    assert jnp.allclose(u, idcht(dcht(u)))