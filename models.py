import jax.numpy as jnp
import jax.random as jr
import neural_tangents as nt
from jax import grad, jit, vmap, jacfwd
import cheb
import scipy as sp
from scipy import fft, sparse
import numpy as np
import pde
from neural_tangents import stax
from typing import NamedTuple

def make_diffusion_field(n_spikes, dim, key):
    spikes = jr.normal(key, shape=(n_spikes, dim))
    return lambda x: jnp.sin(2 * jnp.pi * jnp.sum(spikes @ x.reshape((dim, -1)), axis=0))

class Poisson2D():
    def __init__(self, forcing_expr=None, N_grid_cheb=30, N_grid_fem=64):
        '''
        Forcing must be a pde.ScalarField
        :param forcing:
        :param N_grid_fem:
        '''

        grid = pde.CartesianGrid([[-1, 1], [-1, 1]], N_grid_fem)
        if forcing_expr is None:
            forcing_expr = "1 + sin(6.28*x) * sin(6.28*y)"

        self.forcing = pde.ScalarField.from_expression(grid, forcing_expr)

        bc = [{"value":0}, {"value": 0}]
        result = pde.solve_poisson_equation(self.forcing, [bc, bc])
        self._sol = result

        cheb_interp = cheb.ChebInterpolator2D(grid=result.grid.cell_coords.reshape((N_grid_fem**2, 2)), vals=result.data.flatten(), N_grid=N_grid_cheb)
        self.sol = cheb_interp

    def apply(self, func):
        return lambda x_: jnp.trace(jacfwd(lambda x: grad(func)(x))(x_))

    def eval_solution(self, X):
        return self.sol.eval(X).reshape((-1, 1))

class FickOp1D():
    def __init__(self, length_scale):
        raise NotImplementedError("This is bugged")
        self.length_scale = length_scale
        self.diffusion_field = lambda x: jnp.cos(2 * jnp.pi * x / length_scale)
        #self.diffusion_field = lambda x: 1

    def apply(self, func):
        op = grad(lambda x: self.diffusion_field(x) * grad(func)(x))
        return vmap(op, in_axes=(0,))

    def collocate(self, N, with_boundary_bordering=False):
        grid = cheb.gridpts(N)
        gfield = vmap(self.diffusion_field, in_axes=0)(grid)
        D = cheb.collocate_D(N)
        M = jnp.diag(gfield)
        if not with_boundary_bordering:
            return D @ M @ D
        else:
            gauge = jnp.ones((1,N))
            #right_deriv = D[0:1]
            return jnp.concatenate((D@M@D, gauge), axis=0)

    def solve(self, N, forcing, gauge=1):
        gOp = self.collocate(N, with_boundary_bordering=True)
        gForcing = forcing(cheb.gridpts(N))
        forcing = jnp.concatenate((gForcing.flatten(), jnp.array([gauge])))
        u, res, rank, _ = jnp.linalg.lstsq(gOp, forcing, rcond=1e-10)
        return u

class FickOp():
    def __init__(self, spikes, length_scale):
        raise NotImplementedError("This is bugged")
        assert spikes.shape[-1] == 2, "spikes must be two dimensional"
        self.spikes = spikes
        self.length_scale = length_scale
        #self.diffusion_field = lambda x: jnp.cos(2 * jnp.pi * (jnp.sum(spikes @ x.reshape((2, 1)))) / length_scale)
        self.diffusion_field = lambda x: 1

    def apply(self, func):
        op = jacfwd(lambda x: self.diffusion_field(x) * grad(func)(x))
        return vmap(lambda x: jnp.trace(op(x)), in_axes=(0,))

    def dx(self, N_grid):
        one_d_grid = cheb.gridpts(N_grid)

        #jax lacks support for sparse matrices UGH
        Id = jnp.eye(N_grid)
        D = cheb.collocate_D(one_d_grid)
        Dx = jnp.kron(D, Id)
        return Dx

    def operator(self, N_grid):
        one_d_grid = cheb.gridpts(N_grid)
        grid = jnp.stack(jnp.meshgrid(one_d_grid, one_d_grid), axis=-1).reshape((-1, 2))

        _field = vmap(self.diffusion_field, in_axes=0)
        field_grid = _field(grid)

        #jax lacks support for sparse matrices UGH
        Id = jnp.eye(N_grid)
        D = cheb.collocate_D(one_d_grid)
        Dx = jnp.kron(D, Id)
        Dy = jnp.kron(Id, D)
        M = jnp.diag(field_grid)
        operator = Dx @ M @ Dx + Dy @ M @ Dy
        return operator

    def solve(self, forcing_func, N_grid, dirichlet=None, boundary=None):
        '''
        solve the equation
        div * (a(x) * grad u) = forcing
        u(x) = boundary     x in boundary
        u'(x) = dirichlet   x in boundary

        where the domain is [0, 1]^2

        :param forcing_func: forcing function
        :param N_grid: order of chebyshev solver to use
        :param dirichlet: dirichlet boundary term
        :return: solution vector on the grid
        '''
        if dirichlet is not None or boundary is not None:
            raise NotImplementedError('only zero boundary conditions are implemented')

        one_d_grid = cheb.gridpts(N_grid)
        grid = jnp.stack(jnp.meshgrid(one_d_grid, one_d_grid), axis=-1).reshape((-1, 2))
        _forcing_func = vmap(forcing_func, in_axes=0)
        forcing_grid = _forcing_func(grid)

        _field = vmap(self.diffusion_field, in_axes=0)
        field_grid = _field(grid)

        #jax lacks support for sparse matrices UGH
        Id = jnp.eye(N_grid)
        D = cheb.collocate_D(one_d_grid)
        Dx = jnp.kron(D, Id)
        Dy = jnp.kron(Id, D)
        M = jnp.diag(field_grid)
        operator = Dx @ M @ Dx + Dy @ M @ Dy
        sol = jnp.linalg.solve(operator,forcing_grid)
        return sol

class Kernel2:
    def __init__(self, kfunc, op):
        self.kfunc = kfunc
        self.op = op
    def K(self, X, Y):
        return self.kfunc(X, Y)
    def H(self, X, Y):
        idop = lambda x, y: self.op.apply(lambda y_: self.kfunc(x, y_))(y)

        R = jnp.zeros((len(X), len(Y)))
        for j in range(len(Y)):
            Rj = jit(vmap(lambda x: idop(x, Y[j])))(X)
            R = R.at[:, j].set(Rj)

        return R

    def G(self, X, Y):
        idop = lambda x, y: self.op.apply(lambda y_: self.kfunc(x, y_))(y)
        opop = lambda x, y: self.op.apply(lambda x_: idop(x_, y))(x)
        '''
        TODO: the following uses up so much memory the process gets terminated
        but the for loop works fine and uses negligible extra memory
        h_over_X = lambda y: vmap(lambda x: opop(x, y))(X)
        R = vmap(h_over_X)(Y)
        '''
        R = jnp.zeros((len(X), len(Y)))
        for j in range(len(Y)):
            Rj = jit(vmap(lambda x: opop(x, Y[j])))(X)
            R = R.at[:, j].set(Rj)
        return R
class Kernel:
    def __init__(self, kfunc, op):
        self.kfunc = kfunc
        self.__call__ = vmap(self.kfunc)

        kfy = lambda x: lambda y: self.kfunc(x, y)
        self.hfunc = lambda x, y: op.apply(kfy(x))(y)
        self.gfunc = lambda x, y: op.apply(lambda x_: op.apply(lambda y_: self.kfunc(x_, y_))(y))(x)

    def _gramify(self, X, Y):
        return jnp.stack(jnp.meshgrid(X, Y), axis=-1).reshape((len(X) * len(Y), -1))

    def gram(self, X, Y):
        g = self._gramify(X, Y)
        return self.__call__(g[:, 0], g[:, 1]).reshape((len(X), len(Y)))

    def K(self, X, Y):
        return vmap(lambda x_: vmap(lambda y_: self.kfunc(x_, y_))(Y))(X)

    def H(self, X, Y):
        return vmap(lambda x_: vmap(lambda y_: self.hfunc(x_, y_))(Y))(X)
    def G(self, X, Y):
        return vmap(lambda x_: vmap(lambda y_: self.gfunc(x_, y_))(Y))(X)


class Cauchy(Kernel):
    def __init__(self, variance, op, eps=1e-8):
        kfunc = lambda x, y: (2 * jnp.pi)**(-1) * jnp.log(jnp.sum((x-y)**2/variance+eps)**(0.5))
        super().__init__(kfunc, op)
        self.variance = variance
class RBF(Kernel):
    def __init__(self, variance, op):
        kfunc = lambda x, y: jnp.exp(- 0.5 * variance * jnp.sum((x-y)**2))
        super().__init__(kfunc, op)
        if type(op) == Poisson2D:
            def hfunc(x, y):
                sdiff = jnp.linalg.norm(x-y)**2 / variance
                return (sdiff - 2) * kfunc(x, y) / variance
            def gfunc(x, y):
                sdiff = jnp.linalg.norm(x-y)**2 / variance
                return 2*(4 - sdiff)*kfunc(x, y)/(variance**2) + sdiff * (6 - sdiff) * kfunc(x, y) / (variance**2)
            self.hfunc = hfunc
            self.gfunc = gfunc
            
class DenseNNGP(Kernel):
    def __init__(self, width, depth, op):
        layers = []
        for _ in range(depth):
            layers.extend([stax.Dense(width), stax.Gelu()])
        layers.append(stax.Dense(1))

        _, _, _kfunc = stax.serial(*layers)
        kfunc = lambda x, y: _kfunc(x[None, :], y[None, :], 'nngp')[0,0]
        super().__init__(kfunc, op)

class ResNNGP(Kernel):
    def __init__(self, width, depth, op, parameter_mode="init"):
        layers = []
        for _ in range(depth):
            layers.append(stax.parallel(stax.serial(stax.Dense(width), stax.Gelu()), stax.Identity()))
        layers.append(stax.Dense(1))

        _, _, _kfunc = stax.serial(*layers)
        kfunc = lambda x, y: _kfunc(x[None, :], y[None, :], 'nngp')
        super().__init__(kfunc, op)


class NNet(NamedTuple):
    apply_fn: object
    key: object

def make_mlp(dims, key):
    layers = []
    for width in dims[1:-1]:
        layers.extend([stax.Dense(width, b_std=1.0), stax.Gelu()])
    layers.append(stax.Dense(dims[-1], b_std=1.0))

    init_fn, apply_fn, _ = stax.serial(*layers)
    parameters = init_fn(key, input_shape=(dims[0],))[1]
    return NNet(apply_fn=apply_fn, key=key), parameters


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    func = lambda x: jnp.sin(2 * jnp.pi * x[0]) + x[1] ** 2
    tonp = lambda x: np.asarray(x)

    N_grid = 30
    F = FickOp(spikes = jnp.zeros((1, 2)), length_scale=0.1)

    R = F.solve(func, N_grid=N_grid)

    one_d_grid = cheb.gridpts(N_grid)
    grid = jnp.stack(jnp.meshgrid(one_d_grid, one_d_grid), axis=-1).reshape((-1, 2))
    _forcing_func = vmap(func, in_axes=0)
    forcing_grid = _forcing_func(grid)

    print("Differentiation test")
    Dx = F.dx(N_grid)
    plt.subplot(1, 3, 1)
    plt.imshow(forcing_grid.reshape((N_grid, N_grid)))
    plt.subplot(1, 3, 2)

    func2 = lambda x: 2 * x[1]
    _forcing_func2 = vmap(func2, in_axes=0)
    forcing_grid2 = _forcing_func2(grid)
    plt.imshow(forcing_grid2.reshape((N_grid, N_grid)))

    plt.subplot(1, 3, 3)
    D = cheb.collocate_D(one_d_grid)

    plt.imshow( (jnp.kron(D, jnp.eye(N_grid)) @ forcing_grid ).reshape((N_grid, N_grid)))
    plt.show()

    print("Interpolation test")
    test_interpolator = cheb.ChebInterpolator2D(R.reshape((N_grid, N_grid)))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(R.reshape((N_grid, N_grid)))
    plt.subplot(1, 2, 2)
    plt.title("Interpolated")
    one_d_fine_grid = cheb.gridpts(50)
    fine_grid = jnp.stack(jnp.meshgrid(one_d_fine_grid, one_d_fine_grid), axis=-1).reshape((-1, 2))
    plt.imshow(test_interpolator.eval(fine_grid).reshape((100, 100)))

    plt.show()
    plt.subplot(1, 4, 1)
    plt.title("Forcing function")
    plt.imshow(forcing_grid.reshape((N_grid, N_grid)))
    plt.subplot(1, 4, 2)
    plt.title("Solution")
    plt.imshow(R.reshape(N_grid, N_grid))
    plt.subplot(1, 4, 3)
    plt.title("Op @ solution")
    M = F.operator(100)
    r = test_interpolator.eval(fine_grid)
    diffd_sol = M@r
    plt.imshow(diffd_sol.reshape((100, 100)))
    plt.subplot(1, 4, 4)
    plt.title("F @ interpolated solution")
    diffd_sol = F.apply(lambda x: test_interpolator.eval(x.reshape((-1, 2)))[0])(fine_grid)
    plt.imshow(diffd_sol.reshape((100, 100)))
    plt.show()
    print(R)