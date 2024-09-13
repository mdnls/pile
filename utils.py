import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax import random
import scipy as sp
from scipy import fft

def gaussian_quadrature(dim, gridsize):
    ...

def make_diffusion_field(n_spikes, dim, key):
    spikes = jax.random.normal(key, shape=(n_spikes, dim))
    return lambda x: jnp.sin(2 * jnp.pi * jnp.sum(spikes @ x.reshape((dim, -1)), axis=0))


def dcht(u):
    N = len(u)
    scale = jnp.ones((N,)) + jnp.concatenate(([1], [0] * (N - 2), [1]))
    return sp.fft.dct(u / (N - 1), norm="backward", type=1) / scale


def idcht(u):
    N = len(u)
    scale = jnp.ones((N,)) + jnp.concatenate(([1], [0] * (N - 2), [1]))
    return (N - 1) * sp.fft.idct(u * scale, norm="backward", type=1)


def dcht(u):
    N = len(u)
    coeffs = sp.fft.dct(u / (N - 1), norm="backward", type=1)
    coeffs[0] = coeffs[0] / 2
    coeffs[-1] = coeffs[-1] / 2
    return coeffs



# inverse discrete chebyshev transform
def idcht(u, inplace=True):
    N = len(u)
    if (not inplace):
        u = jnp.copy(u)
    u[0] = 2 * u[0]
    u[-1] = 2 * u[-1]
    return (N - 1) * sp.fft.idct(u, norm="backward", type=1)


def gridpts(N):
    return jnp.cos(jnp.pi * jnp.arange(N) / (N - 1))