import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import factorial
from typing import Tuple

EULER = 0.577215664901532860606512090082402431  # Euler-Mascheroni constant


def compute_sici(
    xvalues: jnp.ndarray, nterms: int = 12, npoints: int = 2000, xmin: float = 1e-5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the sine integral (Si) and cosine integral (Ci) for given input values.

    Args:
        xvalues (jnp.ndarray): Array of input values.
        nterms (int, optional): Number of terms in the series expansion for large x. Defaults to 12.
        npoints (int, optional): Number of points for numerical integration in small x case. Defaults to 2000.
        xmin (float, optional): Minimum value for integration grid. Defaults to 1e-5.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Two arrays containing the computed Si and Ci values for each input.
    """
    xvalues = jnp.atleast_1d(xvalues)

    def small_x_case(x):
        grid = jnp.linspace(xmin, x, npoints)
        f_sin = jnp.sin(grid) / grid
        f_cos = (jnp.cos(grid) - 1) / grid

        si_calc = jnp.trapezoid(f_sin, grid)
        ci_calc = EULER + jnp.log(x) + jnp.trapezoid(f_cos, grid)
        return si_calc, ci_calc

    def large_x_case(x):
        grid = jnp.arange(nterms)
        xpow = x**grid
        fac = factorial(grid)
        ratio = fac / xpow

        even = jnp.sum(
            ratio[0::2] * jnp.tile(jnp.array([1.0, -1.0]), len(ratio[0::2]) // 2)
        )
        odd = jnp.sum(
            ratio[1::2] * jnp.tile(jnp.array([1.0, -1.0]), len(ratio[1::2]) // 2)
        )

        si_calc = 0.5 * jnp.pi - jnp.cos(x) / x * even - jnp.sin(x) / x * odd
        ci_calc = jnp.sin(x) / x * even - jnp.cos(x) / x * odd
        return si_calc, ci_calc

    def compute_single(x):
        return lax.cond(
            x < 10.0, lambda x: small_x_case(x), lambda x: large_x_case(x), x
        )

    compute_vectorized = vmap(compute_single)
    return compute_vectorized(xvalues)
