# This module implements various functions for the background COSMOLOGY
from typing import Tuple
import jax.numpy as np
from jax import lax
from jax import jit, grad
from functools import partial


import jax_cosmo.constants as const
from jax_cosmo.scipy.interpolate import interp
from jax_cosmo.scipy.ode import odeint
from jax_cosmo.utils import z2a
from jax_cosmo.core import Cosmology

__all__ = [
    "w",
    "f_de",
    "Esqr",
    "H",
    "Omega_m_a",
    "Omega_de_a",
    "radial_comoving_distance",
    "dchioverda",
    "transverse_comoving_distance",
    "angular_diameter_distance",
    "growth_factor",
    "growth_rate",
]


def w(cosmo, a):
    r"""Dark Energy equation of state parameter using the Linder
    parametrisation.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like
        Scale factor

    Returns
    -------
    w : ndarray, or float if input scalar
        The Dark Energy equation of state parameter at the specified
        scale factor

    Notes
    -----
    The Linder parametrization [Linder (2003)](https://arxiv.org/abs/astro-ph/0208512) for the Dark Energy
    equation of state $p = w \rho$ is given by:

    $$
    w(a) = w_0 + w_a (1 - a)
    $$
    """
    return cosmo.w0 + (1.0 - a) * cosmo.wa


def f_de(cosmo, a):
    r"""Evolution parameter for the Dark Energy density.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    f : ndarray, or float if input scalar
        The evolution parameter of the Dark Energy density as a function
        of scale factor

    Notes
    -----

    For a given parametrisation of the Dark Energy equation of state,
    the scaling of the Dark Energy density with time can be written as:

    .. math::

        \rho_{de}(a) = \rho_{de}(a=1) e^{f(a)}

    (see :cite:`2005:Percival` and note the difference in the exponent base
    in the parametrizations) where :math:`f(a)` is computed as
    :math:`f(a) = -3 \int_0^{\ln(a)} [1 + w(a')] d \ln(a')`.
    In the case of Linder's parametrisation for the dark energy
    in Eq. :eq:`linderParam` :math:`f(a)` becomes:

    .. math::

        f(a) = -3 (1 + w_0 + w_a) \ln(a) + 3 w_a (a - 1)
    """
    return -3.0 * (1.0 + cosmo.w0 + cosmo.wa) * np.log(a) + 3.0 * cosmo.wa * (a - 1.0)


def Esqr(cosmo, a):
    r"""Square of the scale factor dependent factor E(a) in the Hubble
    parameter.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    E^2 : ndarray, or float if input scalar
        Square of the scaling of the Hubble constant as a function of
        scale factor

    Notes
    -----

    The Hubble parameter at scale factor `a` is given by
    :math:`H^2(a) = E^2(a) H_o^2` where :math:`E^2` is obtained through
    Friedman's Equation (see :cite:`2005:Percival`) :

    .. math::

        E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} e^{f(a)}

    where :math:`f(a)` is the Dark Energy evolution parameter computed
    by :py:meth:`.f_de`.
    """
    return (
        cosmo.Omega_m * np.power(a, -3)
        + cosmo.Omega_k * np.power(a, -2)
        + (cosmo.Omega_g + cosmo.Omega_r) * np.power(a, -4)
        + cosmo.Omega_de * np.exp(f_de(cosmo, a))
    )


def H(cosmo, a):
    r"""Hubble parameter [km/s/(Mpc/h)] at scale factor `a`

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    H : ndarray, or float if input scalar
        Hubble parameter at the requested scale factor.
    """
    return const.H0 * np.sqrt(Esqr(cosmo, a))


def Omega_m_a(cosmo, a):
    r"""Matter density at scale factor `a`.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    Omega_m : ndarray, or float if input scalar
        Non-relativistic matter density at the requested scale factor

    Notes
    -----
    The evolution of matter density :math:`\Omega_m(a)` is given by:

    .. math::

        \Omega_m(a) = \frac{\Omega_m a^{-3}}{E^2(a)}

    see :cite:`2005:Percival` Eq. (6)
    """
    return cosmo.Omega_m * np.power(a, -3) / Esqr(cosmo, a)


def Omega_de_a(cosmo, a):
    r"""Dark Energy density at scale factor `a`.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    Omega_de : ndarray, or float if input scalar
        Dark Energy density at the requested scale factor

    Notes
    -----
    The evolution of Dark Energy density :math:`\Omega_{de}(a)` is given
    by:

    .. math::

        \Omega_{de}(a) = \frac{\Omega_{de} e^{f(a)}}{E^2(a)}

    where :math:`f(a)` is the Dark Energy evolution parameter computed by
    :py:meth:`.f_de` (see :cite:`2005:Percival` Eq. (6)).
    """
    return cosmo.Omega_de * np.exp(f_de(cosmo, a)) / Esqr(cosmo, a)


def radial_comoving_distance(cosmo, a, log10_amin=-4, steps=512):
    r"""Radial comoving distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    chi : ndarray, or float if input scalar
        Radial comoving distance corresponding to the specified scale
        factor.

    Notes
    -----
    The radial comoving distance is computed by performing the following
    integration:

    .. math::

        \chi(a) =  R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}
    """
    # Check if distances have already been computed
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def dchioverdlna(y, x):
            xa = np.exp(x)
            return dchioverda(cosmo, xa) * xa

        chitab = odeint(dchioverdlna, 0.0, np.log(atab))
        # np.clip(- 3000*np.log(atab), 0, 10000)#odeint(dchioverdlna, 0., np.log(atab), cosmo)
        chitab = chitab[-1] - chitab

        cache = {"a": atab, "chi": chitab}
        cosmo._workspace["background.radial_comoving_distance"] = cache
    else:
        cache = cosmo._workspace["background.radial_comoving_distance"]

    a = np.atleast_1d(a)
    # Return the results as an interpolation of the table
    return np.clip(interp(a, cache["a"], cache["chi"]), 0.0)

def luminosity_distance(cosmo, a, log10_amin=-4, steps=512):
    """
    Computes the luminosity distance for a given cosmological model and scale factor.

    Parameters:
    ----------
    cosmo : object
        Cosmological model object containing relevant parameters (e.g., Hubble constant, matter density).
    a : float
        Scale factor of the Universe (inverse of 1 + redshift).
    log10_amin : float, optional
        Logarithm (base 10) of the minimum scale factor to consider, default is -4 (corresponding to very high redshift).
    steps : int, optional
        Number of integration steps for computing the radial comoving distance, default is 512.

    Returns:
    -------
    float
        Luminosity distance in units of Mpc/h (or physical distance divided by the reduced Hubble constant).
    """
    comoving_radial = radial_comoving_distance(cosmo, a, log10_amin, steps) / cosmo.h
    return comoving_radial / a


def distance_modulus(cosmo, a, log10_amin=-4, steps=512):
    """
    Computes the distance modulus, a measure of the difference between the apparent and absolute magnitudes
    of an astronomical object.

    Parameters:
    ----------
    cosmo : object
        Cosmological model object containing relevant parameters (e.g., Hubble constant, matter density).
    a : float
        Scale factor of the Universe (inverse of 1 + redshift).
    log10_amin : float, optional
        Logarithm (base 10) of the minimum scale factor to consider, default is -4 (corresponding to very high redshift).
    steps : int, optional
        Number of integration steps for computing the radial comoving distance, default is 512.

    Returns:
    -------
    float
        Distance modulus in magnitudes, which quantifies the difference between the apparent and absolute
        magnitudes of an object.
    """
    lum_dist = luminosity_distance(cosmo, a, log10_amin, steps)
    return 5.0 * np.log10(lum_dist * 10) + 20.0

def a_of_chi(cosmo, chi):
    r"""Computes the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation.

    Parameters:
    -----------
    cosmo: Cosmology
      Cosmological parameters

    chi: array-like
      radial comoving distance to query.

    Returns:
    --------
    a : array-like
      Scale factors corresponding to requested distances
    """
    # Check if distances have already been computed, force computation otherwise
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        radial_comoving_distance(cosmo, 1.0)
    cache = cosmo._workspace["background.radial_comoving_distance"]
    chi = np.atleast_1d(chi)
    return interp(chi, cache["chi"], cache["a"])


def dchioverda(cosmo, a):
    r"""Derivative of the radial comoving distance with respect to the
    scale factor.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    dchi/da :  ndarray, or float if input scalar
        Derivative of the radial comoving distance with respect to the
        scale factor at the specified scale factor.

    Notes
    -----

    The expression for :math:`\frac{d \chi}{da}` is:

    .. math::

        \frac{d \chi}{da}(a) = \frac{R_H}{a^2 E(a)}
    """
    return const.rh / (a**2 * np.sqrt(Esqr(cosmo, a)))


def transverse_comoving_distance(cosmo, a):
    r"""Transverse comoving distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    f_k : ndarray, or float if input scalar
        Transverse comoving distance corresponding to the specified
        scale factor.

    Notes
    -----
    The transverse comoving distance depends on the curvature of the
    universe and is related to the radial comoving distance through:

    $$
        f_k(a) = \left\lbrace
        \begin{matrix}
        R_H \frac{1}{\sqrt{\Omega_k}}\sinh(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for }\Omega_k > 0 \\
        \chi(a)&
            \mbox{for } \Omega_k = 0 \\
        R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for } \Omega_k < 0
        \end{matrix}
        \right.
    $$
    """
    index = cosmo.k + 1

    def open_universe(chi):
        return const.rh / cosmo.sqrtk * np.sinh(cosmo.sqrtk * chi / const.rh)

    def flat_universe(chi):
        return chi

    def close_universe(chi):
        return const.rh / cosmo.sqrtk * np.sin(cosmo.sqrtk * chi / const.rh)

    branches = (open_universe, flat_universe, close_universe)

    chi = radial_comoving_distance(cosmo, a)

    return lax.switch(cosmo.k + 1, branches, chi)


def angular_diameter_distance(cosmo, a):
    r"""Angular diameter distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    a : array_like
        Scale factor

    Returns
    -------
    d_A : ndarray, or float if input scalar

    Notes
    -----
    Angular diameter distance is expressed in terms of the transverse
    comoving distance as:

    .. math::

        d_A(a) = a f_k(a)
    """
    return a * transverse_comoving_distance(cosmo, a)


def growth_factor(cosmo, a):
    r"""Compute linear growth factor D(a) at a given scale factor,
    normalized such that D(a=1) = 1.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization, for
    instance if the $\gamma$ parameter is defined, the growth will be computed
    assuming the $f = \Omega^\gamma$ growth rate, otherwise the usual ODE for
    growth will be solved.
    """
    if cosmo._flags["gamma_growth"]:
        return _growth_factor_gamma(cosmo, a)
    else:
        return _growth_factor_ODE(cosmo, a)


def growth_rate(cosmo, a):
    r"""Compute growth rate dD/dlna at a given scale factor.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f:  ndarray, or float if input scalar
        Growth rate computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization, for
    instance if the $\gamma$ parameter is defined, the growth will be computed
    assuming the $f = \Omega^\gamma$ growth rate, otherwise the usual ODE for
    growth will be solved.

    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math: `\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`
    """
    if cosmo._flags["gamma_growth"]:
        return _growth_rate_gamma(cosmo, a)
    else:
        return _growth_rate_ODE(cosmo, a)


def grad_H(cosmo: Cosmology, a: float) -> np.array:
    """Calculates the first derivative of the Hubble parameter at scale factor a.

    Args:
        cosmo (Cosmology): a Cosmology object
        a (float): the scale factor a

    Returns:
        np.array: derivative of the Hubble parameter with respect to a
    """
    return grad(H, argnums=1)(cosmo, a)


def alpha_beta(cosmo: Cosmology, a: float) -> np.array:
    """Calculates the matrix which maps the first derivative and the answer we want, that is,

    y' = A y

    y' is a vector which contains g'(a) and g"(a) while y is a vector which contains g(a) and g'(a).

    A is the matrix we calculate here and we call -A[1,0] as beta and -A[1,1] as alpha.

    Args:
        cosmo (Cosmology): a Cosmology object
        a (float): the scale factor

    Returns:
        np.array: the matrix A according to the equation described above.
    """
    gH_over_H = grad_H(cosmo, a) / H(cosmo, a)
    alpha = 5.0 / a + gH_over_H
    beta = 3 / a**2 + gH_over_H / a - 1.5 * Omega_m_a(cosmo, a) / a**2
    return np.array([[0.0, 1.0], [-beta, -alpha]])


def _growth_factor_ODE(cosmo, a, log10_amin=-3, steps=128, eps=1e-4):
    """Compute linear growth factor D(a) at a given scale factor,
    normalised such that D(a=1) = 1.

    Parameters
    ----------
    a: array_like
      Scale factor

    amin: float
      Mininum scale factor, default 1e-3

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor
    """
    # Check if growth has already been computed
    if not "background.growth_factor" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def D_derivs(y, x):
            q = (
                2.0
                - 0.5
                * (
                    Omega_m_a(cosmo, x)
                    + (1.0 + 3.0 * w(cosmo, x)) * Omega_de_a(cosmo, x)
                )
            ) / x
            r = 1.5 * Omega_m_a(cosmo, x) / x / x
            return np.array([y[1], -q * y[1] + r * y[0]])

        y0 = np.array([atab[0], 1.0])
        y = odeint(D_derivs, y0, atab)
        y1 = y[:, 0]
        gtab = y1 / y1[-1]
        # To transform from dD/da to dlnD/dlna: dlnD/dlna = a / D dD/da
        ftab = y[:, 1] / y1[-1] * atab / gtab

        cache = {"a": atab, "g": gtab, "f": ftab}
        cosmo._workspace["background.growth_factor"] = cache
    else:
        cache = cosmo._workspace["background.growth_factor"]
    return np.clip(interp(a, cache["a"], cache["g"]), 0.0, 1.0)


def _growth_rate_ODE(cosmo, a):
    """Compute growth rate dD/dlna at a given scale factor by solving the linear
    growth ODE.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f:  ndarray, or float if input scalar
        Growth rate computed at requested scale factor
    """
    # Check if growth has already been computed, if not, compute it
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return interp(a, cache["a"], cache["f"])


def _growth_factor_gamma(cosmo, a, log10_amin=-3, steps=128):
    r"""Computes growth factor by integrating the growth rate provided by the
    \gamma parametrization. Normalized such that D( a=1) =1

    Parameters
    ----------
    a: array_like
      Scale factor

    amin: float
      Mininum scale factor, default 1e-3

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor

    """
    # Check if growth has already been computed, if not, compute it
    if not "background.growth_factor" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def integrand(y, loga):
            xa = np.exp(loga)
            return _growth_rate_gamma(cosmo, xa)

        gtab = np.exp(odeint(integrand, np.log(atab[0]), np.log(atab)))
        gtab = gtab / gtab[-1]  # Normalize to a=1.
        cache = {"a": atab, "g": gtab}
        cosmo._workspace["background.growth_factor"] = cache
    else:
        cache = cosmo._workspace["background.growth_factor"]
    return np.clip(interp(a, cache["a"], cache["g"]), 0.0, 1.0)


def _growth_rate_gamma(cosmo, a):
    r"""Growth rate approximation at scale factor `a`.

    Parameters
    ----------
    cosmo: `Cosmology`
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    f_gamma : ndarray, or float if input scalar
        Growth rate approximation at the requested scale factor

    Notes
    -----
    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math: `\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`
    """
    return Omega_m_a(cosmo, a) ** cosmo.gamma


def scale_of_chi(
    cosmo: Cosmology, z_min: float, z_max: float, n_z: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the scale factor given a minimum redshift and a maximum redshift.
    This is done over a regular space on the comoving radial distance axis.

    Args:
        cosmo (Cosmology): the cosmology object (set of parameters)
        z_min (float): the minimum redshift
        z_max (float): the maximum redshift
        n_z (int): the number of redshifts

    Returns:
        Tuple[np.ndarray, np.ndarray]: the scalefactors and the comoving radial distance
    """
    a_min = z2a(z_min)
    a_max = z2a(z_max)
    chi_min = radial_comoving_distance(cosmo, a=a_min).item()
    chi_max = radial_comoving_distance(cosmo, a=a_max).item()
    chi_arr = np.linspace(chi_min, chi_max, n_z)
    a_arr = a_of_chi(cosmo, chi_arr)
    return a_arr, chi_arr
