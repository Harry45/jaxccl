# This module implements various functions for the background COSMOLOGY
from typing import Tuple, Union
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
    "distance_modulus",
    "luminosity_distance",
]


def w(cosmo: Cosmology, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Computes the Dark Energy equation of state parameter using the Linder parametrization.

    Args:
        cosmo (Cosmology): Cosmological parameters structure containing `w0` and `wa`.
        a (Union[float, jnp.ndarray]): Scale factor. Can be a scalar or an array.

    Returns:
        The Dark Energy equation of state parameter at the specified scale factor. Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The Linder parametrization [Linder (2003)](https://arxiv.org/abs/astro-ph/0208512)
        for the Dark Energy equation of state $p = w \rho$ is given by:

        $w(a) = w_0 + w_a (1 - a)$
    """
    return cosmo.w0 + (1.0 - a) * cosmo.wa


def f_de(cosmo: Cosmology, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Computes the evolution parameter for the Dark Energy density.

    Args:
        cosmo (Cosmology): Cosmological parameters structure containing `w0` and `wa`.
        a (Union[float, np.ndarray]): Scale factor. Can be a scalar or an array.

    Returns:
        The evolution parameter of the Dark Energy density as a function of the scale factor. Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        For a given parametrization of the Dark Energy equation of state, the scaling of the
        Dark Energy density with time can be written as:

        $$
        \rho_{de}(a) = \rho_{de}(a=1) e^{f(a)}
        $$

        See [Percival (2005)](https://arxiv.org/pdf/astro-ph/0508156) and note the difference in
        the exponent base in the parametrizations where $f(a)$ is computed as:

        $$
        f(a) = -3 \int_0^{\ln(a)} [1 + w(a')] \, d \ln(a')
        $$

        In the case of Linder's parametrization for the Dark Energy equation of state, $f(a)$ becomes:

        $$
        f(a) = -3 (1 + w_0 + w_a) \ln(a) + 3 w_a (a - 1)
        $$
    """
    return -3.0 * (1.0 + cosmo.w0 + cosmo.wa) * np.log(a) + 3.0 * cosmo.wa * (a - 1.0)


def Esqr(cosmo: Cosmology, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Computes the square of the scale-factor-dependent term $E(a)$ in the Hubble parameter.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., matter density, curvature, radiation density).
        a (Union[float, np.ndarray]): Scale factor. Can be a scalar or an array.

    Returns:
        Square of the scaling of the Hubble constant as a function of the scale factor. Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The Hubble parameter at scale factor $a$ is given by:

        $$
        H^2(a) = E^2(a) H_0^2
        $$

        where $E^2(a)$ is obtained through Friedmann's Equation (see [Percival (2005)](https://arxiv.org/pdf/astro-ph/0508156)):

        $$
        E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} +  \Omega_r a^{-4} + \Omega_{de} e^{f(a)}
        $$

        Here, $f(a)$ is the Dark Energy evolution parameter.
    """
    return (
        cosmo.Omega_m * np.power(a, -3)
        + cosmo.Omega_k * np.power(a, -2)
        + cosmo.Omega_r * np.power(a, -4)
        + cosmo.Omega_de * np.exp(f_de(cosmo, a))
    )


def H(cosmo: Cosmology, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Computes the Hubble parameter $H(a)$ at a given scale factor $a$.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., matter density, curvature, radiation density).
        a (Union[float, np.ndarray]): Scale factor. Can be a scalar or an array.

    Returns:
        Hubble parameter at the requested scale factor $a$ in units of [km/s/(Mpc/h)]. Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The Hubble parameter is calculated as:

        $$
        H(a) = H_0 \sqrt{E^2(a)}
        $$

        where $H_0$ is the Hubble constant in [km/s/(Mpc/h)] and $E^2(a)$ is derived from Friedmann's Equation.
    """
    return const.H0 * np.sqrt(Esqr(cosmo, a))


def Omega_m_a(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the non-relativistic matter density $\Omega_m(a)$ at a given scale factor $a$.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., matter density, curvature).
        a (Union[float, np.ndarray]): Scale factor. Can be a scalar or an array.

    Returns:
        Non-relativistic matter density at the requested scale factor $a$. Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The evolution of matter density $\Omega_m(a)$ is given by:

        $$
        \Omega_m(a) = \frac{\Omega_m a^{-3}}{E^2(a)}
        $$

        where $\Omega_m$ is the present-day matter density parameter, and $E^2(a)$ is derived from Friedmann's Equation.
        For more details, see Equation 6 in [Percival (2005)](https://arxiv.org/pdf/astro-ph/0508156).
    """
    return cosmo.Omega_m * np.power(a, -3) / Esqr(cosmo, a)


def Omega_de_a(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the Dark Energy density $\Omega_{de}(a)$ at a given scale factor $a$.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., Dark Energy density, matter density).
        a (Union[float, np.ndarray]): Scale factor. Can be a scalar or an array.

    Returns:
        Dark Energy density at the requested scale factor $a$. Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The evolution of the Dark Energy density $\Omega_{de}(a)$ is given by:

        $$
        \Omega_{de}(a) = \frac{\Omega_{de} e^{f(a)}}{E^2(a)}
        $$

        where $\Omega_{de}$ is the present-day Dark Energy density parameter,
        $E^2(a)$ is derived from Friedmann's Equation, and $f(a)$ is the Dark Energy evolution parameter.

        For more details, see Equation 6 in [Percival (2005)](https://arxiv.org/pdf/astro-ph/0508156).
    """
    return cosmo.Omega_de * np.exp(f_de(cosmo, a)) / Esqr(cosmo, a)


def radial_comoving_distance(
    cosmo: Cosmology,
    a: Union[float, np.ndarray],
    log10_amin: float = -4,
    steps: int = 512,
) -> Union[float, np.ndarray]:
    r"""
    Computes the radial comoving distance $\chi(a)$ at a given scale factor $a$.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., Hubble constant, matter density, Dark Energy density).
        a (Union[float, np.ndarray]): Scale factor. Can be a scalar or an array.
        log10_amin (float, optional): Logarithm (base 10) of the minimum scale factor to consider.
            Default is -4, which corresponds to very high redshift.
        steps (int, optional): Number of integration steps for computing the radial comoving distance.
            Default is 512.

    Returns:
        Radial comoving distance $\chi(a)$ corresponding to the specified scale factor $a$. Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The radial comoving distance is computed by performing the following integration:

        $$
        \chi(a) = R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}
        $$

        where $R_H$ is the Hubble radius, and $E(a)$ is the function dependent on cosmological parameters
        (calculated from Friedmann's Equation).

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


def luminosity_distance(
    cosmo: Cosmology,
    a: Union[float, np.ndarray],
    log10_amin: float = -4,
    steps: int = 512,
) -> Union[float, np.ndarray]:
    r"""
    Computes the luminosity distance for a given cosmological model and scale factor.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., Hubble constant, matter density, Dark Energy density).
        a (Union[float, np.ndarray]): Scale factor of the Universe (inverse of 1 + redshift).
            Can be a scalar or an array.
        log10_amin (float, optional): Logarithm (base 10) of the minimum scale factor to consider.
            Default is -4, which corresponds to very high redshift.
        steps (int, optional): Number of integration steps for computing the radial comoving distance.
            Default is 512.

    Returns:
        Luminosity distance in units of Mpc/h (or physical distance divided by the reduced Hubble constant). Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The luminosity distance is computed by integrating the radial comoving distance and then applying the
        formula for luminosity distance. It is a measure of the distance to an object based on its luminosity.

        The formula for luminosity distance is typically given by:

        $$
        D_L(a) = (1 + z) \cdot \chi(a)
        $$

        where $z = \frac{1}{a} - 1$ is the redshift, and $\chi(a)$ is the radial comoving distance.
    """
    comoving_radial = radial_comoving_distance(cosmo, a, log10_amin, steps) / cosmo.h
    return comoving_radial / a


def distance_modulus(
    cosmo: Cosmology,
    a: float,
    log10_amin: float = -4,
    steps: int = 512,
) -> float:
    r"""
    Computes the distance modulus, which quantifies the difference between the apparent and absolute magnitudes
    of an astronomical object.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters (e.g., Hubble constant, matter density).
        a (float): Scale factor of the Universe (inverse of 1 + redshift).
        log10_amin (float, optional): Logarithm (base 10) of the minimum scale factor to consider.
            Defaults to -4, corresponding to very high redshift.
        steps (int, optional): Number of integration steps for computing the radial comoving distance.
            Defaults to 512.

    Returns:
        Distance modulus in magnitudes, which quantifies the difference between the apparent and absolute magnitudes of an object.

    Notes:
        The distance modulus is calculated using the luminosity distance as:

        $$
        \mu = 5 \log_{10}(d_L \cdot 10) + 20
        $$

        where $d_L$ is the luminosity distance in megaparsecs.
    """
    lum_dist = luminosity_distance(cosmo, a, log10_amin, steps)
    return 5.0 * np.log10(lum_dist * 10) + 20.0


def a_of_chi(
    cosmo: Cosmology, chi: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the scale factor corresponding to a given radial comoving distance $\chi$
    using reverse linear interpolation.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., Hubble constant, matter density, Dark Energy density).
        chi (Union[float, np.ndarray]): Radial comoving distance or an array of radial comoving
            distances to query. Can be a scalar or an array.

    Returns:
        Scale factor(s) corresponding to the given radial comoving distance(s). Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        This function performs reverse linear interpolation to compute the scale factor $a$
        corresponding to the radial comoving distance(s) $\chi$. The relationship between the
        scale factor and comoving distance is based on the cosmological model, and the interpolation
        method allows for efficient calculation for a range of distances.
    """
    # Check if distances have already been computed, force computation otherwise
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        radial_comoving_distance(cosmo, 1.0)
    cache = cosmo._workspace["background.radial_comoving_distance"]
    chi = np.atleast_1d(chi)
    return interp(chi, cache["chi"], cache["a"])


def dchioverda(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the derivative of the radial comoving distance with respect to the scale factor.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., Hubble constant, matter density, Dark Energy density).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors to compute the derivative.
            Can be a scalar or an array.

    Returns:
        Derivative of the radial comoving distance with respect to the scale factor at the specified scale factor(s). Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The expression for the derivative of the radial comoving distance with respect to the scale factor is:

        $$
        \frac{d \chi}{da}(a) = \frac{R_H}{a^2 E(a)}
        $$

        where $R_H$ is the Hubble radius, $a$ is the scale factor, and $E(a)$ is the function derived from Friedmann's Equation.
    """
    return const.rh / (a**2 * np.sqrt(Esqr(cosmo, a)))


def transverse_comoving_distance(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the transverse comoving distance for a given cosmological model and scale factor.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., Hubble constant, matter density, curvature).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors to compute the transverse comoving distance.
            Can be a scalar or an array.

    Returns:
        Transverse comoving distance corresponding to the specified scale factor(s). Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The transverse comoving distance depends on the curvature of the universe and is related to
        the radial comoving distance ($\chi(a)$) through the following piecewise formula:

        $$
        f_k(a) = \left\lbrace
        \begin{matrix}
        R_H \frac{1}{\sqrt{\Omega_k}} \sinh(\sqrt{|\Omega_k|} \chi(a) R_H) & \text{for } \Omega_k > 0 \\
        \chi(a) & \text{for } \Omega_k = 0 \\
        R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|} \chi(a) R_H) & \text{for } \Omega_k < 0
        \end{matrix}
        \right.
        $$

        where:
        - $R_H$ is the Hubble radius.
        - $\Omega_k$ is the curvature parameter.
        - $\chi(a)$ is the radial comoving distance at the given scale factor $a$.
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


def angular_diameter_distance(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the angular diameter distance for a given cosmological model and scale factor.

    Args:
        cosmo (Cosmology): Cosmological model object containing relevant parameters
            (e.g., Hubble constant, matter density, curvature).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors to compute the angular diameter distance.
            Can be a scalar or an array.

    Returns:
        Angular diameter distance corresponding to the specified scale factor(s). Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The angular diameter distance is expressed in terms of the transverse comoving distance as:

        $$
        d_A(a) = a f_k(a)
        $$

        where:

        - $a$ is the scale factor.

        - $f_k(a)$ is the transverse comoving distance at the given scale factor $a$,
          which depends on the curvature of the universe.
    """
    return a * transverse_comoving_distance(cosmo, a)


def growth_factor(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the linear growth factor $D(a)$ at a given scale factor,
    normalized such that $D(a=1) = 1$.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., matter density, Hubble constant, growth rate parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            to compute the growth factor at. Can be a scalar or an array.

    Returns:
        Growth factor computed at the requested scale factor(s). Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The computation of the growth factor depends on the cosmological model and its parameters.
        If the $\gamma$ parameter is defined in the cosmology model, the growth factor is computed
        assuming the $f = \Omega^\gamma$ growth rate. Otherwise, the usual ordinary differential equation
        (ODE) for growth will be solved to compute the growth factor.
    """

    if cosmo._flags["gamma_growth"]:
        return _growth_factor_gamma(cosmo, a)
    else:
        return _growth_factor_ODE(cosmo, a)


def growth_rate(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the growth rate $dD/d\ln a$ at a given scale factor.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., matter density, Hubble constant, growth rate parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            to compute the growth rate at. Can be a scalar or an array.

    Returns:
        Growth rate computed at the requested scale factor(s). Returns a scalar if the input is a scalar, or an array if the input is an array.

    Notes:
        The computation of the growth rate depends on the cosmological model and its parameters.
        If the $\gamma$ parameter is defined in the cosmology model, the growth rate is computed
        assuming the $f = \Omega^\gamma$ growth rate model. Otherwise, the usual ordinary differential
        equation (ODE) for growth will be solved to compute the growth rate.

        The LCDM approximation to the growth rate $f_{\gamma}(a)$ is given by:

        $$
        f_{\gamma}(a) = \Omega_m^{\gamma}(a)
        $$

        where $\gamma$ in LCDM is approximately given by $\gamma \approx 0.55$.

        For more details, see Equation 32 in [2019:Euclid Preparation VII (2019)](https://arxiv.org/abs/1910.09273).
    """
    if cosmo._flags["gamma_growth"]:
        return _growth_rate_gamma(cosmo, a)
    else:
        return _growth_rate_ODE(cosmo, a)


def grad_H(cosmo: Cosmology, a: Union[float, np.ndarray]) -> np.ndarray:
    """Calculates the first derivative of the Hubble parameter at scale factor a.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., matter density, Hubble constant, growth rate parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            to compute the growth rate at. Can be a scalar or an array.

    Returns:
        Derivative of the Hubble parameter with respect to a
    """
    return grad(H, argnums=1)(cosmo, a)


def alpha_beta(cosmo: Cosmology, a: Union[float, np.ndarray]) -> np.ndarray:
    """Calculates the matrix which maps the first derivative and the answer we want, that is,
    $$
    y' = A y
    $$

    $y'$ is a vector which contains $g'(a)$ and $g"(a)$ while $y$ is a vector which contains $g(a)$ and $g'(a)$.

    $A$ is the matrix we calculate here and we call $-A[1,0]$ as beta and $-A[1,1]$ as alpha.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., matter density, Hubble constant, growth rate parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            to compute the growth rate at. Can be a scalar or an array.

    Returns:
        The matrix $A$ according to the equation described above.
    """
    gH_over_H = grad_H(cosmo, a) / H(cosmo, a)
    alpha = 5.0 / a + gH_over_H
    beta = 3 / a**2 + gH_over_H / a - 1.5 * Omega_m_a(cosmo, a) / a**2
    return np.array([[0.0, 1.0], [-beta, -alpha]])


def _growth_factor_ODE(
    cosmo: Cosmology,
    a: Union[float, np.ndarray],
    log10_amin: float = -3,
    steps: int = 128,
    eps: float = 1e-4,
) -> Union[float, np.ndarray]:
    r"""
    Computes the linear growth factor $D(a)$ at a given scale factor,
    normalized such that $D(a=1) = 1$ using the ODE method.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., Hubble constant, matter density, and other cosmological parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            for which the growth factor is computed. Can be a scalar or an array.
        log10_amin (float, optional): Minimum scale factor in log10, default is -3 (corresponding to scale factor $a = 10^{-3}$).
        steps (int, optional): Number of integration steps to be used for solving the ODE, default is 128.
        eps (float, optional): Tolerance for the ODE solver, default is 1e-4.

    Returns:
        Growth factor $D(a)$ at the requested scale factor(s). Returns a scalar if the input scale factor is a scalar, or an array if the input is an array.

    Notes:
        The linear growth factor $D(a)$ is computed by solving the ordinary differential equation (ODE)
        for the growth factor, with the boundary condition that $D(a=1) = 1$. The method uses an
        integration technique with the specified tolerance (`eps`) and the number of steps (`steps`).

        The scale factor range is determined by the parameter `log10_amin`, where the minimum scale
        factor is given by $a_{min} = 10^{\text{log10\_amin}}$.

        This function is used to compute the growth factor at various scale factors in a cosmological model.

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


def _growth_rate_ODE(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the growth rate $dD/d\ln(a)$ at a given scale factor by solving the
    linear growth ODE.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., Hubble constant, matter density, and other cosmological parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            for which the growth rate is computed. Can be a scalar or an array.

    Returns:
        Growth rate $dD/d\ln(a)$ at the requested scale factor(s). Returns a scalar if the input scale factor is a scalar, or an array if the input is an array.

    Notes:
        The growth rate is computed by solving the ordinary differential equation (ODE)
        for the linear growth factor $D(a)$. The equation for the growth rate is given by:

        $$
        f(a) = \frac{dD}{d\ln(a)}
        $$

        The method assumes the cosmology parameters are provided through the `cosmo` object,
        which includes quantities like the Hubble constant, matter density, etc. The growth rate
        will be computed according to the specific cosmological model used.

    """
    # Check if growth has already been computed, if not, compute it
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return interp(a, cache["a"], cache["f"])


def _growth_factor_gamma(
    cosmo: Cosmology,
    a: Union[float, np.ndarray],
    log10_amin: float = -3,
    steps: int = 128,
) -> Union[float, np.ndarray]:
    r"""
    Computes the growth factor by integrating the growth rate provided by the
    $\gamma$ parametrization, normalized such that $D(a=1) = 1$.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., Hubble constant, matter density, and other cosmological parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            for which the growth factor is computed. Can be a scalar or an array.
        log10_amin (float, optional): Logarithm (base 10) of the minimum scale factor to
            consider for integration. Default is -3, corresponding to a very high redshift.
        steps (int, optional): Number of integration steps to use for computing the growth
            factor. Default is 128.

    Returns:
        Growth factor computed at the requested scale factor(s). Returns a scalar if the input scale factor is a scalar, or an array if the input is an array.

    Notes:
        The growth factor is computed by integrating the growth rate function provided by the
        $\gamma$ parametrization. The growth rate $f_{\gamma}(a)$ is given by:

        $$
        f_{\gamma}(a) = \Omega_m^\gamma (a)
        $$

        where $\gamma$ is a parameter of the cosmological model. The growth factor is
        normalized such that $D(a=1) = 1$.
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


def _growth_rate_gamma(
    cosmo: Cosmology, a: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Computes the growth rate approximation at the given scale factor, $a$.

    Args:
        cosmo (Cosmology): Cosmology object containing relevant parameters
            (e.g., Hubble constant, matter density, and other cosmological parameters).
        a (Union[float, np.ndarray]): Scale factor or an array of scale factors
            at which the growth rate is computed. Can be a scalar or an array.

    Returns:
        Growth rate approximation computed at the requested scale factor(s). Returns a scalar if the input scale factor is a scalar, or an array if the input is an array.

    Notes:
        The LCDM approximation to the growth rate $f_{\gamma}(a)$ is given by:

        $$
        f_{\gamma}(a) = \Omega_m^{\gamma} (a)
        $$

        where $\gamma$ is a parameter of the cosmological model. In the LCDM cosmology,
        $\gamma$ is approximately $0.55$.

        See Equation 32 in [2019:Euclid Preparation VII (2019)](https://arxiv.org/abs/1910.09273).
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
        The scalefactors and the comoving radial distance
    """
    a_min = z2a(z_min)
    a_max = z2a(z_max)
    chi_min = radial_comoving_distance(cosmo, a=a_min).item()
    chi_max = radial_comoving_distance(cosmo, a=a_max).item()
    chi_arr = np.linspace(chi_min, chi_max, n_z)
    a_arr = a_of_chi(cosmo, chi_arr)
    return a_arr, chi_arr
