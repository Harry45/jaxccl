# This module computes power spectra
import jax
import jax.numpy as jnp
import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const
import jax_cosmo.transfer as tklib
from jax_cosmo.core import Cosmology
from jax_cosmo.scipy.integrate import romb
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.scipy.interpolate import interp
from jax_cosmo.utils import a2z
from jax_cosmo.emulator import EMUdata, prediction_pklin_jax, prediction_gf_jax
from jax.experimental import checkify

jax.config.update("jax_enable_x64", True)

__all__ = ["primordial_matter_power",
           "linear_matter_power",
           "linear_matter_power_emu",
           "nonlinear_matter_power",
           ]

# option to use the emulator or not
EMUDATA = EMUdata()
USE_EMU = True

def linear_matter_power_emu(cosmo, k: jnp.ndarray, a=1.0) -> jnp.ndarray:
    """
    Calculates the linear matter power spectrum using the emulator. The testpoint is a 5D vector
    and the order of the parameters is as follows:

    testpoint[0] -> sigma8
    testpoint[1] -> Omega_cdm
    testpoint[2] -> Omega_b
    testpoint[3] -> h
    testpoint[4] -> n_s

    Args:
        testpoint (jnp.ndarray): the cosmological parameter
        wavenumber (jnp.ndarray): the new wavenumbers
        scalefactor (float): the scale factor

    Returns:
        jnp.ndarray: the linear matter power spectrum for any scale factor/redshift
    """
    k = jnp.atleast_1d(k)
    a = jnp.atleast_1d(a)

    redshift = a2z(a)
    testpoint = jnp.array(
        [cosmo.sigma8, cosmo.Omega_c, cosmo.Omega_b, cosmo.h, cosmo.n_s]
    )
    # this is the linear matter power spectrum at redshift 0
    pklin_jax_0 = prediction_pklin_jax(testpoint, EMUDATA.quant_pk)

    # this is the growth factor between 0 and 3 (20 values)
    gf_jax = prediction_gf_jax(testpoint, EMUDATA.quant_gf)

    # interpolate the power spectrum for the new wavenumbers
    pklin_jax_0 = jnp.interp(jnp.log(k), jnp.log(EMUDATA.kgrid), jnp.log(pklin_jax_0))
    pklin_jax_0 = jnp.exp(pklin_jax_0)

    # this is the linear matter power spectrum for the queried scale factor.
    pklin_jax_z = pklin_jax_0 * jnp.interp(redshift, EMUDATA.zgrid, gf_jax)

    return pklin_jax_z


def primordial_matter_power(cosmo, k):
    """Primordial power spectrum
    Pk = k^n
    """
    return k**cosmo.n_s


def linear_matter_power(cosmo, k, a=1.0, transfer_fn=tklib.Eisenstein_Hu, **kwargs):
    r"""Computes the linear matter power spectrum.

    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1}

    a: array_like, optional
        Scale factor (def: 1.0)

    transfer_fn: transfer_fn(cosmo, k, **kwargs)
        Transfer function

    Returns
    -------
    pk: array_like
        Linear matter power spectrum at the specified scale
        and scale factor.

    """
    k = jnp.atleast_1d(k)
    a = jnp.atleast_1d(a)
    g = bkgrd.growth_factor(cosmo, a)
    t = transfer_fn(cosmo, k, **kwargs)
    pknorm = cosmo.sigma8**2 / sigmasqr(cosmo, 8.0, transfer_fn, **kwargs)
    pk = primordial_matter_power(cosmo, k) * t**2 * g**2

    # Apply normalisation
    pk = pk * pknorm

    return pk.squeeze()


def sigmasqr(cosmo, R, transfer_fn, kmin=0.0001, kmax=1000.0, ksteps=5, **kwargs):
    """Computes the energy of the fluctuations within a sphere of R h^{-1} Mpc

    .. math::

       \\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)

    where

    .. math::

       W(kR) = \\frac{3j_1(kR)}{kR}
    """

    def int_sigma(logk):
        k = jnp.exp(logk)
        x = k * R
        w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / (x * x * x)
        pk = transfer_fn(cosmo, k, **kwargs) ** 2 * primordial_matter_power(cosmo, k)
        return k * (k * w) ** 2 * pk

    y = romb(int_sigma, jnp.log10(kmin), jnp.log10(kmax), divmax=7)
    return 1.0 / (2.0 * jnp.pi**2.0) * y


def linear(cosmo, k, a, transfer_fn):
    """Linear matter power spectrum"""
    return linear_matter_power(cosmo, k, a, transfer_fn)


def _halofit_parameters(cosmo, a, transfer_fn):
    r"""Computes the non linear scale,
    effective spectral index,
    spectral curvature
    """
    # Step 1: Finding the non linear scale for which sigma(R)=1
    # That's our search range for the non linear scale
    logr = jnp.linspace(jnp.log(1e-4), jnp.log(1e1), 256)

    # TODO: implement a better root finding algorithm to compute the non linear scale
    @jax.vmap
    def R_nl(a):
        def int_sigma(logk):
            k = jnp.exp(logk)
            r = jnp.exp(logr)
            y = jnp.outer(k, r)
            if USE_EMU:
                pk = linear_matter_power_emu(cosmo, k)
            else:
                pk = linear_matter_power(cosmo, k, transfer_fn=transfer_fn)
            g = bkgrd.growth_factor(cosmo, jnp.atleast_1d(a))
            return (
                jnp.expand_dims(pk * k**3, axis=1)
                * jnp.exp(-(y**2))
                / (2.0 * jnp.pi**2)
                * g**2
            )

        sigma = simps(int_sigma, jnp.log(1e-4), jnp.log(1e4), 256)
        root = interp(jnp.atleast_1d(1.0), sigma, logr)
        return jnp.exp(root).clip(
            1e-6
        )  # To ensure that the root is not too close to zero

    # Compute non linear scale
    k_nl = 1.0 / R_nl(jnp.atleast_1d(a)).squeeze()

    # Step 2: Retrieve the spectral index and spectral curvature
    def integrand(logk):
        k = jnp.exp(logk)
        y = jnp.outer(k, 1.0 / k_nl)
        if USE_EMU:
            pk = linear_matter_power_emu(cosmo, k)
        else:
            pk = linear_matter_power(cosmo, k, transfer_fn=transfer_fn)
        g = jnp.expand_dims(bkgrd.growth_factor(cosmo, jnp.atleast_1d(a)), 0)
        res = (
            jnp.expand_dims(pk * k**3, axis=1)
            * jnp.exp(-(y**2))
            * g**2
            / (2.0 * jnp.pi**2)
        )
        dneff_dlogk = 2 * res * y**2
        dC_dlogk = 4 * res * (y**2 - y**4)
        return jnp.stack([dneff_dlogk, dC_dlogk], axis=1)

    res = simps(integrand, jnp.log(1e-4), jnp.log(1e4), 256)

    n_eff = res[0] - 3.0
    C = res[0] ** 2 + res[1]

    return k_nl, n_eff, C


def halofit(cosmo, k, a, transfer_fn, prescription="takahashi2012"):
    r"""Computes the non linear halofit correction to the matter power spectrum.

    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1}

    a: array_like, optional
        Scale factor (def: 1.0)

    prescription: str, optional
        Either 'smith2003' or 'takahashi2012'

    Returns
    -------
    pk: array_like
        Non linear matter power spectrum at the specified scale
        and scale factor.

    Notes
    -----
    The non linear corrections are implemented following :cite:`2003:smith`

    """
    a = jnp.atleast_1d(a)

    # Compute the linear power spectrum
    if USE_EMU:
        print("Using the emulator")
        pklin = linear_matter_power_emu(cosmo, k, a)

    else:
        print("Using EH method")
        pklin = linear_matter_power(cosmo, k, a, transfer_fn)

    # Compute non linear scale, effective spectral index and curvature
    k_nl, n, C = _halofit_parameters(cosmo, a, transfer_fn)

    om_m = bkgrd.Omega_m_a(cosmo, a)
    om_de = bkgrd.Omega_de_a(cosmo, a)
    w = bkgrd.w(cosmo, a)
    frac = om_de / (1.0 - om_m)

    if prescription == "smith2003":
        # eq C9 to C18
        a_n = 10 ** (
            1.4861
            + 1.8369 * n
            + 1.6762 * n**2
            + 0.7940 * n**3
            + 0.1670 * n**4
            - 0.6206 * C
        )
        b_n = 10 ** (0.9463 + 0.9466 * n + 0.3084 * n**2 - 0.9400 * C)
        c_n = 10 ** (-0.2807 + 0.6669 * n + 0.3214 * n**2 - 0.0793 * C)
        gamma_n = 0.8649 + 0.2989 * n + 0.1631 * C
        alpha_n = 1.3884 + 0.3700 * n - 0.1452 * n**2
        beta_n = 0.8291 + 0.9854 * n + 0.3401 * n**2
        mu_n = 10 ** (-3.5442 + 0.1908 * n)
        nu_n = 10 ** (0.9585 + 1.2857 * n)
    elif prescription == "takahashi2012":
        a_n = 10 ** (
            1.5222
            + 2.8553 * n
            + 2.3706 * n**2
            + 0.9903 * n**3
            + 0.2250 * n**4
            - 0.6038 * C
            + 0.1749 * om_de * (1 + w)
        )
        b_n = 10 ** (
            -0.5642 + 0.5864 * n + 0.5716 * n**2 - 1.5474 * C + 0.2279 * om_de * (1 + w)
        )
        c_n = 10 ** (0.3698 + 2.0404 * n + 0.8161 * n**2 + 0.5869 * C)
        gamma_n = 0.1971 - 0.0843 * n + 0.8460 * C
        alpha_n = jnp.abs(6.0835 + 1.3373 * n - 0.1959 * n**2 - 5.5274 * C)
        beta_n = (
            2.0379
            - 0.7354 * n
            + 0.3157 * n**2
            + 1.2490 * n**3
            + 0.3980 * n**4
            - 0.1682 * C
        )
        mu_n = 0.0
        nu_n = 10 ** (5.2105 + 3.6902 * n)
    else:
        raise NotImplementedError

    f1a = om_m ** (-0.0732)
    f2a = om_m ** (-0.1423)
    f3a = om_m**0.0725
    f1b = om_m ** (-0.0307)
    f2b = om_m ** (-0.0585)
    f3b = om_m ** (0.0743)

    if prescription == "takahashi2012":
        f1 = f1b
        f2 = f2b
        f3 = f3b
    elif prescription == "smith2003":
        f1 = frac * f1b + (1 - frac) * f1a
        f2 = frac * f2b + (1 - frac) * f2a
        f3 = frac * f3b + (1 - frac) * f3a
    else:
        raise NotImplementedError

    f = lambda x: x / 4.0 + x**2 / 8.0

    d2l = k**3 * pklin / (2.0 * jnp.pi**2)

    y = k / k_nl

    # Eq C2
    d2q = d2l * ((1.0 + d2l) ** beta_n / (1 + alpha_n * d2l)) * jnp.exp(-f(y))
    d2hprime = (
        a_n * y ** (3 * f1) / (1.0 + b_n * y**f2 + (c_n * f3 * y) ** (3.0 - gamma_n))
    )
    d2h = d2hprime / (1.0 + mu_n / y + nu_n / y**2)
    # Eq. C1
    d2nl = d2q + d2h
    pk_nl = 2.0 * jnp.pi**2 / k**3 * d2nl

    return pk_nl.squeeze()


def nonlinear_matter_power(
    cosmo, k, a=1.0, transfer_fn=tklib.Eisenstein_Hu, nonlinear_fn=halofit
):
    """Computes the non-linear matter power spectrum.

    This function is just a wrapper over several nonlinear power spectra.
    """
    return nonlinear_fn(cosmo, k, a, transfer_fn=transfer_fn)


def dlogP_dlogk(cosmo, k: jnp.ndarray, a: float, transfer_fn=tklib.Eisenstein_Hu, **kwargs) -> jnp.ndarray:
    """
    Computes the logarithmic derivative of the linear matter power spectrum with respect to k.

    Args:
        cosmo: Cosmology object.
        k (jnp.ndarray): Wavenumber(s) in h/Mpc.
        a (float): Scale factor.
        transfer_fn (callable): Transfer function.
        **kwargs: Extra arguments to pass to the transfer function.

    Returns:
        jnp.ndarray: The derivative d ln P / d ln k.
    """
    def log_P(log_k: float) -> float:
        """Helper function to compute log(P) at log_k."""
        return jnp.log(linear_matter_power(cosmo, jnp.exp(log_k), a, transfer_fn, **kwargs))

    # Vectorize gradient computation over k
    dlogP_dlogk_fn = jax.vmap(jax.grad(log_P))

    return dlogP_dlogk_fn(jnp.log(k))