from typing import Union, Tuple
import jax
import jax.numpy as jnp
from jax_cosmo.core import Cosmology
from jax_cosmo.scipy.special import compute_sici
from interpax import Interpolator2D


def norm_nfw(mass: Union[float, jnp.ndarray],
             radius: Union[float, jnp.ndarray],
             conc: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Computes the NFW normalization given mass, radius, and concentration.

    Args:
        mass: The mass of the halo.
        radius: The scale radius of the halo.
        conc: The concentration parameter.

    Returns:
        The NFW normalization factor.
    """
    return mass / (4 * jnp.pi * radius**3 * (jnp.log(1 + conc) - conc / (1 + conc)))

def fx_projected_nfw(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the projected NFW profile function.

    Args:
        x: The scaled radius (radius / scale radius).

    Returns:
        The projected NFW profile.
    """
    def f1(xx: jnp.ndarray) -> jnp.ndarray:
        x2m1 = xx * xx - 1
        sqx2m1 = jnp.sqrt(-x2m1)
        return 1 / x2m1 + jnp.arcsinh(sqx2m1 / xx) / (-x2m1)**1.5

    def f2(xx: jnp.ndarray) -> jnp.ndarray:
        x2m1 = xx * xx - 1
        sqx2m1 = jnp.sqrt(x2m1)
        return 1 / x2m1 - jnp.arcsin(sqx2m1 / xx) / x2m1**1.5

    xf = x.flatten()
    return jnp.piecewise(xf,
                        [xf < 1, xf > 1],
                        [f1, f2, 1./3.]).reshape(x.shape)

def fx_cumul2d_nfw(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the 2D cumulative NFW profile function.

    Args:
        x: The scaled radius (radius / scale radius).

    Returns:
        The 2D cumulative NFW profile.
    """
    def f1(xx: jnp.ndarray) -> jnp.ndarray:
        sqx2m1 = jnp.sqrt(-(xx * xx - 1))
        return jnp.log(0.5 * xx) + jnp.arcsinh(sqx2m1 / xx) / sqx2m1

    def f2(xx: jnp.ndarray) -> jnp.ndarray:
        sqx2m1 = jnp.sqrt(xx * xx - 1)
        return jnp.log(0.5 * xx) + jnp.arcsin(sqx2m1 / xx) / sqx2m1

    xf = x.flatten()
    omln2 = 1 - jnp.log(2)

    f = jnp.piecewise(xf,
                        [xf < 1, xf > 1],
                        [f1, f2, omln2]).reshape(x.shape)
    return 2 * f / x**2

class JAXHaloProfileNFW:
    """
    Class to compute the NFW halo profile, including Fourier, projected, and 2D cumulative profiles.

    Args:
        mass_def: The mass definition object.
        concentration: The concentration object.
        fourier_analytic: Whether to compute the Fourier transform analytically.
        projected_analytic: Whether to compute the projected profile analytically.
        cumul2d_analytic: Whether to compute the cumulative 2D profile analytically.
        truncated: Whether to apply a truncation at the virial radius.

    """

    def __init__(self, mass_def: Cosmology,
                 concentration,
                 fourier_analytic: bool = True,
                 projected_analytic: bool = False,
                 cumul2d_analytic: bool = False,
                 truncated: bool = True):

        self.mass_def = mass_def
        self.con = concentration
        self.fourier_analytic = fourier_analytic
        self.projected_analytic = projected_analytic
        self.cumul2d_analytic = cumul2d_analytic
        self.truncated = truncated

        if fourier_analytic:
            self._fourier = self._fourier_analytic

        if projected_analytic:
            if truncated:
                raise ValueError("Analytic projected profile not supported for truncated NFW. "
                "Set `truncated` or `projected_analytic` to `False`.")
            self._projected = self._projected_analytic

        if cumul2d_analytic:
            if truncated:
                raise ValueError("Analytic cumulative 2D profile not supported for truncated NFW. "
                "Set `truncated` or `cumul2d_analytic` to `False`.")
            self._cumul2d = self._cumul2d_analytic

    def _real(self, cosmo: Cosmology,
              radius: Union[float, jnp.ndarray],
              mass: Union[float, jnp.ndarray],
              scale_factor: float,
              interpolator: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Computes the real-space NFW profile.

        Args:
            cosmo: The cosmology object used for computing the radius and concentration.
            radius: The radius at which the profile is evaluated.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The NFW profile at the given radius.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        x = radius_use[None, :] / R_s[:, None]
        prof = 1./(x * (1 + x)**2)

        if self.truncated:
            prof[radius_use[None, :] > R_M[:, None]] = 0

        norm = norm_nfw(mass_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if jnp.ndim(radius) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof

    def _projected_analytic(self, cosmo: Cosmology,
                            radius: Union[float, jnp.ndarray],
                            mass: Union[float, jnp.ndarray],
                            scale_factor: float,
                            interpolator: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Computes the analytic projected NFW profile.

        Args:
            cosmo: The cosmology object.
            radius: The radius at which the profile is evaluated.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The projected NFW profile.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        x = radius_use[None, :] / R_s[:, None]
        prof = fx_projected_nfw(x)
        norm = 2 * R_s * norm_nfw(mass_use, R_s, c_M)
        prof *= norm[:, None]

        if jnp.ndim(radius) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof

    def _cumul2d_analytic(self, cosmo: Cosmology,
                          radius: Union[float, jnp.ndarray],
                          mass: Union[float, jnp.ndarray],
                          scale_factor: float,
                          interpolator: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Computes the analytic 2D cumulative NFW profile.

        Args:
            cosmo: The cosmology object.
            radius: The radius at which the profile is evaluated.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The 2D cumulative NFW profile.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        x = radius_use[None, :] / R_s[:, None]
        prof = fx_cumul2d_nfw(x)
        norm = 2 * R_s * norm_nfw(mass_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if jnp.ndim(radius) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof

    def _fourier_analytic(self, cosmo: Cosmology,
                         k: Union[float, jnp.ndarray],
                         mass: Union[float, jnp.ndarray],
                         scale_factor: float,
                         interpolator: Interpolator2D) -> Union[float, jnp.ndarray]:
        """
        Computes the Fourier transform of the NFW profile.

        Args:
            cosmo: The cosmology object.
            k: The Fourier space k values.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The Fourier transform of the NFW profile.
        """
        mass_use = jnp.atleast_1d(mass)
        k_use = jnp.atleast_1d(k)

        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        x = k_use[None, :] * R_s[:, None]
        Si2, Ci2 = compute_sici(x.flatten())
        Si2 = Si2.reshape(mass_use.shape[0], k_use.shape[0])
        Ci2 = Ci2.reshape(mass_use.shape[0], k_use.shape[0])

        P1 = mass_use / (jnp.log(1 + c_M) - c_M / (1 + c_M))

        if self.truncated:
            xnew = (1 + c_M[:, None]) * x
            Si1, Ci1 = compute_sici(xnew.flatten())
            Si1 = Si1.reshape(mass_use.shape[0], k_use.shape[0])
            Ci1 = Ci1.reshape(mass_use.shape[0], k_use.shape[0])
            P2 = jnp.sin(x) * (Si1 - Si2) + jnp.cos(x) * (Ci1 - Ci2)
            P3 = jnp.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
            prof = P1[:, None] * (P2 - P3)
        else:
            P2 = jnp.sin(x) * (0.5 * jnp.pi - Si2) - jnp.cos(x) * Ci2
            prof = P1[:, None] * P2

        if jnp.ndim(k) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof
