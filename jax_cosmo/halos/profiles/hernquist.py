from typing import Union, Tuple
import jax
import jax.numpy as jnp
from jax_cosmo.core import Cosmology
from jax_cosmo.scipy.special import compute_sici
from interpax import Interpolator2D

def norm_hq(mass: float, radius: float, conc: float) -> float:
    """
    Computes the Hernquist normalization from mass, radius, and concentration.

    Args:
        mass (float): The mass of the halo.
        radius (float): The scale radius.
        conc (float): The concentration parameter.

    Returns:
        float: The Hernquist normalization factor.
    """
    return mass / (2 * jnp.pi * radius**3 * (conc / (1 + conc))**2)


def fx_projected_hq(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the projected Hernquist profile function.

    Args:
        x (jnp.ndarray): The input radius values.

    Returns:
        jnp.ndarray: The computed projected Hernquist function values.
    """
    def f1(xx):
        x2m1 = xx**2 - 1
        sqx2m1 = jnp.sqrt(-x2m1)
        return (-3 / (2 * x2m1**2) + (x2m1 + 3) * jnp.arcsinh(sqx2m1 / xx) / (2 * (-x2m1)**2.5))

    def f2(xx):
        x2m1 = xx**2 - 1
        sqx2m1 = jnp.sqrt(x2m1)
        return (-3 / (2 * x2m1**2) + (x2m1 + 3) * jnp.arcsin(sqx2m1 / xx) / (2 * x2m1**2.5))

    xf = x.flatten()
    return jnp.piecewise(xf, [xf < 1, xf > 1], [f1, f2, 2./15.]).reshape(x.shape)


def fx_cumul2d_hq(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cumulative 2D Hernquist profile function.

    Args:
        x (jnp.ndarray): The input radius values.

    Returns:
        jnp.ndarray: The computed cumulative 2D Hernquist function values.
    """
    def f1(xx):
        x2m1 = xx**2 - 1
        sqx2m1 = jnp.sqrt(-x2m1)
        return (1 + 1 / x2m1 + (x2m1 + 1) * jnp.arcsinh(sqx2m1 / xx) / (-x2m1)**1.5)

    def f2(xx):
        x2m1 = xx**2 - 1
        sqx2m1 = jnp.sqrt(x2m1)
        return (1 + 1 / x2m1 - (x2m1 + 1) * jnp.arcsin(sqx2m1 / xx) / x2m1**1.5)

    xf = x.flatten()
    f = jnp.piecewise(xf, [xf < 1, xf > 1], [f1, f2, 1./3.]).reshape(x.shape)
    return f / x**2

class JAXHaloProfileHernquist:
    """ Hernquist (1990) halo density profile.

    This class implements the Hernquist profile, which describes the density distribution of dark matter halos.

    Attributes:
        mass_def (MassDef): The mass definition object.
        con (Concentration): The concentration-mass relation.
        truncated (bool): If True, truncates the profile at :math:r = r_Delta.
        fourier_analytic (bool): If True, uses an analytic Fourier transform.
        projected_analytic (bool): If True, uses an analytic projected profile.
        cumul2d_analytic (bool): If True, uses an analytic 2D cumulative surface density profile.
    """

    def __init__(self, mass_def,
                 concentration,
                 truncated=True,
                 fourier_analytic=False,
                 projected_analytic=False,
                 cumul2d_analytic=False):

        self.mass_def = mass_def
        self.con = concentration
        self.truncated = truncated
        self.fourier_analytic = fourier_analytic
        self.projected_analytic = projected_analytic
        self.cumul2d_analytic = cumul2d_analytic

        if fourier_analytic:
            self._fourier = self._fourier_analytic
        if projected_analytic:
            if truncated:
                raise ValueError("Analytic projected profile not supported for truncated Hernquist. "
                "Set `truncated` or `projected_analytic` to `False`.")
            self._projected = self._projected_analytic

        if cumul2d_analytic:
            if truncated:
                raise ValueError("Analytic cumulative 2D profile not supported for truncated Hernquist. "
                "Set `truncated` or `cumul2d_analytic` to `False`.")
            self._cumul2d = self._cumul2d_analytic


    def _real(self, cosmo: Cosmology,
              radius: Union[float, jnp.ndarray],
              mass: Union[float, jnp.ndarray],
              scale_factor: float,
              interpolator: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Computes the real-space Hernquist profile.

        Args:
            cosmo: The cosmology object used for computing the radius and concentration.
            radius: The radius at which the profile is evaluated.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The Hernquist profile at the given radius.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        norm = norm_hq(mass_use, R_s, c_M)
        x = radius_use[None, :] / R_s[:, None]
        prof = norm[:, None] / (x * (1 + x)**3)

        if self.truncated:
            prof[radius_use[None, :] > R_M[:, None]] = 0

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
        Computes the analytic projected Hernquist profile.

        Args:
            cosmo: The cosmology object.
            radius: The radius at which the profile is evaluated.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The projected Hernquist profile.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        x = radius_use[None, :] / R_s[:, None]
        prof = fx_projected_hq(x)
        norm = 2 * R_s * norm_hq(mass_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

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
        Computes the analytic 2D cumulative Hernquist profile.

        Args:
            cosmo: The cosmology object.
            radius: The radius at which the profile is evaluated.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The 2D cumulative Hernquist profile.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        x = radius_use[None, :] / R_s[:, None]
        prof = fx_cumul2d_hq(x)
        norm = 2 * R_s * norm_hq(mass_use, R_s, c_M)
        prof = prof * norm[:, None]

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
        Computes the Fourier transform of the Hernquist profile.

        Args:
            cosmo: The cosmology object.
            k: The Fourier space k values.
            mass: The mass of the halo.
            scale_factor: The scale factor.
            interpolator: An interpolator function for the concentration.

        Returns:
            The Fourier transform of the Hernquist profile.
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

        P1 = mass / ((c_M / (c_M + 1))**2 / 2)
        c_Mp1 = c_M[:, None] + 1

        if self.truncated:
            xnew = c_Mp1 * x
            Si1, Ci1 = compute_sici(xnew.flatten())
            Si1 = Si1.reshape(mass_use.shape[0], k_use.shape[0])
            Ci1 = Ci1.reshape(mass_use.shape[0], k_use.shape[0])

            P2 = x * jnp.sin(x) * (Ci1 - Ci2) - x * jnp.cos(x) * (Si1 - Si2)
            P3 = (-1 + jnp.sin(c_M[:, None] * x) / (c_Mp1**2 * x)
                  + c_Mp1 * jnp.cos(c_M[:, None] * x) / (c_Mp1**2))
            prof = P1[:, None] * (P2 - P3) / 2

        else:
            P2 = (-x * (2 * jnp.sin(x) * Ci2 + jnp.pi * jnp.cos(x))
                  + 2 * x * jnp.cos(x) * Si2 + 2) / 4
            prof = P1[:, None] * P2

        if jnp.ndim(k) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof