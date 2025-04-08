from typing import Union, Tuple
import jax
import jax.numpy as jnp
from jax_cosmo.core import Cosmology
from jax_cosmo.halos.hmbase import MassDefinition, mass_translator, sigmaM, get_delta_c
from jax_cosmo.scipy.special import compute_sici
from interpax import Interpolator2D
from jax.scipy.special import gammainc, gamma
from quadax import simpson

def norm_en(mass: jnp.ndarray,
            radius: jnp.ndarray,
            conc: jnp.ndarray,
            alpha: jnp.ndarray) -> jnp.ndarray:
    # Einasto normalization from mass, radius, concentration and alpha
    return mass / (jnp.pi * radius**3 * 2**(2-3/alpha) * alpha**(-1+3/alpha)
                * jnp.exp(2/alpha)
                * gamma(3/alpha) * gammainc(3/alpha, 2/alpha*conc**alpha))



class JAXHaloProfileEinasto:
    def __init__(self, mass_def,
                 concentration,
                 truncated=False,
                 projected_quad=False,
                 alpha='cosmo'):

        self.mass_def = mass_def
        self.con = concentration

        self.truncated = truncated
        self.projected_quad = projected_quad
        self.alpha = alpha

        if projected_quad:
            if truncated:
                raise ValueError("projected_quad profile not supported "
                                 "for truncated Einasto. Set `truncated` or "
                                 "`projected_quad` to `False`.")
            self._projected = self._projected_quad

        self.mass_vir = MassDefinition("vir", "matter")

    def _get_alpha(self, cosmo: Cosmology,
                   mass: jnp.ndarray,
                   scale_factor: jnp.ndarray,
                   interpolator: Interpolator2D):
        if self.alpha == 'cosmo':
            mass_vir = mass_translator(self.mass_def,
                                   self.mass_vir,
                                   cosmo,
                                   scale_factor,
                                   mass,
                                   self.con,
                                   interpolator)
            sigma_mass = sigmaM(jnp.log10(mass_vir), scale_factor, interpolator)
            nu = get_delta_c(cosmo, scale_factor, method='EdS_approx') / sigma_mass
            return 0.155 + 0.0095 * nu * nu
        return jnp.full_like(mass, self.alpha)

    def _real(self, cosmo: Cosmology,
              radius: jnp.ndarray,
              mass: jnp.ndarray,
              scale_factor: jnp.ndarray,
              interpolator: Interpolator2D) -> jnp.ndarray:
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, mass_use, scale_factor, interpolator)

        norm = norm_en(mass_use, R_s, c_M, alpha)

        x = radius_use[None, :] / R_s[:, None]
        prof = norm[:, None] * jnp.exp(-2. * (x**alpha[:, None] - 1) /
                                      alpha[:, None])
        if self.truncated:
            prof[radius_use[None, :] > R_M[:, None]] = 0

        if jnp.ndim(radius) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof

    def _projected_quad(self, cosmo: Cosmology,
                        radius: jnp.ndarray,
                        mass: jnp.ndarray,
                        scale_factor: jnp.ndarray,
                        interpolator: Interpolator2D) -> jnp.ndarray:
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(cosmo, mass_use, scale_factor, interpolator)
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, mass_use, scale_factor, interpolator)

        # # Integration grid in z
        # z = jnp.linspace(0.0, 1E20, 2000)  # Adjust upper limit for accuracy
        # dz = z[1] - z[0]

        # # Broadcasted variables for vectorized integration
        # z_grid = z[None, None, :]                     # shape (1, 1, N_z)
        # r_grid = radius_use[None, :, None]            # shape (1, N_r, 1)
        # R_s_grid = R_s[:, None, None]                 # shape (N_m, 1, 1)
        # alpha_grid = alpha[:, None, None]             # shape (N_m, 1, 1)

        # # Compute x = sqrt(z^2 + r^2) / Rs
        # x = jnp.sqrt(z_grid**2 + r_grid**2) / R_s_grid
        # integrand = jnp.exp(-2 * (x**alpha_grid - 1) / alpha_grid)

        # # Trapezoidal integration over z-axis
        # prof = jnp.trapezoid(integrand, dx=dz, axis=-1)  # shape (N_m, N_r)

        prof *= 2 * norm_en(mass_use, R_s, c_M, alpha)[:, None]

        if jnp.ndim(radius) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof
