import jax.numpy as jnp
from jax_cosmo.core import Cosmology
from jax_cosmo.halos.hmbase import MassDefinition, mass_translator, sigmaM, get_delta_c
from interpax import Interpolator2D
from jax.scipy.special import gammainc, gamma


def norm_en(
    mass: jnp.ndarray, radius: jnp.ndarray, conc: jnp.ndarray, alpha: jnp.ndarray
) -> jnp.ndarray:
    """Computes normalization constant for the Einasto profile.

    Args:
        mass: Halo mass array.
        radius: Scale radius (R_s).
        concentration: Concentration parameter.
        alpha: Einasto profile shape parameter.

    Returns:
        Normalization factor of the Einasto profile.
    """
    prefactor = mass / (jnp.pi * radius**3)
    shape_term = 2 ** (2 - 3 / alpha) * alpha ** (-1 + 3 / alpha)
    exp_term = jnp.exp(2 / alpha)
    gamma_term = gamma(3 / alpha) * gammainc(3 / alpha, 2 / alpha * conc**alpha)

    return prefactor / (shape_term * exp_term * gamma_term)


def projected_quad_integrand(
    z: jnp.ndarray, R: jnp.ndarray, R_s: jnp.ndarray, alpha: jnp.ndarray
) -> jnp.ndarray:
    """Integrand for the projected Einasto profile.

    Args:
        z (jnp.ndarray): Integration variable.
        R (jnp.ndarray): Projected radius [shape: (1, N_r)].
        R_s (jnp.ndarray): Scale radius [shape: (N_m, 1)].
        alpha (jnp.ndarray): Einasto alpha [shape: (N_m, 1)].

    Returns:
        jnp.ndarray: Integrand values [shape: (N_m, N_r, N_z)].
    """
    x = jnp.sqrt(z**2 + R**2) / R_s
    return jnp.exp(-2 * (x**alpha - 1) / alpha)


class JAXHaloProfileEinasto:
    def __init__(
        self,
        mass_def,
        concentration,
        truncated=False,
        projected_quad=False,
        alpha="cosmo",
    ):
        """
        Initializes the Einasto profile.

        Args:
            mass_def: Mass definition used.
            concentration: Function to compute concentration.
            truncated: Whether to truncate the profile at R_vir.
            projected_quad: Whether to use projected profile with integration.
            alpha: Either 'cosmo' to compute from ν or a fixed float value.
        """
        self.mass_def = mass_def
        self.con = concentration
        self.truncated = truncated
        self.projected_quad = projected_quad
        self.alpha = alpha

        if projected_quad and truncated:
            raise ValueError("Cannot use both projected_quad and truncated profile.")
        if projected_quad:
            self._projected = self._projected_quad

        self.mass_vir = MassDefinition("vir", "matter")

    def _get_alpha(
        self,
        cosmo: Cosmology,
        mass: jnp.ndarray,
        scale_factor: jnp.ndarray,
        interpolator: Interpolator2D,
    ):
        """Returns alpha from either a fixed value or the ν-dependent fit.

        Args:
            cosmo: Cosmology object.
            mass: Halo mass.
            scale_factor: Scale factor (a = 1 / (1 + z)).
            interpolator: Interpolator for sigma_M

        Returns:
            Alpha value.
        """
        if self.alpha == "cosmo":
            mass_vir = mass_translator(
                self.mass_def,
                self.mass_vir,
                cosmo,
                scale_factor,
                mass,
                self.con,
                interpolator,
            )
            sigma_mass = sigmaM(jnp.log10(mass_vir), scale_factor, interpolator)
            nu = get_delta_c(cosmo, scale_factor, method="EdS_approx") / sigma_mass
            return 0.155 + 0.0095 * nu * nu
        return jnp.full_like(mass, self.alpha)

    def _real(
        self,
        cosmo: Cosmology,
        radius: jnp.ndarray,
        mass: jnp.ndarray,
        scale_factor: jnp.ndarray,
        interpolator: Interpolator2D,
    ) -> jnp.ndarray:
        """Computes the 3D density profile.

        Args:
            cosmo: Cosmology object.
            radius: Radius or radii at which to compute the profile.
            mass: Halo mass.
            scale_factor: Scale factor (a = 1 / (1 + z)).
            interpolator: Interpolator for sigma_M

        Returns:
            The real-space density profile.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(
            cosmo, mass_use, scale_factor, interpolator
        )
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, mass_use, scale_factor, interpolator)
        norm = norm_en(mass_use, R_s, c_M, alpha)

        x = radius_use[None, :] / R_s[:, None]
        prof = norm[:, None] * jnp.exp(
            -2.0 * (x ** alpha[:, None] - 1) / alpha[:, None]
        )

        if self.truncated:
            prof[radius_use[None, :] > R_M[:, None]] = 0

        if jnp.ndim(radius) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof

    def _projected_quad(
        self,
        cosmo: Cosmology,
        radius: jnp.ndarray,
        mass: jnp.ndarray,
        scale_factor: jnp.ndarray,
        interpolator: Interpolator2D,
        delta_z: float = 1e-2,
        maxrad_factor: float = 20.0,
    ) -> jnp.ndarray:
        """Computes the 2D projected density profile via quadrature integration.

        Args:
            cosmo: Cosmology object.
            radius: Projected radius or radii.
            mass: Halo mass.
            scale_factor: Scale factor.
            interpolator: Interpolator for sigma_M
            delta_z: Spacing for the line-of-sight integration grid.
            maxrad_factor: Max z = maxrad_factor * max(radius).

        Returns:
            Projected density profile.
        """
        radius_use = jnp.atleast_1d(radius)
        mass_use = jnp.atleast_1d(mass)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, mass_use, scale_factor) / scale_factor
        c_M = self.con.compute_concentration(
            cosmo, mass_use, scale_factor, interpolator
        )
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, mass_use, scale_factor, interpolator)
        z_grid = jnp.arange(
            0.0, maxrad_factor * max(radius_use), delta_z
        )  # Shape: (N_z,)

        z_grid = z_grid[None, None, :]  # (1, 1, N_z)
        R_grid = radius_use[None, :, None]  # (1, N_r, 1)
        R_s_grid = R_s[:, None, None]  # (N_m, 1, 1)
        alpha_grid = alpha[:, None, None]  # (N_m, 1, 1)

        # Compute integrand
        integrand = projected_quad_integrand(
            z_grid, R_grid, R_s_grid, alpha_grid
        )  # (N_m, N_r, N_z)

        # Integrate over z using trapezoid
        prof = jnp.trapezoid(integrand, dx=delta_z, axis=-1)  # (N_m, N_r)
        prof *= 2 * norm_en(mass_use, R_s, c_M, alpha)[:, None]

        if jnp.ndim(radius) == 0:
            prof = jnp.squeeze(prof, axis=-1)
        if jnp.ndim(mass) == 0:
            prof = jnp.squeeze(prof, axis=0)
        return prof
