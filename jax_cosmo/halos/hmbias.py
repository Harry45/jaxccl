from typing import Union, Tuple
import jax.numpy as jnp
from jax_cosmo.core import Cosmology
from interpax import Interpolator1D
from jax_cosmo.halos.hmbase import MassDefinition, get_delta_c


class HaloBiasBhattacharya11:
    """
    Implements halo bias as described in Bhattacharya et al. (2011).
    This parametrization is only valid for 'fof' masses.

    Reference:
        Bhattacharya et al. 2011 (https://arxiv.org/abs/1005.2239)
    """

    name = "Bhattacharya11"

    def __init__(self, overdensity: str = "fof"):
        """
        Initializes the halo bias parameters.

        Args:
            overdensity (str): Overdensity definition (default is 'fof').
        """
        self.overdensity = overdensity
        self.scaling_factor = 0.788
        self.redshift_exponent = 0.01
        self.power_law_index = 0.807
        self.bias_offset = 1.795
        self.critical_density = get_delta_c(None, None, method="EdS")

    def compute_bias(
        self, cosmology: Cosmology, sigma_mass: float, scale_factor: float
    ) -> float:
        """
        Computes the halo bias as a function of mass variance and scale factor.

        Args:
            cosmology (Any): Cosmological parameters (unused in current implementation).
            sigma_mass (float): Variance of mass fluctuations.
            scale_factor (float): Scale factor of the universe.

        Returns:
            float: Computed halo bias.
        """
        peak_height = self.critical_density / sigma_mass
        adjusted_scaling = self.scaling_factor * scale_factor**self.redshift_exponent
        scaled_peak_height_sq = adjusted_scaling * peak_height**2

        power_law_term = 1 + scaled_peak_height_sq**self.power_law_index
        bias_numerator = (
            scaled_peak_height_sq
            - self.bias_offset
            + (2 * self.power_law_index / power_law_term)
        )
        bias = 1 + (bias_numerator / self.critical_density)

        return bias


class HaloBiasSheth01:
    """
    Implements halo bias as described in Sheth et al. (2001).
    This parametrization is only valid for 'fof' masses.

    Reference:
        Sheth et al. 2001 (https://arxiv.org/abs/astro-ph/9907024)
    """

    name = "Sheth01"

    def __init__(self, overdensity: str = "fof"):
        """
        Initializes the halo bias parameters.

        Args:
            overdensity (str): Overdensity definition (default is 'fof').
        """
        self.overdensity = overdensity
        self.scaling_factor = 0.707
        self.sqrt_scaling_factor = self.scaling_factor**0.5
        self.bias_coefficient = 0.5
        self.exponent = 0.6
        self.critical_density = get_delta_c(None, None, method="EdS")

    def compute_bias(
        self, cosmology: Cosmology, sigma_mass: float, scale_factor: float
    ) -> float:
        """
        Computes the halo bias as a function of mass variance and scale factor.

        Args:
            cosmology (Any): Cosmological parameters (unused in current implementation).
            sigma_mass (float): Variance of mass fluctuations.
            scale_factor (float): Scale factor of the universe.

        Returns:
            float: Computed halo bias.
        """
        peak_height = self.critical_density / sigma_mass
        scaled_peak_height_sq = self.scaling_factor * peak_height**2
        exponent_term = scaled_peak_height_sq**self.exponent
        correction_term = (
            self.bias_coefficient * (1.0 - self.exponent) * (1.0 - 0.5 * self.exponent)
        )

        first_term = (
            self.sqrt_scaling_factor
            * scaled_peak_height_sq
            * (1 + self.bias_coefficient / exponent_term)
        )
        second_term = exponent_term / (exponent_term + correction_term)

        numerator = first_term - second_term
        bias = 1 + numerator / (self.sqrt_scaling_factor * self.critical_density)

        return bias


class HaloBiasSheth99:
    """
    Implements halo bias as described in Sheth & Tormen (1999).
    This parametrization is only valid for 'fof' masses.

    Reference:
        Sheth & Tormen 1999 (https://arxiv.org/abs/astro-ph/9901122)
    """

    name = "Sheth99"

    def __init__(self, overdensity: str = "fof", use_delta_c_fit: bool = False):
        """
        Initializes the halo bias parameters.

        Args:
            overdensity (str): Overdensity definition (default is 'fof').
            use_delta_c_fit (bool): If True, uses the fit to the critical overdensity by Nakamura & Suto (1997).
        """
        self.overdensity = overdensity
        self.use_delta_c_fit = use_delta_c_fit
        self.power_law_index = 0.3
        self.scaling_factor = 0.707

    def compute_bias(
        self, cosmology: Cosmology, sigma_mass: float, scale_factor: float
    ) -> float:
        """
        Computes the halo bias as a function of mass variance and scale factor.

        Args:
            cosmology (Any): Cosmological parameters (unused in current implementation).
            sigma_mass (float): Variance of mass fluctuations.
            scale_factor (float): Scale factor of the universe.

        Returns:
            float: Computed halo bias.
        """
        method = "NakamuraSuto97" if self.use_delta_c_fit else "EdS"
        critical_density = get_delta_c(cosmology, scale_factor, method)

        peak_height = critical_density / sigma_mass
        scaled_peak_height_sq = self.scaling_factor * peak_height**2

        correction_term = 1 + 2 * self.power_law_index / (
            1 + scaled_peak_height_sq**self.power_law_index
        )
        bias = 1 + (scaled_peak_height_sq - 1 + correction_term) / critical_density

        return bias


class HaloBiasTinker10:
    """
    Implements halo bias as described in Tinker et al. (2010).
    This parametrization accepts spherical overdensity (S.O.) masses with
    200 < Delta < 3200, defined with respect to the matter density.
    It can be automatically translated to S.O. masses defined with respect to
    the critical density.

    Reference:
        Tinker et al. 2010 (https://arxiv.org/abs/1001.3162)
    """

    name = "Tinker10"

    def __init__(
        self, overdensity: Union[float, str] = 200, density_type: str = "matter"
    ):
        """
        Initializes the halo bias parameters.

        Args:
            overdensity (Union[float, str]): Overdensity threshold (default is 200).
            density_type (str): Type of density reference ('matter' or 'critical').
        """
        self.bias_coefficient = 0.183
        self.exponent_b = 1.5
        self.exponent_c = 2.4
        self.critical_density = get_delta_c(None, None, method="EdS")
        self.mass_definition = MassDefinition(overdensity, density_type)

    def compute_bias(
        self, cosmology: Cosmology, sigma_mass: float, scale_factor: float
    ) -> float:
        """
        Computes the halo bias as a function of mass variance and scale factor.

        Args:
            cosmology (Cosmology): Cosmological parameters.
            sigma_mass (float): Variance of mass fluctuations.
            scale_factor (float): Scale factor of the universe.

        Returns:
            float: Computed halo bias.
        """
        peak_height = self.critical_density / sigma_mass
        log_overdensity = jnp.log10(
            self.mass_definition.convert_to_matter_overdensity(cosmology, scale_factor)
        )

        exponential_term = jnp.exp(-((4.0 / log_overdensity) ** 4))
        coefficient_a = 1.0 + 0.24 * log_overdensity * exponential_term
        coefficient_c = 0.019 + 0.107 * log_overdensity + 0.19 * exponential_term
        exponent_a = 0.44 * log_overdensity - 0.88

        peak_height_scaled = peak_height**exponent_a

        bias = 1 - coefficient_a * peak_height_scaled / (
            peak_height_scaled + self.critical_density**exponent_a
        )
        bias += (
            self.bias_coefficient * peak_height**self.exponent_b
            + coefficient_c * peak_height**self.exponent_c
        )

        return bias
