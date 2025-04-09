from typing import Union, Tuple
import jax.numpy as jnp
from jax_cosmo.core import Cosmology
from interpax import Interpolator1D
from jax_cosmo.halos.hmbase import (
    MassDefinition,
    generate_massfunc_name,
    omega_x,
    get_delta_c,
)


class JAXMassFuncAngulo12:
    """
    Implements the Angulo et al. (2012) mass function model for halo abundance.

    This model describes the halo mass function using a fitting function
    based on cosmological simulations.
    """

    name: str = "Angulo12"

    def __init__(
        self, overdensity: Union[float, str] = "fof", density_type: str = "matter"
    ):
        """
        Initializes the mass function with default parameters from Angulo et al. (2012).

        Args:
            overdensity (Union[float, str], optional): Overdensity parameter. Defaults to "fof".
            density_type (str, optional): Type of density (e.g., "matter"). Defaults to "matter".
        """
        self.normalization_factor = 0.201
        self.shape_parameter = 2.08
        self.power_law_exponent = 1.7
        self.exponential_cutoff = 1.172

    def compute_fsigma(
        self, cosmo: Cosmology, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """
        Computes the halo mass function f(σ) based on the Angulo12 model.

        Args:
            cosmo (Cosmology): Cosmology object containing necessary parameters.
            sigma_mass (float): Standard deviation of the linear density field at mass scale M.
            scale_factor (float): Scale factor of the universe (a = 1 / (1+z)).
            log_mass (float): Logarithm of the halo mass.

        Returns:
            float: The value of f(σ), the mass function.
        """
        return (
            self.normalization_factor
            * ((self.shape_parameter / sigma_mass) ** self.power_law_exponent + 1.0)
            * jnp.exp(-self.exponential_cutoff / sigma_mass**2)
        )


class JAXMassFuncBocquet16:
    """
    Implements the mass function of Bocquet et al. (2016).

    This parametrization supports spherical overdensity (S.O.) masses defined with respect to:

    - Δ = 200, measured against the total matter density.
    - Δ = 200, measured against the critical density.
    - Δ = 500, measured against the critical density.
    """

    name: str = "Bocquet16"

    def __init__(
        self,
        overdensity: Union[float, str] = 200,
        density_type: str = "matter",
        includes_hydro_effects: bool = True,
    ):
        """
        Initializes the Bocquet16 mass function model with appropriate parameters.

        Args:
            overdensity (Union[float, str], optional): Overdensity value. Defaults to 200.
            density_type (str, optional): Reference density type ("matter" or "critical"). Defaults to "matter".
            includes_hydro_effects (bool, optional): Whether to include hydrodynamical effects. Defaults to True.
        """
        self.overdensity = overdensity
        self.density_type = density_type
        self.includes_hydro_effects = includes_hydro_effects
        self.mass_def_name = generate_massfunc_name(self.overdensity, self.density_type)

        # Mass function parameters (based on Bocquet et al. 2016)
        mass_function_params = {
            (True, "200m"): (0.228, 2.15, 1.69, 1.30, 0.285, -0.058, -0.366, -0.045),
            (False, "200m"): (0.175, 1.53, 2.55, 1.19, -0.012, -0.040, -0.194, -0.021),
            (True, "200c"): (0.202, 2.21, 2.00, 1.57, 1.147, 0.375, -1.074, -0.196),
            (False, "200c"): (0.222, 1.71, 2.24, 1.46, 0.269, 0.321, -0.621, -0.153),
            (True, "500c"): (0.180, 2.29, 2.44, 1.97, 1.088, 0.150, -1.008, -0.322),
            (False, "500c"): (0.241, 2.18, 2.35, 2.02, 0.370, 0.251, -0.698, -0.310),
        }

        self.A0, self.a0, self.b0, self.c0, self.Az, self.az, self.bz, self.cz = (
            mass_function_params[(self.includes_hydro_effects, self.mass_def_name)]
        )

    def _convert_M200c_to_M200m(
        self, cosmo: Cosmology, scale_factor: float
    ) -> Tuple[float, float]:
        """
        Converts M200c parameters to M200m, which is the base definition in Bocquet16.

        Args:
            cosmo (Cosmology): Cosmological model instance.
            scale_factor (float): Scale factor of the universe (a = 1 / (1+z)).

        Returns:
            Tuple[float, float]: (gamma, delta) correction factors.
        """
        redshift = 1 / scale_factor - 1
        Omega_m = omega_x(cosmo, scale_factor, "matter")

        gamma_0 = 3.54e-2 + Omega_m**0.09
        gamma_1 = 4.56e-2 + 2.68e-2 / Omega_m
        gamma_2 = 0.721 + 3.50e-2 / Omega_m
        gamma_3 = 0.628 + 0.164 / Omega_m
        delta_0 = -1.67e-2 + 2.18e-2 * Omega_m
        delta_1 = 6.52e-3 - 6.86e-3 * Omega_m

        gamma = gamma_0 + gamma_1 * jnp.exp(-(((gamma_2 - redshift) / gamma_3) ** 2))
        delta = delta_0 + delta_1 * redshift
        return gamma, delta

    def _convert_M500c_to_M200m(
        self, cosmo: Cosmology, scale_factor: float
    ) -> Tuple[float, float]:
        """
        Converts M500c parameters to M200m, which is the base definition in Bocquet16.

        Args:
            cosmo (Cosmology): Cosmological model instance.
            scale_factor (float): Scale factor of the universe (a = 1 / (1+z)).

        Returns:
            Tuple[float, float]: (alpha, beta) correction factors.
        """
        redshift = 1 / scale_factor - 1
        Omega_m = omega_x(cosmo, scale_factor, "matter")

        alpha_0 = 0.880 + 0.329 * Omega_m
        alpha_1 = 1.00 + 4.31e-2 / Omega_m
        alpha_2 = -0.365 + 0.254 / Omega_m
        alpha = alpha_0 * (alpha_1 * redshift + alpha_2) / (redshift + alpha_2)
        beta = -1.7e-2 + 3.74e-3 * Omega_m

        return alpha, beta

    def compute_fsigma(
        self, cosmo: Cosmology, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """
        Computes the halo mass function f(σ) based on Bocquet16.

        Args:
            cosmo (Cosmology): Cosmological model instance.
            sigma_mass (float): Standard deviation of the linear density field at mass scale M.
            scale_factor (float): Scale factor of the universe (a = 1 / (1+z)).
            log_mass (float): Logarithm of the halo mass.

        Returns:
            float: The value of f(σ), the mass function.
        """
        redshift_factor = 1.0 / scale_factor
        A_param = self.A0 * redshift_factor**self.Az
        a_param = self.a0 * redshift_factor**self.az
        b_param = self.b0 * redshift_factor**self.bz
        c_param = self.c0 * redshift_factor**self.cz

        f_sigma = (
            A_param
            * ((sigma_mass / b_param) ** -a_param + 1.0)
            * jnp.exp(-c_param / sigma_mass**2)
        )

        # Apply corrections for different mass definitions
        if self.mass_def_name == "200c":
            gamma, delta = self._convert_M200c_to_M200m(cosmo, scale_factor)
            f_sigma *= gamma + delta * log_mass
        elif self.mass_def_name == "500c":
            alpha, beta = self._convert_M500c_to_M200m(cosmo, scale_factor)
            f_sigma *= alpha + beta * log_mass

        return f_sigma


class JAXMassFuncDespali16:
    """
    Implements the Despali et al. (2016) mass function. This model extends the
    Sheth-Tormen formalism by incorporating ellipsoidal collapse corrections.

    This mass function supports different overdensity definitions and allows for ellipsoidal corrections.
    """

    name: str = "Despali16"

    def __init__(
        self,
        overdensity: Union[float, str] = 200,
        density_type: str = "matter",
        ellipsoidal: bool = False,
    ):
        """
        Initializes the Despali16 mass function model.

        Args:
            overdensity (Union[float, str], optional): Overdensity parameter (e.g., 200). Defaults to 200.
            density_type (str, optional): Type of density threshold ('matter' or 'critical'). Defaults to "matter".
            ellipsoidal (bool, optional): Whether to use ellipsoidal collapse corrections. Defaults to False.
        """
        self.overdensity = overdensity
        self.density_type = density_type
        self.ellipsoidal = ellipsoidal
        self.mass_definition = MassDefinition(overdensity, density_type)

        # Coefficients from Despali et al. (2016) Table 2
        coefficient_sets = {
            True: (0.3953, -0.1768, 0.7057, 0.2125, 0.3268, 0.2206, 0.1937, -0.04570),
            False: (0.3292, -0.1362, 0.7665, 0.2263, 0.4332, 0.2488, 0.2554, -0.1151),
        }
        A0, A1, a0, a1, a2, p0, p1, p2 = coefficient_sets[self.ellipsoidal]

        # Store polynomial coefficients (for use with jnp.polyval)
        self.coeffs_A = jnp.array([A1, A0])  # A(x) = A1 * x + A0
        self.coeffs_a = jnp.array([a2, a1, a0])  # a(x) = a2 * x² + a1 * x + a0
        self.coeffs_p = jnp.array([p2, p1, p0])  # p(x) = p2 * x² + p1 * x + p0

    def compute_fsigma(
        self, cosmo: Cosmology, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """
        Computes the mass function multiplicity function f(σ), which represents the
        fraction of mass contained in halos of a given variance.

        Args:
            cosmo (Cosmology): The cosmological model.
            sigma_mass (float): The standard deviation of the linear density field.
            scale_factor (float): The scale factor `a = 1 / (1 + z)`.
            log_mass (float): The logarithm of the halo mass.

        Returns:
            float: The computed f(σ) value.
        """
        # Compute linear collapse threshold using Nakamura & Suto (1997) prescription
        delta_collapse = get_delta_c(cosmo, scale_factor, "NakamuraSuto97")

        # Compute the virial overdensity contrast
        virial_overdensity = self.mass_definition.compute_virial_overdensity(
            cosmo, scale_factor
        )

        # Compute the logarithmic ratio of halo overdensity to virial overdensity
        overdensity_ratio_log = jnp.log10(
            self.mass_definition.get_overdensity(cosmo, scale_factor)
            / virial_overdensity
        )

        # Evaluate polynomial functions using jnp.polyval
        coeff_A = jnp.polyval(self.coeffs_A, overdensity_ratio_log)
        coeff_a = jnp.polyval(self.coeffs_a, overdensity_ratio_log)
        coeff_p = jnp.polyval(self.coeffs_p, overdensity_ratio_log)

        # Compute ν' (ellipsoidal collapse correction factor)
        nu_prime = coeff_a * (delta_collapse / sigma_mass) ** 2

        # Compute f(σ) using Despali et al. (2016) formula
        return (
            2.0
            * coeff_A
            * jnp.sqrt(nu_prime / (2.0 * jnp.pi))
            * jnp.exp(-0.5 * nu_prime)
            * (1.0 + nu_prime**-coeff_p)
        )


class JAXMassFuncJenkins01:
    """
    Implements the Jenkins et al. (2001) mass function.

    Reference: [Jenkins et al. 2001](https://arxiv.org/abs/astro-ph/0005260).

    This parametrization is valid only for `fof` (friends-of-friends) masses.
    """

    name: str = "Jenkins01"

    def __init__(
        self, overdensity: Union[float, str] = "fof", density_type: str = "matter"
    ):
        """
        Initializes the Jenkins01 mass function model.

        Args:
            overdensity (Union[float, str], optional): Overdensity parameter (default: "fof").
            density_type (str, optional): Type of density ('matter' or 'critical'). Defaults to "matter".
        """
        self.mass_definition = MassDefinition(overdensity, density_type)
        self.amplitude = 0.315
        self.shape_param = 0.61
        self.steepness_param = 3.8

    def compute_fsigma(
        self, cosmo: Cosmology, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """
        Computes the mass function multiplicity function f(σ), which represents the fraction of mass contained
        in halos of a given variance.

        Args:
            cosmo (Cosmology): The cosmological model.
            sigma_mass (float): Standard deviation of the linear density field.
            scale_factor (float): Scale factor.
            log_mass (float): Logarithm of the halo mass.

        Returns:
            float: The computed f(σ) value.
        """
        log_sigma = -jnp.log(sigma_mass)
        exponent_term = jnp.abs(log_sigma + self.shape_param) ** self.steepness_param
        return self.amplitude * jnp.exp(-exponent_term)


class JAXMassFuncPress74:
    """Implements the mass function of Press & Schechter (1974).

    This parametrization is only valid for 'fof' masses.
    """

    name: str = "Press74"

    def __init__(
        self, overdensity: Union[float, str] = "fof", density_type: str = "matter"
    ):
        """
        Initializes the mass function with a specified overdensity and density type.

        Args:
            overdensity (Union[float, str], optional): The overdensity parameter (default is "fof").
            density_type (str, optional): The type of density (default is "matter").
        """
        self.mass_definition = MassDefinition(overdensity, density_type)
        self._normalization_factor = jnp.sqrt(2 / jnp.pi)

    def compute_fsigma(
        self, cosmo: Cosmology, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """Computes the Press & Schechter mass function.

        Args:
            cosmology (Cosmology): An instance of the Cosmology class containing cosmological parameters.
            sigma_mass (float): The variance in the mass function.
            scale_factor (float): The scale factor at the time of calculation.
            log_mass (float): The logarithm of the mass.

        Returns:
            float: The value of the mass function at the given conditions.
        """
        critical_overdensity = get_delta_c(cosmo, scale_factor, "EdS")
        nu = critical_overdensity / sigma_mass
        return self._normalization_factor * nu * jnp.exp(-0.5 * nu**2)


class JAXMassFuncSheth99:
    """Implements the mass function of Sheth & Tormen (1999).

    This parametrization is only valid for 'fof' masses. For details, refer to the paper:
    https://arxiv.org/abs/astro-ph/9901122
    """

    name: str = "Sheth99"

    def __init__(
        self,
        overdensity: Union[float, str] = "fof",
        density_type: str = "matter",
        use_custom_delta_c: bool = False,
    ):
        """
        Initializes the mass function with a specified overdensity, density type, and delta_c fit option.

        Args:
            overdensity (Union[float, str], optional): The overdensity parameter (default is "fof").
            density_type (str, optional): The type of density (default is "matter").
            use_custom_delta_c (bool, optional): Whether to use the Nakamura-Suto (1997) delta_c fit (default is False).
        """
        self.mass_definition = MassDefinition(overdensity, density_type)
        self.use_custom_delta_c = use_custom_delta_c
        self.amplitude = 0.21615998645
        self.slope = 0.3
        self.scale = 0.707

    def compute_fsigma(
        self, cosmo: Cosmology, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """Computes the Sheth & Tormen (1999) mass function.

        Args:
            cosmology (Cosmology): An instance of the Cosmology class containing cosmological parameters.
            sigma_mass (float): The variance in the mass function.
            scale_factor (float): The scale factor at the time of calculation.
            log_mass (float): The logarithm of the mass.

        Returns:
            float: The value of the mass function at the given conditions.
        """
        if self.use_custom_delta_c:
            critical_overdensity = get_delta_c(cosmo, scale_factor, "NakamuraSuto97")
        else:
            critical_overdensity = get_delta_c(cosmo, scale_factor, "EdS")

        nu = critical_overdensity / sigma_mass
        return (
            nu
            * self.amplitude
            * (1.0 + (self.scale * nu**2) ** (-self.slope))
            * jnp.exp(-self.scale * nu**2 / 2.0)
        )


class JAXMassFuncTinker08:
    def __init__(
        self, overdensity: Union[float, str] = 200, density_type: str = "matter"
    ):
        """Initializes the Tinker08 mass function model.

        Args:
            overdensity: Overdensity parameter (default: 200).
            density_type: Type of density ('matter' or 'critical').
        """
        name: str = "Tinker08"

        # Initialize mass definition and compute corresponding matter overdensity
        self.mass_definition = MassDefinition(overdensity, density_type)

        # Define lookup tables for overdensity and fitting parameters
        overdensity_grid = jnp.array(
            [200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0]
        )
        alpha_grid = jnp.array(
            [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260]
        )
        beta_grid = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        gamma_grid = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        phi_grid = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])

        # Store interpolators for fitting parameters
        log_overdensity = jnp.log10(overdensity_grid)
        self.interp_alpha = Interpolator1D(log_overdensity, alpha_grid, method="linear")
        self.interp_beta = Interpolator1D(log_overdensity, beta_grid, method="linear")
        self.interp_gamma = Interpolator1D(log_overdensity, gamma_grid, method="linear")
        self.interp_phi = Interpolator1D(log_overdensity, phi_grid, method="linear")

    def compute_fsigma(
        self, cosmo, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """Computes the Tinker08 mass function for a given mass scale.

        Args:
            cosmo: Cosmology object.
            sigma_mass: Mass variance.
            scale_factor: Scale factor (a).
            log_mass: Logarithmic mass.

        Returns:
            f(sigma), the mass function value.
        """
        delta = self.mass_definition.convert_to_matter_overdensity(cosmo, scale_factor)
        log_delta = jnp.log10(delta)

        # Interpolated parameter values
        alpha = self.interp_alpha(log_delta) * scale_factor**0.14
        beta = self.interp_beta(log_delta) * scale_factor**0.06
        gamma_exponent = -((0.75 / (log_delta - 1.8750612633)) ** 1.2)
        gamma = 10.0**gamma_exponent
        phi = self.interp_gamma(log_delta) * scale_factor**gamma

        # Compute the mass function
        return (
            alpha
            * ((phi / sigma_mass) ** beta + 1)
            * jnp.exp(-self.interp_phi(log_delta) / sigma_mass**2)
        )


class JAXMassFuncTinker10:
    """Implements the mass function of Tinker et al. (2010).

    This parametrization accepts spherical overdensity (S.O.) masses within the range
    200 < overdensity < 3200, defined with respect to the matter density. These values
    can be automatically converted to S.O. masses defined relative to the critical density.

    Reference: https://arxiv.org/abs/1001.3162
    """

    name: str = "Tinker10"

    def __init__(
        self,
        overdensity: Union[float, str] = 200,
        density_type: str = "matter",
        normalize_across_redshifts: bool = False,
    ):
        """
        Initializes the Tinker et al. (2010) mass function with specified overdensity and density type.

        Args:
            overdensity (Union[float, str], optional): Overdensity value (default is 200).
            density_type (str, optional): Type of density ('matter' or 'critical', default is 'matter').
            normalize_across_redshifts (bool, optional): Whether to normalize across redshifts (default is False).
        """
        self.overdensity = overdensity
        self.density_type = density_type
        self.normalize_across_redshifts = normalize_across_redshifts
        self.mass_definition = MassDefinition(overdensity, density_type)

        overdensity_values = jnp.array(
            [200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0]
        )
        alpha_values = jnp.array(
            [0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327]
        )
        beta_values = jnp.array(
            [0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702]
        )
        gamma_values = jnp.array(
            [0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81]
        )
        phi_values = jnp.array(
            [-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49]
        )
        eta_values = jnp.array(
            [-0.243, -0.261, -0.261, -0.273, -0.278, -0.301, -0.301, -0.319, -0.336]
        )

        log_overdensity = jnp.log10(overdensity_values)
        self.alpha_interp = Interpolator1D(
            log_overdensity, alpha_values, method="linear"
        )
        self.eta_interp = Interpolator1D(log_overdensity, eta_values, method="linear")
        self.beta_interp = Interpolator1D(log_overdensity, beta_values, method="linear")
        self.gamma_interp = Interpolator1D(
            log_overdensity, gamma_values, method="linear"
        )
        self.phi_interp = Interpolator1D(log_overdensity, phi_values, method="linear")

        if self.normalize_across_redshifts:
            p_values = jnp.array(
                [-0.158, -0.195, -0.213, -0.254, -0.281, -0.349, -0.367, -0.435, -0.504]
            )
            q_values = jnp.array(
                [0.0128, 0.0128, 0.0143, 0.0154, 0.0172, 0.0174, 0.0199, 0.0203, 0.0205]
            )
            self.p_interp = Interpolator1D(log_overdensity, p_values, method="linear")
            self.q_interp = Interpolator1D(log_overdensity, q_values, method="linear")

    def compute_fsigma(
        self, cosmo: Cosmology, sigma_mass: float, scale_factor: float, log_mass: float
    ) -> float:
        """Computes the Tinker et al. (2010) mass function.

        Args:
            cosmo (Cosmology): An instance of the Cosmology class containing cosmological parameters.
            sigma_mass (float): The variance in the mass function.
            scale_factor (float): The scale factor at the time of calculation.
            log_mass (float): The logarithm of the mass.

        Returns:
            float: The computed mass function value.
        """
        overdensity_matter = self.mass_definition.convert_to_matter_overdensity(
            cosmo, scale_factor
        )
        peak_height = get_delta_c(cosmo, scale_factor, "EdS_approx") / sigma_mass
        log_delta = jnp.log10(overdensity_matter)

        # Limit redshift evolution to z <= 3
        scale_factor = jnp.clip(scale_factor, 0.25, 1)

        alpha = self.eta_interp(log_delta) * scale_factor ** (-0.27)
        beta = self.beta_interp(log_delta) * scale_factor ** (-0.20)
        gamma = self.gamma_interp(log_delta) * scale_factor**0.01
        phi = self.phi_interp(log_delta) * scale_factor**0.08
        base_amplitude = self.alpha_interp(log_delta)

        if self.normalize_across_redshifts:
            redshift = 1 / scale_factor - 1
            p_factor = self.p_interp(log_delta)
            q_factor = self.q_interp(log_delta)
            base_amplitude *= jnp.exp(redshift * (p_factor + q_factor * redshift))

        return (
            peak_height
            * base_amplitude
            * (1 + (beta * peak_height) ** (-2 * phi))
            * (peak_height ** (2 * alpha) * jnp.exp(-0.5 * gamma * peak_height**2))
        )


class JAXMassFuncWatson13:
    """Implements the mass function of Watson et al. (2013).

    This parametrization is valid for both Friends-of-Friends (FoF) masses and
    spherical overdensity (S.O.) masses.

    Reference: https://arxiv.org/abs/1212.0095
    """

    name: str = "Watson13"

    def __init__(
        self, overdensity: Union[float, str] = "fof", density_type: str = "matter"
    ):
        """
        Initializes the Watson et al. (2013) mass function with the given overdensity definition.

        Args:
            overdensity (Union[float, str], optional): Overdensity value ('fof' for Friends-of-Friends or a numeric overdensity for S.O.).
            density_type (str, optional): Type of density ('matter' or 'critical', default is 'matter').
        """
        self.overdensity = overdensity
        self.density_type = density_type
        self.mass_definition = MassDefinition(overdensity, density_type)

    def compute_fsigma_fof(
        self,
        cosmology: Cosmology,
        sigma_mass: float,
        scale_factor: float,
        log_mass: float,
    ) -> float:
        """Computes the mass function for Friends-of-Friends (FoF) masses.

        Args:
            cosmology (Cosmology): The cosmology instance containing cosmological parameters.
            mass_variance (float): The variance in the mass function.
            scale_factor (float): The scale factor at the time of calculation.
            log_mass (float): The logarithm of the mass.

        Returns:
            float: The computed mass function value for FoF masses.
        """
        normalization = 0.282
        exponent_a = 2.163
        exponent_b = 1.406
        exponent_c = 1.210

        return (
            normalization
            * ((exponent_b / sigma_mass) ** exponent_a + 1.0)
            * jnp.exp(-exponent_c / sigma_mass**2)
        )

    def compute_fsigma_so(
        self,
        cosmology: Cosmology,
        sigma_mass: float,
        scale_factor: float,
        log_mass: float,
    ) -> float:
        """Computes the mass function for spherical overdensity (S.O.) masses.

        Args:
            cosmology (Cosmology): The cosmology instance containing cosmological parameters.
            sigma_mass (float): The variance in the mass function.
            scale_factor (float): The scale factor at the time of calculation.
            log_mass (float): The logarithm of the mass.

        Returns:
            float: The computed mass function value for S.O. masses.
        """
        matter_density_fraction = omega_x(cosmology, scale_factor, "matter")
        relative_overdensity = self.mass_definition.overdensity / 178

        # Define parameters based on scale factor (redshift dependence)
        if scale_factor == 1:
            normalization = 0.194
            exponent_a = 1.805
            exponent_b = 2.267
            exponent_c = 1.287
        elif scale_factor < 1 / (1 + 6):  # Corresponds to redshift z > 6
            normalization = 0.563
            exponent_a = 3.810
            exponent_b = 0.874
            exponent_c = 1.453
        else:
            normalization = matter_density_fraction * (
                1.097 * scale_factor**3.216 + 0.074
            )
            exponent_a = matter_density_fraction * (5.907 * scale_factor**3.058 + 2.349)
            exponent_b = matter_density_fraction * (3.136 * scale_factor**3.599 + 2.344)
            exponent_c = 1.318

        base_mass_function = (
            normalization
            * ((exponent_b / sigma_mass) ** exponent_a + 1.0)
            * jnp.exp(-exponent_c / sigma_mass**2)
        )

        # Additional correction factor for overdensities other than 178
        correction_factor = jnp.exp(0.023 * (relative_overdensity - 1.0))
        exponent_d = -0.456 * matter_density_fraction - 0.139
        gamma_factor = (
            correction_factor
            * relative_overdensity**exponent_d
            * jnp.exp(0.072 * (1.0 - relative_overdensity) / sigma_mass**2.130)
        )

        return base_mass_function * gamma_factor

    def compute_fsigma(
        self,
        cosmology: Cosmology,
        sigma_mass: float,
        scale_factor: float,
        log_mass: float,
    ) -> float:
        """Computes the mass function based on whether FoF or S.O. mass definitions are used.

        Args:
            cosmology (Cosmology): The cosmology instance containing cosmological parameters.
            mass_variance (float): The variance in the mass function.
            scale_factor (float): The scale factor at the time of calculation.
            log_mass (float): The logarithm of the mass.

        Returns:
            float: The computed mass function value.
        """
        if self.mass_definition.overdensity == "fof":
            return self.compute_fsigma_fof(
                cosmology, sigma_mass, scale_factor, log_mass
            )
        return self.compute_fsigma_so(cosmology, sigma_mass, scale_factor, log_mass)
