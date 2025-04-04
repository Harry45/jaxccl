import jax
import jax.numpy as jnp
from jax_cosmo.cclconstants import PhysicalConstants, CCLSplineParams, CCLGSLParams, SPECIES_CRIT, SPECIES_M
from jax_cosmo.core import Cosmology
from typing import Union, Tuple, Callable
from jax_cosmo.background import Esqr
import jax_cosmo.power as jcp
from quadax import simpson
from interpax import Interpolator2D

CCLCST = PhysicalConstants()
CCL_SPLINE_PARAMS = CCLSplineParams()
CCL_GSL_PARAMS = CCLGSLParams()

def omega_x(cosmo, a, label):
    """Computes the density parameter Omega_x at a given scale factor for a specified species.

    This function calculates the fractional energy density (Ω_x) of a cosmological component
    at a given scale factor `a`, normalized by the critical density.

    Args:
        cosmo: An object containing cosmological parameters (e.g., Omega_c, Omega_b).
        a (float): The scale factor at which to evaluate the density parameter.
        label (str): The species label (e.g., "matter", "lambda", "radiation").

    Returns:
        float: The density parameter Ω_x(a) for the specified species.

    """
    # Compute the Hubble normalization factor
    hnorm = jnp.sqrt(Esqr(cosmo, a))

    # Handle different cosmological species
    if label == SPECIES_CRIT:
        return 1.0

    elif label == SPECIES_M:
        quantity_1 = (cosmo.Omega_c + cosmo.Omega_b) / (a**3 * hnorm**2)
        quantity_2 = 0.0  # Placeholder for potential neutrino contribution
        return quantity_1 + quantity_2

def rho_x(cosmo: Cosmology, a: Union[float, jnp.ndarray], label: str) -> Union[float, jnp.ndarray]:
    """Computes the physical density of a given component in the universe.

    This function calculates the density of a specified cosmological component
    (e.g., matter, dark energy, radiation) at a given scale factor.

    Args:
        cosmo (Cosmology): The cosmological model containing parameters such as Hubble constant.
        a (Union[float, np.ndarray]): The scale factor at which the density is computed.
        label (str): The label identifying the cosmological component (e.g., 'matter', 'radiation').

    Returns:
        Union[float, np.ndarray]: The density of the specified component in units of
        critical density times Omega_x.
    """
    hnorm = jnp.sqrt(Esqr(cosmo, a))
    rhocrit = CCLCST.RHO_CRITICAL * cosmo.h**2 * hnorm**2
    return omega_x(cosmo, a, label) * rhocrit



def mass2radius_lagrangian(cosmo: Cosmology, mass: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Computes the Lagrangian radius for a given halo mass.

    The Lagrangian radius is defined as the radius that encloses the given halo mass
    under the assumption of a homogeneous Universe.

    Args:
        cosmo (Cosmology): An object containing cosmological parameters.
        mass (Union[float, np.ndarray]): The mass of the halo in solar masses.

    Returns:
        Union[float, np.ndarray]: The corresponding Lagrangian radius in the same shape as `mass`.
    """
    mass_use = jnp.atleast_1d(mass)
    rho_matter = rho_x(cosmo, 1.0, 'matter')  # Density of matter at a = 1
    radius_cube = mass_use / (4.0 / 3.0 * jnp.pi * rho_matter)
    radius = jnp.cbrt(radius_cube)  # Cube root to get radius

    return radius[0] if jnp.ndim(mass) == 0 else radius



def get_delta_c(cosmo: Cosmology, scale_factor: float, method: str = "EdS") -> Union[float, jnp.ndarray]:
    """
    Computes the linear collapse threshold, which represents the density contrast
    required for a region to collapse under self-gravity.

    Args:
        cosmo (Cosmology): A cosmology object providing the background cosmology.
        scale_factor (float): The scale factor of the universe.
        method (str, optional): The prescription used to compute the linear collapse threshold. Options include:
            - "EdS": Standard spherical collapse prediction in an Einstein-de Sitter universe.
              Uses the formula: δ_c = (3/20) * (12π)^(2/3) ≈ 1.68647.
            - "EdS_approx": A common approximation to the EdS result, δ_c = 1.686.
            - "NakamuraSuto97": Uses the prescription from
              [Nakamura & Suto (1997)](https://arxiv.org/abs/astro-ph/9612074).
            - "Mead16": Uses the prescription from
              [Mead et al. (2016)](https://arxiv.org/abs/1602.02154).

    Returns:
        Union[float, np.ndarray]: The computed linear collapse threshold δ_c.

    Raises:
        ValueError: If an unknown method is provided.
    """
    # Standard collapse threshold in an Einstein-de Sitter (EdS) universe
    delta_c_eds = 1.68647019984

    if method == "EdS":
        return delta_c_eds
    elif method == "EdS_approx":
        return 1.686
    elif method == "NakamuraSuto97":
        omega_matter = omega_x(cosmo, scale_factor, "matter")
        return delta_c_eds * (1 + 0.012299 * jnp.log10(omega_matter))
    elif method == "Mead16":
        omega_matter = omega_x(cosmo, scale_factor, "matter")
        sigma_8_scaled = cosmo.sigma8 * cosmo.growth_factor(scale_factor)
        correction_sigma8 = (1.59 + 0.0314 * jnp.log(sigma_8_scaled))
        correction_omega = (1 + 0.0123 * jnp.log10(omega_matter))
        return correction_sigma8 * correction_omega
    else:
        raise ValueError(f"Unknown collapse threshold method: {method}")

def sigmaM_m2r(cosmo: Cosmology, halomass: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Computes the smoothing radius corresponding to a given halo mass.

    The smoothing radius is defined as the Lagrangian radius enclosing the given halo
    mass, assuming a homogeneous Universe.

    Args:
        cosmo (Cosmology): An object containing cosmological parameters.
        halomass (Union[float, jnp.ndarray]): The halo mass in solar masses.

    Returns:
        Union[float, jnp.ndarray]: The smoothing radius in the same shape as `halomass`.
    """
    # Comoving matter density at a = 1
    rho_m = rho_x(cosmo, 1.0, 'matter')

    # Compute the smoothing radius
    smooth_radius = jnp.cbrt((3.0 * halomass) / (4.0 * jnp.pi * rho_m))

    return smooth_radius

class MassDefinition:
    def __init__(self, overdensity: Union[float, str] = 200, density_type: str = "matter"):
        self.overdensity = overdensity
        self.density_type = density_type

    def compute_virial_overdensity(self, cosmo, scale_factor: float) -> float:
        """Computes the virial collapse overdensity contrast relative to the critical density
        in a ΛCDM model using the Bryan & Norman (1998) fitting function.

        Args:
            cosmo: A Cosmology object.
            scale_factor: The cosmic scale factor.

        Returns:
            The virial overdensity relative to the specified density type.
        """
        omega_matter = omega_x(cosmo, scale_factor, "matter")
        omega_deviation = omega_matter - 1
        virial_overdensity = (18 * jnp.pi**2 + 82 * omega_deviation - 39 * omega_deviation**2)
        return (
            virial_overdensity / omega_matter
            if self.density_type == "matter"
            else virial_overdensity
        )

    def get_overdensity(self, cosmo, scale_factor: float) -> float:
        """Retrieves the overdensity parameter associated with this mass definition.

        Args:
            cosmo: A Cosmology object.
            scale_factor: The cosmic scale factor.

        Returns:
            The overdensity parameter.
        """
        if self.overdensity == "fof":
            raise ValueError(
                "FoF masses do not have a defined overdensity and cannot be converted."
            )
        return (
            self.compute_virial_overdensity(cosmo, scale_factor)
            if self.overdensity == "vir"
            else self.overdensity
        )

    def convert_to_matter_overdensity(self, cosmo, scale_factor: float) -> float:
        """Converts the overdensity parameter to be relative to the matter density.

        Args:
            cosmo: A Cosmology object.
            scale_factor: The cosmic scale factor.

        Returns:
            The equivalent overdensity parameter relative to matter.
        """
        overdensity_value = self.get_overdensity(cosmo, scale_factor)
        if self.density_type == "matter":
            return overdensity_value
        omega_target = omega_x(cosmo, scale_factor, self.density_type)
        omega_matter = omega_x(cosmo, scale_factor, "matter")
        return overdensity_value * omega_target / omega_matter

    def get_mass(self, cosmo: Cosmology,
                 radius: Union[float, jnp.ndarray],
                 scale_factor: float) -> Union[float, jnp.ndarray]:
        """
        Translates a halo radius into a mass.

        Args:
            cosmo (Cosmology): A Cosmology object.
            radius (float or jnp.ndarray): Halo radius in Mpc (physical, not comoving).
            scale_factor (float): Scale factor.

        Returns:
            float or jnp.ndarray: Halo mass in units of solar mass.
        """
        radius_use = jnp.atleast_1d(radius)
        delta = self.get_overdensity(cosmo, scale_factor)
        rho_x_calc = rho_x(cosmo, scale_factor, self.density_type)
        mass = 4.18879020479 * rho_x_calc * delta * radius_use**3
        if jnp.ndim(radius) == 0:
            return float(mass[0])
        return mass

    def get_radius(self, cosmo: Cosmology,
                   mass: Union[float, jnp.ndarray],
                   scale_factor: float) -> Union[float, jnp.ndarray]:
        """
        Translates a halo mass into a radius.

        Args:
            cosmo (Cosmology): A Cosmology object.
            mass (float or jnp.ndarray): Halo mass in units of solar mass.
            scale_factor (float): Scale factor.

        Returns:
            float or jnp.ndarray: Halo radius in Mpc (physical, not comoving).
        """
        mass_use = jnp.atleast_1d(mass)
        delta = self.get_overdensity(cosmo, scale_factor)
        rho_x_calc = rho_x(cosmo, scale_factor, self.density_type)
        radius = (mass_use / (4.18879020479 * delta * rho_x_calc))**(1./3.)
        if jnp.ndim(mass) == 0:
            return float(radius[0])
        return radius


def parse_mass_def(mass_def: Union[int, str]) -> Union[int, str]:
    """Parses a mass definition string and returns the appropriate value.

    Args:
        mass_def (Union[int, str]): The mass definition, which can be:
            - A string ending in "c" with a numeric prefix (e.g., "500c", "200c"),
            - The string "vir",
            - An integer (will be returned as-is).

    Returns:
        Union[int, str]: The extracted integer if the input is of the form "<number>c",
        the string "vir" if specified, or the input integer if provided.

    Raises:
        ValueError: If the input is an invalid mass definition.
    """
    if isinstance(mass_def, int):
        return mass_def
    if isinstance(mass_def, str) and mass_def[:-1].isdigit():
        return int(mass_def[:-1])
    elif mass_def == "vir":
        return "vir"
    else:
        raise ValueError("Invalid mass definition")

def generate_massfunc_name(delta: int, density_type: str) -> str:
    """Generates a standardized name for a mass function based on the overdensity and density type.

    Args:
        delta (int): The overdensity value.
        density_type (str): The type of density (e.g., "matter").

    Returns:
        str: The standardized mass function name.
    """
    return f"{delta}{density_type[0]}"

def w_tophat(k_times_r: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Computes the top-hat window function in Fourier space.

    This function implements a Maclaurin expansion for small kR values to avoid numerical issues.

    Args:
        kR (Union[float, jnp.ndarray]): Dimensionless wave number times the smoothing radius.

    Returns:
        Union[float, jnp.ndarray]: The value of the top-hat window function.
    """
    k_times_r_2 = k_times_r ** 2

    # Maclaurin expansion for small kR to avoid numerical instabilities
    small_kR = k_times_r < 0.1
    w_small = 1. + k_times_r_2 * (-1.0 / 10.0 + k_times_r_2 * (1.0 / 280.0 +
                k_times_r_2 * (-1.0 / 15120.0 + k_times_r_2 * (1.0 / 1330560.0 +
                k_times_r_2 * (-1.0 / 172972800.0)))))

    # General case
    w_general = 3. * (jnp.sin(k_times_r) - k_times_r * jnp.cos(k_times_r)) / (k_times_r_2 * k_times_r)

    # Use JAX where for conditional branching
    return jnp.where(small_kR, w_small, w_general)


def sigmaR_integrand(logk: Union[float, jnp.ndarray],
                     cosmo: Cosmology,
                     a: float,
                     radius: float) -> Union[float, jnp.ndarray]:
    """Computes the integrand for the sigma_R calculation.

    Args:
        logk (Union[float, jnp.ndarray]): Logarithm of the wave number (log10(k)).
        cosmo (Cosmology): Cosmological parameters object.
        a (float): Scale factor.
        radius (float): Smoothing radius.

    Returns:
        Union[float, jnp.ndarray]: The value of the integrand.
    """
    wavenumber = 10. ** logk  # Convert log10(k) back to k
    if jcp.USE_EMU:
        pk_lin = jcp.linear_matter_power_emu(cosmo, wavenumber, a)
    else:
        pk_lin = jcp.linear_matter_power(cosmo, wavenumber, a) # Get power spectrum
    k_times_r = wavenumber * radius
    w_kernel = w_tophat(k_times_r)

    return pk_lin * wavenumber**3 * w_kernel**2  # k^3 weighting for integral

def sigmaR(cosmo: Cosmology, radius: float, a: float) -> float:
    """
    Computes the variance of the linear density field (sigma_R) for a given smoothing scale R and scale factor a.

    Args:
        cosmo (Cosmology): Cosmological parameter object.
        R (float): Smoothing radius in comoving Mpc.
        a (float): Scale factor.

    Returns:
        float: The square root of the variance sigma(R).
    """
    # Define the integrand function
    def integrand(log_k):
        return sigmaR_integrand(log_k, cosmo, a, radius)

    # Perform the integral over log10(k) space
    k_min_log = jnp.log10(CCL_SPLINE_PARAMS.K_MIN)
    k_max_log = jnp.log10(CCL_SPLINE_PARAMS.K_MAX)

    k_values = jnp.linspace(k_min_log, k_max_log, num=100)
    vectorized_integrand = jax.vmap(integrand)

    # Compute values efficiently
    integrand_values = vectorized_integrand(k_values).squeeze()
    sigma_R = simpson(y=integrand_values, x=k_values)

    # Compute final sigma_R
    return jnp.sqrt(sigma_R * jnp.log(10) / (2 * jnp.pi**2))

def compute_sigma(cosmo: Cosmology):
    """Computes the variance of the matter density field (sigma) on a grid of mass and scale factor values.

    Args:
        cosmo (Cosmology): Cosmology object containing necessary cosmological parameters.

    Returns:
        Interpolator2D: A 2D interpolator object that allows evaluation of sigma at arbitrary mass and scale factor values.

    Notes:
        - The function constructs a 2D grid in log-mass and scale factor space.
        - It computes sigma using the sigmaR function at each grid point.
        - The final result is a bicubic 2D interpolator for efficient queries.
    """
    # Define mass and scale factor grids
    num_a = CCL_SPLINE_PARAMS.A_SPLINE_NA_SM + CCL_SPLINE_PARAMS.A_SPLINE_NLOG_SM - 1
    num_m = CCL_SPLINE_PARAMS.LOGM_SPLINE_NM

    # Log-spaced mass array
    log_mass = jnp.linspace(CCL_SPLINE_PARAMS.LOGM_SPLINE_MIN,
                            CCL_SPLINE_PARAMS.LOGM_SPLINE_MAX, num_m)

    # Log-linear spaced scale factor array
    scale_factors = jnp.concatenate([
        jnp.geomspace(CCL_SPLINE_PARAMS.A_SPLINE_MINLOG_SM,
                      CCL_SPLINE_PARAMS.A_SPLINE_MIN_SM,
                      CCL_SPLINE_PARAMS.A_SPLINE_NLOG_SM),
        jnp.linspace(CCL_SPLINE_PARAMS.A_SPLINE_MIN_SM,
                     CCL_SPLINE_PARAMS.A_SPLINE_MAX,
                     CCL_SPLINE_PARAMS.A_SPLINE_NA_SM)
    ])

    # Compute sigma
    def compute_sigma_element(log_m, a_sf):
        radius = sigmaM_m2r(cosmo, 10 ** log_m) * cosmo.h
        return jnp.log(sigmaR(cosmo, radius, a_sf))

    # Compute sigma on a grid
    log_m_grid, a_grid = jnp.meshgrid(log_mass, scale_factors, indexing="ij")
    func = jax.vmap(lambda a: jax.vmap(lambda log_m: compute_sigma_element(log_m, a))(log_mass))(scale_factors)
    func = func.squeeze().T

    # Create 2D interpolator
    interpolator = Interpolator2D(x=log_mass, y=scale_factors, f=func, method='cubic2')
    return interpolator

def sigmaM(log_halomass: jnp.ndarray,
           a: Union[float, jnp.ndarray],
           interpolator: Interpolator2D) -> jnp.ndarray:
    """Computes the variance of the matter density field (sigma) for given halo masses and scale factors.

    Args:
        log_halomass (jnp.ndarray): Logarithm of halo masses.
        a (Union[float, jnp.ndarray]): Scale factor(s).
        interpolator (Interpolator2D): Precomputed interpolator for sigma.

    Returns:
        jnp.ndarray: The computed sigma values.
    """
    # Evaluate the interpolation
    lgsigmaM = interpolator(log_halomass, a)
    return jnp.exp(lgsigmaM)

def logsigmaM(log_halomass: jnp.ndarray,
              a: Union[float, jnp.ndarray],
              interpolator: Interpolator2D) -> jnp.ndarray:
    """Computes the logarithm of the variance of the matter density field (sigma) for given halo masses and scale factors.

    Args:
        log_halomass (jnp.ndarray): Logarithm of halo masses.
        a (Union[float, jnp.ndarray]): Scale factor(s).
        interpolator (Interpolator2D): Precomputed interpolator for sigma.

    Returns:
        jnp.ndarray: The computed log-sigma values.
    """
    return interpolator(log_halomass, a)

def d_sigmaM_dlogM(log_halomass: jnp.ndarray,
                   a: Union[float, jnp.ndarray],
                   interpolator: Interpolator2D) -> jnp.ndarray:
    """Computes the derivative of sigma with respect to log mass.

    Args:
        log_halomass (jnp.ndarray): Logarithm of halo masses.
        a (Union[float, jnp.ndarray]): Scale factor(s).
        interpolator (Interpolator2D): Precomputed interpolator for sigma.

    Returns:
        jnp.ndarray: The computed derivative values.
    """
    # Define the function for a scalar log_halomass
    d_sigmaM_dlogM_fn = jax.grad(logsigmaM, argnums=0)

    # Vectorize the function to handle vector inputs
    d_sigmaM_dlogM_vec = jax.vmap(d_sigmaM_dlogM_fn, in_axes=(0, None, None))
    return -d_sigmaM_dlogM_vec(log_halomass, a, interpolator)

def get_logM_sigM(cosmo: Cosmology,
                  mass: jnp.ndarray,
                  a: Union[float, jnp.ndarray],
                  interpolator: Interpolator2D=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes log-mass, sigma mass, and its derivative.

    Args:
        cosmo (Cosmology): Cosmology object containing necessary parameters.
        mass (jnp.ndarray): Array of halo masses.
        a (Union[float, jnp.ndarray]): Scale factor(s).
        interpolator (Interpolator2D, optional): Precomputed interpolator for sigma. Defaults to None.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Log-mass, sigma mass, and its derivative.
    """
    if interpolator is None:
        interpolator = compute_sigma(cosmo)
    log_mass = jnp.log10(mass)
    sigma_mass = sigmaM(log_mass, a, interpolator)
    dlns_dlogM = d_sigmaM_dlogM(log_mass, a, interpolator)
    return log_mass, sigma_mass, dlns_dlogM

def calculate_mass_function(cosmo: Cosmology,
                            mass: jnp.ndarray,
                            a: Union[float, jnp.ndarray],
                            hmfunc: Callable,
                            interpolator: Interpolator2D=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates the halo mass function.

    Args:
        cosmo (Cosmology): Cosmology object containing necessary parameters.
        mass (jnp.ndarray): Array of halo masses.
        a (Union[float, jnp.ndarray]): Scale factor(s).
        hmfunc (Callable): Halo mass function model.
        interpolator (Interpolator2D, optional): Precomputed interpolator for sigma. Defaults to None.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The computed mass function and the function values.
    """
    mass_use = jnp.atleast_1d(mass)
    log_mass, sigma_mass, dlns_dlogM = get_logM_sigM(cosmo, mass_use, a, interpolator)
    rho = CCLCST.RHO_CRITICAL * cosmo.h**2 * cosmo.Omega_m
    func = hmfunc.compute_fsigma(cosmo, sigma_mass, a, 2.302585092994046 * log_mass)
    mf = func * rho * dlns_dlogM / mass_use
    return mf, func