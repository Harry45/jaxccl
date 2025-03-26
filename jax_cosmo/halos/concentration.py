import jax
import jax.numpy as jnp

from typing import Union, Optional, Callable
from jax_cosmo.core import Cosmology
from jax_cosmo.background import growth_factor, growth_rate
from jax_cosmo.power import dlogP_dlogk
from jax_cosmo.halos.hmbase import (get_delta_c,
                                    sigmaM,
                                    mass2radius_lagrangian,
                                    d_sigmaM_dlogM,
                                    parse_mass_def)
import optimistix as optx
from interpax import Interpolator2D


class JAXConBhattacharya13:
    """
    Implements the concentration-mass relation by Bhattacharya et al. (2013).

    This parametrization is valid only for spherical overdensity (S.O.) masses.
    By default, it is initialized for 200c.

    Attributes:
        name (str): Name of the concentration model.
        mass_def (str): Mass definition object or name string.
        A (float): Model parameter A.
        B (float): Model parameter B.
        C (float): Model parameter C.

    Args:
        mass_def (str, optional): Mass definition string. Defaults to "200c".
    """
    name: str = "Bhattacharya13"

    def __init__(self, mass_def: str = "200c") -> None:
        self.mass_def: str = mass_def

        vals = {
            "vir": (7.7, 0.9, -0.29),
            "200m": (9.0, 1.15, -0.29),
            "200c": (5.9, 0.54, -0.35),
        }

        if self.mass_def not in vals:
            raise ValueError(f"Invalid mass definition: {self.mass_def}")

        self.A, self.B, self.C = vals[self.mass_def]

    def compute_concentration(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Interpolator2D
    ) -> jnp.ndarray:
        """
        Computes the concentration parameter given a cosmology and mass of halo.

        Args:
            cosmology (Cosmology): Cosmological model instance.
            mass (Union[float, jnp.ndarray]): Halo mass.
            scale_factor (float): Scale factor at which to compute concentration.
            interpolator (Interpolator2D): Interpolator for sigmaM.

        Returns:
            jnp.ndarray: Computed concentration parameter.
        """
        gz = growth_factor(cosmology, jnp.atleast_1d(scale_factor))
        delta_c = get_delta_c(cosmology, scale_factor, method="NakamuraSuto97")
        log_mass = jnp.log10(mass)
        sigma_mass = sigmaM(log_mass, scale_factor, interpolator)
        nu = delta_c / sigma_mass

        return self.A * gz**self.B * nu**self.C

class JAXConConstant:
    """
    Constant concentration-mass relation.

    This model assumes a fixed concentration value, independent of mass or redshift.

    Attributes:
        name (str): Name of the concentration model.
        c (float): Constant concentration value.
        mass_def (str): Mass definition (arbitrary in this case).

    Args:
        c (float, optional): Constant concentration value. Defaults to 1.
        mass_def (str, optional): Mass definition. Defaults to "200c".
    """
    name: str = "Constant"

    def __init__(self, c: float = 1.0, mass_def: str = "200c") -> None:
        self.c: float = c
        self.mass_def: str = mass_def

    def compute_concentration(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Optional[Interpolator2D] = None
    ) -> jnp.ndarray:
        """
        Computes the concentration parameter as a constant value.

        Args:
            cosmology (Cosmology): Cosmological model instance.
            mass (Union[float, jnp.ndarray]): Halo mass.
            scale_factor (float): Scale factor at which to compute concentration.
            interpolator (Optional[Interpolator2D], optional): Interpolator for sigmaM.
                Defaults to None.

        Returns:
            jnp.ndarray: Array of constant concentration values matching input mass shape.
        """
        return jnp.full_like(mass, self.c)

class JAXConDiemer15:
    """
    Concentration-mass relation by Diemer & Kravtsov (2015).

    This parametrization is only valid for spherical overdensity (S.O.) masses
    with Delta = 200 times the critical density.

    Attributes:
        name (str): Name of the concentration model.
        mass_def (Union[MassDef, str]): Mass definition object or name string.
        kappa (float): Wavenumber factor.
        phi_0 (float): Base phi parameter.
        phi_1 (float): Secondary phi parameter.
        eta_0 (float): Base eta parameter.
        eta_1 (float): Secondary eta parameter.
        alpha (float): Exponent for low-nu regime.
        beta (float): Exponent for high-nu regime.

    Args:
        mass_def (str): Mass definition object or string. Defaults to "200c".
    """
    name: str = "Diemer15"

    def __init__(self, mass_def: str = "200c") -> None:
        self.mass_def: str = mass_def
        self.kappa: float = 1.0
        self.phi_0: float = 6.58
        self.phi_1: float = 1.27
        self.eta_0: float = 7.28
        self.eta_1: float = 1.56
        self.alpha: float = 1.08
        self.beta: float = 1.77

    def compute_concentration(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Interpolator2D
    ) -> jnp.ndarray:
        """
        Computes the concentration parameter given a cosmology and halo mass.

        Args:
            cosmology (Cosmology): Cosmological model instance.
            mass (Union[float, jnp.ndarray]): Halo mass.
            scale_factor (float): Scale factor at which to compute concentration.
            interpolator (Interpolator2D): Interpolator for sigmaM.

        Returns:
            jnp.ndarray: Computed concentration parameter.
        """
        radius = mass2radius_lagrangian(cosmology, mass)
        k_radius = 2.0 * jnp.pi / radius * self.kappa
        logP_der = dlogP_dlogk(cosmology, k_radius/cosmology.h, scale_factor)

        delta_c = get_delta_c(cosmology, scale_factor, method="EdS")
        log_mass = jnp.log10(mass)
        sigma_mass = sigmaM(log_mass, scale_factor, interpolator)
        nu = delta_c / sigma_mass

        floor = self.phi_0 + logP_der * self.phi_1
        nu0 = self.eta_0 + logP_der * self.eta_1
        conc = 0.5 * floor * ((nu0 / nu)**self.alpha + (nu / nu0)**self.beta)
        return conc

class JAXConDuffy08:
    """
    Implements the concentration-mass relation from Duffy et al. (2008).

    This parametrization is valid only for spherical overdensity (S.O.) masses.
    By default, it is initialized for 200c.

    Attributes:
        name (str): Name of the concentration model.
        mass_def (str): Mass definition.
        A (float): Model parameter A.
        B (float): Model parameter B.
        C (float): Model parameter C.

    Args:
        mass_def (str, optional): Mass definition, must be one of {"vir", "200m", "200c"}.
            Defaults to "200c".

    Raises:
        ValueError: If an invalid mass definition is provided.
    """

    name: str = "Duffy08"

    def __init__(self, mass_def: str = "200c") -> None:
        self.mass_def = mass_def
        param_map = {
            "vir": (7.85, -0.081, -0.71),
            "200m": (10.14, -0.081, -1.01),
            "200c": (5.71, -0.084, -0.47),
        }

        if mass_def not in param_map:
            raise ValueError(f"Invalid mass definition '{mass_def}'. Must be one of {list(param_map.keys())}.")

        self.A, self.B, self.C = param_map[mass_def]

    def compute_concentration(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Optional[Interpolator2D] = None,
    ) -> jnp.ndarray:
        """
        Computes the halo concentration parameter for a given cosmology and mass.

        Args:
            cosmology (Cosmology): Cosmological model instance.
            mass (Union[float, jnp.ndarray]): Halo mass (in solar masses).
            scale_factor (float): Scale factor (1 / (1 + z)).
            interpolator (Optional[Interpolator2D]): Interpolator for sigmaM (not used in this model).

        Returns:
            jnp.ndarray: Computed concentration parameter.
        """
        M_pivot_inv = cosmology.h * 5e-13  # Convert pivot mass to appropriate units
        return self.A * (mass * M_pivot_inv) ** self.B * scale_factor ** (-self.C)

class JAXConKlypin11:
    """
    Implements the concentration-mass relation from Klypin et al. (2011).

    This parametrization is valid only for spherical overdensity (S.O.) masses.

    Attributes:
        name (str): Name of the concentration model.
        mass_def (str): Mass definition.

    Args:
        mass_def (str, optional): Mass definition, must be "vir". Defaults to "vir".

    Raises:
        ValueError: If an unsupported mass definition is provided.
    """
    name = 'Klypin11'

    def __init__(self, mass_def="vir"):
        if mass_def != "vir":
            raise ValueError(f"Invalid mass definition '{mass_def}'. Only 'vir' is supported.")

        self.mass_def = mass_def

    def compute_concentration(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Optional[Interpolator2D] = None,
    ) -> jnp.ndarray:
        """
        Computes the halo concentration parameter for a given cosmology and mass.

        Args:
            cosmology (Cosmology): Cosmological model instance.
            mass (Union[float, jnp.ndarray]): Halo mass (in solar masses).
            scale_factor (float): Scale factor (1 / (1 + z)).
            interpolator (Optional[Interpolator2D]): Interpolator for sigmaM (not used in this model).

        Returns:
            jnp.ndarray: Computed concentration parameter.
        """
        M_pivot_inv = cosmology["h"] * 1E-12
        return 9.6 * (mass * M_pivot_inv) ** -0.075


class JAXConPrada12:
    """Concentration-mass relation from Prada et al. 2012 for halo dynamics.

    This class implements the concentration-mass parametrization specifically
    for spherical overdensity (S.O.) masses with Δ = 200 times the critical density.

    The implementation follows the concentration-mass relation methodology
    proposed in the original research paper (https://arxiv.org/abs/1104.5130).

    Attributes:
        name (str): Identifier for this concentration-mass relation model.
        mass_def (str): Mass definition used for concentration calculation.
        c0 (float): First concentration parameter.
        c1 (float): Second concentration parameter.
        al (float): Alpha parameter for concentration calculation.
        x0 (float): First scaling parameter.
        i0 (float): First intensity parameter.
        i1 (float): Second intensity parameter.
        be (float): Beta parameter for concentration calculation.
        x1 (float): Second scaling parameter.
        cnorm (float): Normalization factor for concentration.
        inorm (float): Normalization factor for intensity.

    Example:
        >>> model = JAXConPrada12(mass_def="200c")
    """

    name: str = 'Prada12'

    def __init__(self, *, mass_def: str = "200c") -> None:
        """Initialize the Prada12 concentration-mass relation model.

        Args:
            mass_def: Mass definition to use for concentration calculation.
                Defaults to "200c" (200 times critical density).
        """
        self.mass_def = mass_def

        # Model parameters
        self.c0: float = 3.681
        self.c1: float = 5.033
        self.al: float = 6.948
        self.x0: float = 0.424
        self.i0: float = 1.047
        self.i1: float = 1.646
        self.be: float = 7.386
        self.x1: float = 0.526

        # Compute normalization factors
        self.cnorm: float = 1.0 / self._cmin(
            x=1.393,
            x0=self.x0,
            v0=self.c0,
            v1=self.c1,
            v2=self.al
        )

        self.inorm: float = 1.0 / self._cmin(
            x=1.393,
            x0=self.x1,
            v0=self.i0,
            v1=self.i1,
            v2=self.be
        )

    def _cmin(self, x: float,
              x0: float,
              v0: float,
              v1: float,
              v2: float) -> float:
        """Compute the form factor for concentration and intensity.

        Args:
            x: Input value for calculation.
            x0: Scaling parameter.
            v0: First value parameter.
            v1: Second value parameter.
            v2: Scaling coefficient.

        Returns:
            Computed form factor value.
        """
        return v0 + (v1 - v0) * (jnp.arctan(v2 * (x - x0)) / jnp.pi + 0.5)

    def compute_concentration(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Interpolator2D
    ) -> jnp.ndarray:
        """
        Computes the concentration parameter given a cosmology and halo mass.

        Args:
            cosmology (Cosmology): Cosmological model instance.
            mass (Union[float, jnp.ndarray]): Halo mass.
            scale_factor (float): Scale factor at which to compute concentration.
            interpolator (Interpolator2D): Interpolator for sigmaM.

        Returns:
            jnp.ndarray: Computed concentration parameter.
        """
        log_mass = jnp.log10(mass)
        sigma_mass = sigmaM(log_mass, scale_factor, interpolator)
        x = scale_factor * (cosmology.Omega_de / cosmology.Omega_m)**(1. / 3.)
        b_factor_0 = self._cmin(x, self.x0, self.c0, self.c1, self.al) * self.cnorm
        b_factor_1 = self._cmin(x, self.x1, self.i0, self.i1, self.be) * self.inorm
        sig_p = b_factor_1 * sigma_mass
        conc = 2.881 * ((sig_p / 1.257)**1.022 + 1) * jnp.exp(0.060 / sig_p**2)
        return b_factor_0 * conc

# class JAXConIshiyama21:
#     """Concentration-mass relation by Ishiyama et al. (2021).

#     This parametrization is valid for S.O. masses. By default, it initializes for 500c.

#     Args:
#         mass_def (str): A mass definition object or a string.
#         relaxed (bool, optional): If True, use concentration for relaxed halos. Default is False.
#         Vmax (bool, optional): If True, use the concentration found with the "Vmax" method.
#                                Otherwise, use concentration from profile fitting. Default is False.
#     """

#     name = "Ishiyama21"

#     _vals = {
#         (True, True, 200): (1.79, 2.15, 2.06, 0.88, 9.24, 0.51),
#         (True, False, 200): (1.10, 2.30, 1.64, 1.72, 3.60, 0.32),
#         (False, True, 200): (0.60, 2.14, 2.63, 1.69, 6.36, 0.37),
#         (False, False, 200): (1.19, 2.54, 1.33, 4.04, 1.21, 0.22),
#         (True, True, "vir"): (2.40, 2.27, 1.80, 0.56, 13.24, 0.079),
#         (True, False, "vir"): (0.76, 2.34, 1.82, 1.83, 3.52, -0.18),
#         (False, True, "vir"): (1.22, 2.52, 1.87, 2.13, 4.19, -0.017),
#         (False, False, "vir"): (1.64, 2.67, 1.23, 3.92, 1.30, -0.19),
#         (False, True, 500): (0.38, 1.44, 3.41, 2.86, 2.99, 0.42),
#         (False, False, 500): (1.83, 1.95, 1.17, 3.57, 0.91, 0.26),
#     }

#     def __init__(self, mass_def: str = "500c", relaxed: bool = False, Vmax: bool = False):
#         self.relaxed = relaxed
#         self.Vmax = Vmax
#         self.mass_def = parse_mass_def(mass_def)

#         # Load coefficients from dictionary lookup
#         key = (self.Vmax, self.relaxed, self.mass_def)
#         self.kappa, self.a0, self.a1, self.b0, self.b1, self.c_alpha = self._vals[key]

#     def _dlsigmaR(self, cosmology: Cosmology, mass: Union[float, jnp.ndarray],
#                   scale_factor: float, interpolator: Interpolator2D) -> jnp.ndarray:
#         """Computes the logarithmic derivative of sigma(M) w.r.t. mass."""
#         log_mass = 3.0 * jnp.log10(self.kappa) + jnp.log10(mass)
#         dlns_dlogM = d_sigmaM_dlogM(log_mass, scale_factor, interpolator)
#         return -3.0 / jnp.log(10.0) * dlns_dlogM

#     def G_quant(self, x: jnp.ndarray, n_eff: jnp.ndarray) -> jnp.ndarray:
#         """Computes the G(x) function used in concentration calculations."""
#         f_quant = jnp.log(1.0 + x) - x / (1.0 + x)
#         return x / f_quant ** ((5.0 + n_eff) / 6.0)

#     def solve_single(self, params, arguments):
#         """Function whose root we seek. Arguments = (val, n_eff)."""
#         val, n_eff = arguments
#         return self.G_quant(params, n_eff) - val

#     def optimisation(self, val: jnp.ndarray, n_eff: jnp.ndarray) -> jnp.ndarray:
#         solver = optx.Newton(rtol=1e-8, atol=1e-8)
#         sol = optx.root_find(self.solve_single, solver, y0=jnp.ones_like(val), args=(val, n_eff))
#         G_inv = sol.value
#         return G_inv

#     def compute_concentration(
#         self,
#         cosmology: Cosmology,
#         mass: Union[float, jnp.ndarray],
#         scale_factor: float,
#         interpolator: Interpolator2D
#     ) -> jnp.ndarray:
#         """
#         Computes the concentration parameter given a cosmology and halo mass.

#         Args:
#             cosmology (Cosmology): Cosmological model instance.
#             mass (Union[float, jnp.ndarray]): Halo mass.
#             scale_factor (float): Scale factor at which to compute concentration.
#             interpolator (Interpolator2D): Interpolator for sigmaM.

#         Returns:
#             jnp.ndarray: Computed concentration parameter.
#         """
#         delta_c = get_delta_c(cosmology, scale_factor, method='EdS_approx')
#         log_mass = jnp.log10(mass)
#         sigma_mass = sigmaM(log_mass, scale_factor, interpolator)
#         nu = delta_c / sigma_mass

#         n_eff = -2 * self._dlsigmaR(cosmology, mass, scale_factor, interpolator) - 3
#         alpha_eff = growth_rate(cosmology, jnp.atleast_1d(scale_factor))

#         A_factor = self.a0 * (1 + self.a1 * (n_eff + 3))
#         B_factor = self.b0 * (1 + self.b1 * (n_eff + 3))
#         C_factor = 1 - self.c_alpha * (1 - alpha_eff)

#         val = A_factor / nu * (1 + nu**2 / B_factor)

#         # run the optimisation
#         G_inv = self.optimisation(val, n_eff)

#         return C_factor * G_inv


class JAXConIshiyama21:
    """Concentration-mass relation by Ishiyama et al. (2021).

    This parametrization is valid for S.O. masses. By default, it initializes for `500c`.

    Attributes:
        mass_def (str): A mass definition object or a string.
        relaxed (bool): If True, uses concentration for relaxed halos.
        Vmax (bool): If True, uses concentration from the "Vmax" method.
        kappa (float): Model parameter extracted from `_vals` lookup.
        a0 (float): Model parameter extracted from `_vals` lookup.
        a1 (float): Model parameter extracted from `_vals` lookup.
        b0 (float): Model parameter extracted from `_vals` lookup.
        b1 (float): Model parameter extracted from `_vals` lookup.
        c_alpha (float): Model parameter extracted from `_vals` lookup.
    """

    name = "Ishiyama21"

    _vals = {
        (True, True, 200): (1.79, 2.15, 2.06, 0.88, 9.24, 0.51),
        (True, False, 200): (1.10, 2.30, 1.64, 1.72, 3.60, 0.32),
        (False, True, 200): (0.60, 2.14, 2.63, 1.69, 6.36, 0.37),
        (False, False, 200): (1.19, 2.54, 1.33, 4.04, 1.21, 0.22),
        (True, True, "vir"): (2.40, 2.27, 1.80, 0.56, 13.24, 0.079),
        (True, False, "vir"): (0.76, 2.34, 1.82, 1.83, 3.52, -0.18),
        (False, True, "vir"): (1.22, 2.52, 1.87, 2.13, 4.19, -0.017),
        (False, False, "vir"): (1.64, 2.67, 1.23, 3.92, 1.30, -0.19),
        (False, True, 500): (0.38, 1.44, 3.41, 2.86, 2.99, 0.42),
        (False, False, 500): (1.83, 1.95, 1.17, 3.57, 0.91, 0.26),
    }

    def __init__(self, mass_def: str = "500c", relaxed: bool = False, Vmax: bool = False):
        """Initializes the concentration-mass relation model.

        Args:
            mass_def (str, optional): Mass definition string. Default is "500c".
            relaxed (bool, optional): If True, uses relaxed halo concentration. Default is False.
            Vmax (bool, optional): If True, uses the Vmax method instead of profile fitting. Default is False.
        """
        self.relaxed = relaxed
        self.Vmax = Vmax
        self.mass_def = parse_mass_def(mass_def)

        # Load coefficients from dictionary lookup
        key = (self.Vmax, self.relaxed, self.mass_def)
        self.kappa, self.a0, self.a1, self.b0, self.b1, self.c_alpha = self._vals[key]

    def _dlsigmaR(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Interpolator2D
    ) -> jnp.ndarray:
        """Computes the logarithmic derivative of sigma(M) with respect to mass.

        Args:
            cosmology (Cosmology): Cosmology model.
            mass (Union[float, jnp.ndarray]): Halo mass.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for sigma(M).

        Returns:
            jnp.ndarray: Logarithmic derivative of sigma(M).
        """
        log_mass = 3.0 * jnp.log10(self.kappa) + jnp.log10(mass)
        dlns_dlogM = d_sigmaM_dlogM(log_mass, scale_factor, interpolator)
        return -3.0 / jnp.log(10.0) * dlns_dlogM

    def G_quant(self, x: jnp.ndarray, n_eff: jnp.ndarray) -> jnp.ndarray:
        """Computes the G(x) function used in concentration calculations.

        Args:
            x (jnp.ndarray): Input variable.
            n_eff (jnp.ndarray): Effective spectral index.

        Returns:
            jnp.ndarray: Computed G(x) values.
        """
        f_quant = jnp.log(1.0 + x) - x / (1.0 + x)
        return x / f_quant ** ((5.0 + n_eff) / 6.0)

    def solve_single(self, params: jnp.ndarray, arguments: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Computes the residual function for root-finding.

        Args:
            params (jnp.ndarray): Parameter to optimize.
            arguments (tuple[jnp.ndarray, jnp.ndarray]): Tuple (val, n_eff).

        Returns:
            jnp.ndarray: Difference between G_quant and target value.
        """
        val, n_eff = arguments
        return self.G_quant(params, n_eff) - val

    def optimisation(self, val: jnp.ndarray, n_eff: jnp.ndarray) -> jnp.ndarray:
        """Solves the root-finding problem for concentration using Newton's method.

        Args:
            val (jnp.ndarray): Target values.
            n_eff (jnp.ndarray): Effective spectral indices.

        Returns:
            jnp.ndarray: Optimized values (G_inv).
        """
        solver = optx.Newton(rtol=1e-8, atol=1e-8)

        # Vectorized root-finding
        G_inv = jax.vmap(lambda v, n: optx.root_find(self.solve_single,
                                                     solver,
                                                     y0=jnp.ones_like(v),
                                                     args=(v, n), throw=False).value)(val, n_eff)
        return G_inv

    def compute_concentration(
        self,
        cosmology: Cosmology,
        mass: Union[float, jnp.ndarray],
        scale_factor: float,
        interpolator: Interpolator2D
    ) -> jnp.ndarray:
        """Computes the concentration parameter given a cosmology and halo mass.

        Args:
            cosmology (Cosmology): Cosmological model instance.
            mass (Union[float, jnp.ndarray]): Halo mass.
            scale_factor (float): Scale factor at which to compute concentration.
            interpolator (Interpolator2D): Interpolator for sigmaM.

        Returns:
            jnp.ndarray: Computed concentration parameter.
        """
        delta_c = get_delta_c(cosmology, scale_factor, method="EdS_approx")
        log_mass = jnp.log10(mass)
        sigma_mass = sigmaM(log_mass, scale_factor, interpolator)
        nu = delta_c / sigma_mass

        n_eff = -2 * self._dlsigmaR(cosmology, mass, scale_factor, interpolator) - 3
        alpha_eff = growth_rate(cosmology, jnp.atleast_1d(scale_factor))

        A_factor = self.a0 * (1 + self.a1 * (n_eff + 3))
        B_factor = self.b0 * (1 + self.b1 * (n_eff + 3))
        C_factor = 1 - self.c_alpha * (1 - alpha_eff)

        val = A_factor / nu * (1 + nu**2 / B_factor)

        # Run the optimization
        G_inv = self.optimisation(val, n_eff)

        return C_factor * G_inv