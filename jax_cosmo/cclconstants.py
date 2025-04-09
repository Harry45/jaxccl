import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

# Precision parameters
GSL_EPSREL = 1e-4  # Default relative precision
GSL_N_ITERATION = 1000  # Default number of iterations for integration and root-finding
GSL_INTEGRATION_GAUSS_KRONROD_POINTS = (
    41  # Default number of Gauss-Kronrod points in QAG integration
)
GSL_EPSREL_SIGMAR = 1e-5  # Relative precision in sigma_R calculations
GSL_EPSREL_KNL = 1e-5  # Relative precision in k_NL calculations
GSL_EPSREL_DIST = 1e-6  # Relative precision in distance calculations
GSL_EPSREL_GROWTH = 1e-6  # Relative precision in growth calculations
GSL_EPSREL_DNDZ = 1e-6  # Relative precision in dNdz calculations

# Define species labels
SPECIES_CRIT = "critical"
SPECIES_M = "matter"


@dataclass
class PhysicalConstants:
    """Defines physical constants based on CODATA 2022 and other sources."""

    SIDEREAL_YEAR_SEC: float = 365.256363004 * 86400.0  # Seconds in a sidereal year
    LIGHT_SPEED_H0_MPC: float = 2997.92458  # c/H0 in Mpc/h
    GRAVITATIONAL_CONSTANT: float = 6.67430e-11  # m^3/kg/s^2
    SOLAR_MASS_KG: float = 1.988409871e30  # kg
    MPC_TO_METERS: float = 3.085677581491367399198952281e22  # Mpc to meters
    RHO_CRITICAL: float = (
        (3 * 100**2)
        / (8 * jnp.pi * GRAVITATIONAL_CONSTANT)
        * ((1000 * 1000 * MPC_TO_METERS) / SOLAR_MASS_KG)
    )
    BOLTZMANN_CONSTANT: float = 1.380649e-23  # J/K
    STEFAN_BOLTZMANN_CONSTANT: float = 5.670374419e-8  # kg/s^3/K^4
    PLANCK_CONSTANT: float = 6.62607015e-34  # kg m^2 / s
    LIGHT_SPEED: float = 299792458.0  # m/s
    ELECTRON_VOLT_TO_JOULE: float = 1.602176634e-19  # eV to Joules
    NEUTRINO_SPLITTING_1: float = 7.62e-5
    NEUTRINO_SPLITTING_2: float = 2.55e-3
    NEUTRINO_SPLITTING_3: float = -2.43e-3
    T_CMB: float = 2.7255  # K
    T_NCDM: float = 0.71611  # K


@dataclass
class CCLSplineParams:
    # Scale factor spline parameters
    A_SPLINE_NA: int = 250
    A_SPLINE_MIN: float = 0.1
    A_SPLINE_MINLOG_PK: float = 0.01
    A_SPLINE_MIN_PK: float = 0.1
    A_SPLINE_MINLOG_SM: float = 0.01
    A_SPLINE_MIN_SM: float = 0.1
    A_SPLINE_MAX: float = 1.0
    A_SPLINE_MINLOG: float = 0.0001
    A_SPLINE_NLOG: int = 250

    # Mass splines
    LOGM_SPLINE_DELTA: float = 0.025
    LOGM_SPLINE_NM: int = 50
    LOGM_SPLINE_MIN: int = 6
    LOGM_SPLINE_MAX: int = 17

    # Power spectrum a and k splines
    A_SPLINE_NA_SM: int = 13
    A_SPLINE_NLOG_SM: int = 6
    A_SPLINE_NA_PK: int = 40
    A_SPLINE_NLOG_PK: int = 11

    # k-splines and integrals
    K_MAX_SPLINE: int = 50
    K_MAX: float = 1e3
    K_MIN: float = 5e-5
    DLOGK_INTEGRATION: float = 0.025
    DCHI_INTEGRATION: float = 5.0
    N_K: int = 167
    N_K_3DCOR: int = 100000

    # Correlation function parameters
    ELL_MIN_CORR: float = 0.01
    ELL_MAX_CORR: int = 60000
    N_ELL_CORR: int = 5000

    # Spline types (placeholders for now)
    spline1: Optional[None] = None
    spline2: Optional[None] = None
    spline3: Optional[None] = None
    spline4: Optional[None] = None
    spline5: Optional[None] = None
    spline6: Optional[None] = None
    spline7: Optional[None] = None


@dataclass
class CCLGSLParams:
    N_ITERATION: int = GSL_N_ITERATION
    INTEGRATION_GAUSS_KRONROD_POINTS: int = GSL_INTEGRATION_GAUSS_KRONROD_POINTS
    INTEGRATION_EPSREL: float = GSL_EPSREL
    INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS: int = GSL_INTEGRATION_GAUSS_KRONROD_POINTS
    INTEGRATION_LIMBER_EPSREL: float = GSL_EPSREL
    INTEGRATION_DISTANCE_EPSREL: float = GSL_EPSREL_DIST
    INTEGRATION_SIGMAR_EPSREL: float = GSL_EPSREL_SIGMAR
    INTEGRATION_KNL_EPSREL: float = GSL_EPSREL_KNL
    ROOT_EPSREL: float = GSL_EPSREL
    ROOT_N_ITERATION: int = GSL_N_ITERATION
    ODE_GROWTH_EPSREL: float = GSL_EPSREL_GROWTH
    EPS_SCALEFAC_GROWTH: float = 1e-6
    NZ_NORM_SPLINE_INTEGRATION: bool = True
    LENSING_KERNEL_SPLINE_INTEGRATION: bool = True
