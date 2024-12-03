# This module defines kernel functions for various tracers
from typing import Tuple, List, Union, Optional
from jax.typing import ArrayLike
import jax.numpy as np
from jax import jit
from jax import vmap
from jax.tree_util import register_pytree_node_class

import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const
import jax_cosmo.redshift as rds
from jax_cosmo.jax_utils import container
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a
from jax_cosmo.core import Cosmology
from jax_cosmo.redshift import redshift_distribution
import jax_cosmo.cclconstants as cst

__all__ = ["WeakLensing",
           "NumberCounts",
           "CIBTracer",
           "CMBLensingTracer",
           "zPowerTracer",
           "tSZTracer"]


@jit
def weak_lensing_kernel(
    cosmo: Cosmology,
    pzs: List[redshift_distribution],
    z: ArrayLike,
    ell: float
) -> np.ndarray:
    """
    Computes the weak lensing kernel.

    This function calculates the weak lensing kernel for a given cosmology,
    redshift distributions (either extended or delta functions), redshift values,
    and multipole moment `ell`. It separates the input redshift distributions
    into extended distributions and delta functions to process them accordingly.

    Args:
        cosmo (Cosmology): The cosmological parameters and background model.
        pzs (List[RedshiftDistribution]): List of redshift distributions (can include delta distributions).
        z (ArrayLike): The redshift values for which the kernel is calculated.
        ell (float): The multipole moment for weak lensing.

    Returns:
        The computed weak lensing kernel for the provided redshifts and `ell`.

    Notes:
        - The function treats extended redshift distributions and delta functions differently.
        - Assumes redshift distributions implement `__call__` for dndz evaluation.
    """
    z = np.atleast_1d(z)
    zmax = max([pz.zmax for pz in pzs])
    # Retrieve comoving distance corresponding to z
    chi = bkgrd.radial_comoving_distance(cosmo, z2a(z))

    # Extract the indices of pzs that can be treated as extended distributions,
    # and the ones that need to be treated as delta functions.
    pzs_extended_idx = [
        i for i, pz in enumerate(pzs) if not isinstance(pz, rds.delta_nz)
    ]
    pzs_delta_idx = [i for i, pz in enumerate(pzs) if isinstance(pz, rds.delta_nz)]
    # Here we define a permutation that would put all extended pzs at the begining of the list
    perm = pzs_extended_idx + pzs_delta_idx
    # Compute inverse permutation
    inv = np.argsort(np.array(perm, dtype=np.int32))

    # Process extended distributions, if any
    radial_kernels = []
    if len(pzs_extended_idx) > 0:

        @vmap
        def integrand(z_prime):
            chi_prime = bkgrd.radial_comoving_distance(cosmo, z2a(z_prime))
            # Stack the dndz of all redshift bins
            dndz = np.stack([pzs[i](z_prime) for i in pzs_extended_idx], axis=0)
            return dndz * np.clip(chi_prime - chi, 0) / np.clip(chi_prime, 1.0)

        radial_kernels.append(simps(integrand, z, zmax, 256) * (1.0 + z) * chi)
    # Process single plane redshifts if any
    if len(pzs_delta_idx) > 0:

        @vmap
        def integrand_single(z_prime):
            chi_prime = bkgrd.radial_comoving_distance(cosmo, z2a(z_prime))
            return np.clip(chi_prime - chi, 0) / np.clip(chi_prime, 1.0)

        radial_kernels.append(
            integrand_single(np.array([pzs[i].params[0] for i in pzs_delta_idx]))
            * (1.0 + z)
            * chi
        )
    # Fusing the results together
    radial_kernel = np.concatenate(radial_kernels, axis=0)
    # And perfoming inverse permutation to put all the indices where they should be
    radial_kernel = radial_kernel[inv]

    # Constant term
    constant_factor = 3.0 * const.H0**2 * cosmo.Omega_m / 2.0 / const.c
    # Ell dependent factor
    ell_factor = np.sqrt((ell - 1) * (ell) * (ell + 1) * (ell + 2)) / (ell + 0.5) ** 2
    return constant_factor * ell_factor * radial_kernel


@jit
def density_kernel(
    cosmo: Cosmology,
    pzs: List[redshift_distribution],
    bias: List[float],
    z: np.ndarray,
    ell: np.ndarray
) -> np.ndarray:
    r"""
    Computes the number counts density kernel for a given cosmology, redshift bins,
    bias, redshift values, and multipole moment.

    Args:
        cosmo (Cosmology): The cosmology object containing cosmological parameters.

        pzs (List[redshift_distribution]): A list of redshift distribution functions (or objects) for each redshift bin. Each element in the list should be callable and take a redshift `z` as input.

        bias (List[float]): A list of bias functions (or constants) for each redshift bin. If a list of functions is provided, each function should accept `cosmo` and `z` as arguments and return the bias at a given redshift.

        z (np.ndarray): A 1D array of redshift values for which the density kernel will be computed.

        ell (np.ndarray): A 1D array of multipole moments for the kernel computation.

    Returns:
        np.ndarray: The computed density kernel, with shape `(nbins, nz)` where `nbins` is the number of redshift bins and `nz` is the number of redshift values provided in `z`.

    Raises:
        NotImplementedError: If any of the redshift distributions are of type `rds.delta_nz`, which is not supported.

    Notes:
        The density kernel is computed as the product of the redshift distribution `dndz`,
        the bias function, and the background Hubble parameter at each redshift value.
        The resulting kernel is then multiplied by normalization and `ell`-dependent factors,
        though both factors are currently set to `1.0`.
    """
    if any(isinstance(pz, rds.delta_nz) for pz in pzs):
        raise NotImplementedError(
            "Density kernel not properly implemented for delta redshift distributions"
        )
    # stack the dndz of all redshift bins
    dndz = np.stack([pz(z) for pz in pzs], axis=0)
    # Compute radial NLA kernel: same as clustering
    if isinstance(bias, list):
        # This is to handle the case where we get a bin-dependent bias
        b = np.stack([b(cosmo, z) for b in bias], axis=0)
    else:
        b = bias(cosmo, z)
    radial_kernel = dndz * b * bkgrd.H(cosmo, z2a(z))
    # Normalization,
    constant_factor = 1.0
    # Ell dependent factor
    ell_factor = 1.0
    return constant_factor * ell_factor * radial_kernel

def kappa_kernel(
    cosmo: Cosmology, z_source: float, n_samples: int = 100, ell: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """This convenience function returns the radial kernel for CMB-lensing-like tracers.

    Args:
        cosmo (Cosmology): the cosmology object with all cosmological parameters.
        z_source (float): the redshift of the source
        n_samples (int, optional): the number of samples of comoving radial distance. Defaults to 100.
        ell (int, optional): the ell mode. Defaults to 2000.

    Returns:
        the kernel and the comoving radial distance
    """
    # get the comoving radial distance of the source
    chi_source = bkgrd.radial_comoving_distance(cosmo, z2a(z_source))

    # get the scale factor and the comoving radial distance
    a_arr, chi_arr = bkgrd.scale_of_chi(cosmo, 0.0, z_source, n_samples)

    # calculate the radial kernel
    radial_kernel = (1.0 / a_arr) * chi_arr * (chi_source - chi_arr) / chi_source

    # Constant term (note that I have edited this prefactor so it agrees with CCL implementation, essentially multiplying by h^2 / c)
    constant_factor = (3.0 * const.H0**2 * cosmo.h**2 * cosmo.Omega_m) / (
        2.0 * const.c**2
    )

    # prefactor containing the ell terms
    f_ell = ell * (ell + 1) / (ell + 0.5) ** 2

    # the final kernel
    w_arr = constant_factor * f_ell * radial_kernel

    return w_arr, chi_arr

@jit
def nla_kernel(
    cosmo: Cosmology,
    pzs: List,
    bias: List,
    z: np.ndarray,
    ell: int
) -> np.ndarray:
    r"""Computes the Non-Linear Alignment (NLA) Intrinsic Alignment (IA) kernel.

    The NLA kernel is used in cosmic shear and intrinsic alignment studies to model
    the non-linear alignment of galaxy intrinsic alignments. This implementation
    follows the formulation in [Joachimi et al. (2011)](https://arxiv.org/abs/1008.3491). See Equation 6.

    Args:
        cosmo (Cosmology): Cosmological parameters object.
        pzs (List): List of photometric redshift distribution functions.
        bias (List): Bias function or list of bias functions.
        z (np.ndarray): Redshift values to evaluate the kernel.
        ell (int): Multipole moment for angular power spectrum calculation.

    Returns:
        Computed NLA kernel values.

    Raises:
        NotImplementedError: If delta redshift distributions are used.
    """
    # Check for delta redshift distributions
    if any(isinstance(pz, rds.delta_nz) for pz in pzs):
        raise NotImplementedError(
            "NLA kernel not properly implemented for delta redshift distributions"
        )

    # Stack the dndz of all redshift bins
    dndz = np.stack([pz(z) for pz in pzs], axis=0)

    # Compute radial NLA kernel: same as clustering
    if isinstance(bias, list):
        # Handle bin-dependent bias
        b = np.stack([b(cosmo, z) for b in bias], axis=0)
    else:
        b = bias(cosmo, z)

    # Compute radial kernel components
    radial_kernel = dndz * b * bkgrd.H(cosmo, z2a(z))

    # Apply A_IA normalization to the kernel
    radial_kernel *= (
        -(5e-14 * const.rhocrit) * cosmo.Omega_m / bkgrd.growth_factor(cosmo, z2a(z))
    )

    # Constant and ell-dependent factors
    constant_factor = 1.0
    ell_factor = np.sqrt((ell - 1) * (ell) * (ell + 1) * (ell + 2)) / (ell + 0.5) ** 2

    return constant_factor * ell_factor * radial_kernel

def power_kernel(
    cosmo: Cosmology, amplitude: float, alpha: float, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Tracer obeying a power law alpha with an amplitude.

    $$
    W(\chi) = A\chi^{\alpha}
    $$

    Args:
        cosmo (Cosmology): the cosmology object in JAXCOSMO
        amplitude (float): the amplitude of the kernel
        alpha (float): the power of the kernel

    Returns:
        The kernel and the comoving radial distance
    """
    z_min = kwargs["z_min"]
    z_max = kwargs["z_max"]
    n_z = kwargs["n_z"]
    a_arr, chi_arr = bkgrd.scale_of_chi(cosmo, z_min, z_max, n_z)
    w_arr = amplitude * a_arr**alpha
    return w_arr, chi_arr


def isw_kernel(cosmo: Cosmology, z_max: float, n_z: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the kernel for the Integrated Sachs-Wolfe (ISW) effect.

    The ISW effect contributes to temperature fluctuations in the Cosmic Microwave Background (CMB)
    anisotropies due to the time evolution of the gravitational potential in a matter-dominated
    universe. The contribution to the CMB temperature is given by:

    $$
    \Delta T_{\textrm CMB} = 2T_{\textrm CMB} \int_0^{\chi_{LSS}}d\chi a\,\dot{\phi}
    $$

    This function calculates the kernel required to compute angular power spectra involving
    the ISW effect. Note that any angular power spectra computed with this tracer should involve
    a three-dimensional power spectrum using the matter power spectrum.

    Args:
        cosmo (Cosmology): A cosmology object containing relevant parameters for calculations.
        z_max (float): The maximum redshift to compute the kernel up to.
        n_z (int): The number of redshift points for the kernel computation.

    Returns:
        The ISW kernel values as an array.
        The comoving radial distance corresponding to the redshift points.

    Raises:
        ValueError: If `z_max` or `n_z` are not valid positive numbers.
    """
    # Compute the scale factor and comoving distance arrays
    a_arr, chi = bkgrd.scale_of_chi(cosmo, 0.0, z_max, n_z)

    # Extract cosmological parameters
    H0 = cosmo.h / cst.CLIGHT_HMPC  # Hubble constant in h/Mpc
    OM = cosmo.Omega_m  # Matter density parameter

    # Calculate background functions
    Ez = bkgrd.H(cosmo, a_arr) / 100  # Hubble parameter normalized to 100 km/s/Mpc
    fz = bkgrd.growth_rate(cosmo, a_arr)  # Growth rate of matter fluctuations

    # Compute the ISW kernel
    w_arr = 3 * cst.T_CMB * H0**3 * OM * Ez * chi**2 * (1 - fz)

    return w_arr, chi

class ISWTracer(container):
    """
    Class representing the tracer associated with the Integrated Sachs-Wolfe (ISW) effect.

    The ISW effect describes the temperature fluctuations in the Cosmic Microwave Background (CMB)
    due to the time evolution of gravitational potentials, which primarily occurs at late times in
    a universe with dark energy or curvature.

    Attributes:
        z_max (float, optional): The maximum redshift up to which the kernel is computed. Defaults to 6.0.
        n_z (int, optional): The number of redshift points for the computation. Defaults to 1024.
    """

    def __init__(self, z_max: float = 6.0, n_z: int = 1024):
        super(ISWTracer, self).__init__(z_max, n_z)

    @property
    def zmax(self) -> float:
        """
        Returns the maximum redshift probed by this tracer.

        Returns:
            Maximum redshift.
        """
        return self.params[0]

    @property
    def n_z(self) -> int:
        """
        Returns the number of redshift points used in this tracer.

        Returns:
            Number of redshift points.
        """
        return self.params[1]

    def kernel(self, cosmo: Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the ISW kernel and comoving radial distances.

        The kernel represents the contribution of the ISW effect across different
        redshifts and distances.

        Args:
            cosmo (Cosmology): The cosmology object providing the cosmological parameters.

        Returns:
            The ISW kernel values as an array.
            The comoving radial distances corresponding to the redshift points.
        """
        inputs = {
            "z_max": self.params[0],
            "n_z": self.params[1],
        }
        return isw_kernel(cosmo, **inputs)

@register_pytree_node_class
class WeakLensing(container):
    """
    Represents a weak lensing probe with a set of redshift bins.

    This class models weak lensing observations, including optional intrinsic alignments (IA)
    and multiplicative biases.

    Attributes:
        redshift_bins (List): A list of redshift distributions for each redshift bin.
        ia_bias (Optional[Union[float, List[float]]]): If provided, intrinsic alignments (IA) are added
            using the NLA model. Can be a single value or a list matching the number of redshift bins.
        multiplicative_bias (Union[float, List[float]]): Adds a (1+m) multiplicative bias. Can be a single
            value or a list matching the number of redshift bins.
        sigma_e (float): The intrinsic ellipticity of galaxies. Defaults to 0.26.
        config (dict): Configuration settings, including flags such as `ia_enabled`.
    """

    def __init__(
        self,
        redshift_bins: List,
        ia_bias: Optional[Union[float, List[float]]] = None,
        multiplicative_bias: Union[float, List[float]] = 0.0,
        sigma_e: float = 0.26,
        **kwargs,
    ):

        if ia_bias is None:
            ia_enabled = False
            args = (redshift_bins, multiplicative_bias)
        else:
            ia_enabled = True
            args = (redshift_bins, multiplicative_bias, ia_bias)

        if "ia_enabled" not in kwargs:
            kwargs["ia_enabled"] = ia_enabled

        super(WeakLensing, self).__init__(*args, sigma_e=sigma_e, **kwargs)

    @property
    def n_tracers(self) -> int:
        """
        Returns the number of tracers (redshift bins) for this probe.

        Returns:
            The number of redshift bins.
        """
        pzs = self.params[0]
        return len(pzs)

    @property
    def zmax(self) -> float:
        """
        Returns the maximum redshift probed by this probe.

        Returns:
            The maximum redshift across all redshift bins.
        """
        pzs = self.params[0]
        return max(pz.zmax for pz in pzs)

    def kernel(
        self,
        cosmo: Cosmology,
        z: Union[float, np.ndarray],
        ell: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Computes the radial kernel for all redshift bins in this probe.

        Args:
            cosmo (Cosmology): The cosmology object providing necessary cosmological parameters.
            z (Union[float, np.ndarray]): The redshift(s) at which to compute the kernel.
            ell (Union[float, np.ndarray]): The angular scale(s) at which to compute the kernel.

        Returns:
            The radial kernel with shape `(nbins, nz)`, where `nbins` is the number of bins and `nz` is the number of redshift points.
        """
        z = np.atleast_1d(z)
        pzs, m = self.params[:2]
        kernel = weak_lensing_kernel(cosmo, pzs, z, ell)

        # Add intrinsic alignment kernel if enabled
        if self.config["ia_enabled"]:
            bias = self.params[2]
            kernel += nla_kernel(cosmo, pzs, bias, z, ell)

        # Apply multiplicative bias
        if isinstance(m, list):
            m = np.expand_dims(np.stack(m, axis=0), 1)
        kernel *= 1.0 + m
        return kernel

    def noise(self) -> np.ndarray:
        """
        Computes the noise power for all redshift bins.

        Returns:
            The noise power for each bin with shape `(nbins,)`.
        """
        pzs = self.params[0]
        ngals = np.array([pz.gals_per_steradian for pz in pzs])

        if isinstance(self.config["sigma_e"], list):
            sigma_e = np.array(self.config["sigma_e"])
        else:
            sigma_e = self.config["sigma_e"]

        return sigma_e**2 / ngals


@register_pytree_node_class
class NumberCounts(container):
    """
    Represents a galaxy clustering probe with a set of redshift bins.

    This class handles the modeling of galaxy clustering, optionally including the effect of redshift space distortions (RSD).

    Attributes:
        redshift_bins (List): A list of redshift distributions for each redshift bin.
        bias (List[float]): The bias parameter for each redshift bin.
        has_rsd (bool): Indicates whether the redshift space distortion (RSD) effect is included.
    """

    def __init__(self, redshift_bins:List, bias: List, has_rsd: bool=False, **kwargs):
        super(NumberCounts, self).__init__(
            redshift_bins, bias, has_rsd=has_rsd, **kwargs
        )

    @property
    def zmax(self) -> float:
        """
        Returns the maximum redshift probed by this probe.

        Returns:
            float: The maximum redshift across all redshift bins.
        """
        # Extract parameters
        pzs = self.params[0]
        return max([pz.zmax for pz in pzs])

    @property
    def n_tracers(self) -> int:
        """
        Returns the number of tracers (redshift bins) for this probe.

        Returns:
            int: The number of redshift bins.
        """
        # Extract parameters
        pzs = self.params[0]
        return len(pzs)

    def kernel(self, cosmo: Cosmology, z: Union[float, np.ndarray], ell: Union[float, np.ndarray]) -> np.ndarray:
        """
        Computes the radial kernel for all redshift bins in this probe.

        Args:
            cosmo (Cosmology): The cosmology object providing necessary cosmological parameters.
            z (Union[float, np.ndarray]): The redshift(s) at which to compute the kernel.
            ell (Union[float, np.ndarray]): The angular scale(s) at which to compute the kernel.

        Returns:
            The radial kernel with shape `(nbins, nz)`, where `nbins` is the number of bins and `nz` is the number of redshift points.
        """
        z = np.atleast_1d(z)
        # Extract parameters
        pzs, bias = self.params
        # Retrieve density kernel
        kernel = density_kernel(cosmo, pzs, bias, z, ell)
        return kernel

    def noise(self) -> np.ndarray:
        """
        Computes the noise power for all redshift bins.

        Returns:
            np.ndarray: The noise power for each bin with shape `(nbins,)`.
        """
        # Extract parameters
        pzs = self.params[0]
        ngals = np.array([pz.gals_per_steradian for pz in pzs])
        return 1.0 / ngals




@register_pytree_node_class
class CIBTracer(container):
    r"""
    Class representing the tracer associated with the Cosmic Infrared Background (CIB).

    The kernel function associated with the CIB is represented as:

    $$
    W(\chi) = \frac{1}{1+z}
    $$

    Attributes:
        z_min (float): The minimum redshift. Defaults to 0.0.
        z_max (float): The maximum redshift. Defaults to 6.0.
        n_z (int): The number of redshifts. Defaults to 1024.
    """

    def __init__(self, z_min=0.0, z_max=6.0, n_z=1024):
        super(CIBTracer, self).__init__(z_min, z_max, n_z)

    @property
    def zmin(self) -> float:
        """
        Returns the minimum redshift probed by this tracer.

        Returns:
            float: The minimum redshift.
        """
        return self.params[0]

    @property
    def zmax(self) -> float:
        """
        Returns the maximum redshift probed by this tracer.

        Returns:
            float: The maximum redshift.
        """
        return self.params[1]

    @property
    def n_z(self) -> int:
        """
        Returns the number of redshifts.

        Returns:
            int: The number of redshifts.
        """
        return self.params[2]

    def kernel(self, cosmo: Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Cosmic Infrared Background (CIB) kernel.

        The kernel is computed using the power spectrum method with a given cosmology.

        Args:
            cosmo (Cosmology): The cosmology object, typically in the JAX COSMO framework.

        Returns:
            The CIB kernel values.
            The comoving radial distances corresponding to the kernel.
        """
        inputs = {
            "z_min": self.params[0],
            "z_max": self.params[1],
            "n_z": self.params[2],
        }
        return power_kernel(cosmo, amplitude=1.0, alpha=1, **inputs)


@register_pytree_node_class
class zPowerTracer(container):
    r"""
    Represents the tracer associated with the power kernel.

    The kernel is computed using the formula:

    $$
    W(\chi) = \dfrac{A}{(1+z)^\alpha}
    $$

    Attributes:
        z_min (float): The minimum redshift to compute the kernel. Defaults to 0.0.
        z_max (float): The maximum redshift to compute the kernel. Defaults to 6.0.
        n_z (int): The number of redshift points in the kernel. Defaults to 1024.
    """

    def __init__(self, z_min: float = 0.0, z_max: float = 6.0, n_z: int = 1024):
        super(zPowerTracer, self).__init__(z_min, z_max, n_z)

    @property
    def zmin(self) -> float:
        """
        Returns the minimum redshift probed by this tracer.

        Returns:
            float: The minimum redshift.
        """
        return self.params[0]

    @property
    def zmax(self) -> float:
        """
        Returns the maximum redshift probed by this tracer.

        Returns:
            float: The maximum redshift.
        """
        return self.params[1]

    @property
    def n_z(self) -> int:
        """
        Returns the number of redshift samples used in the kernel.

        Returns:
            int: The number of redshift samples.
        """
        return self.params[2]

    def kernel(
        self, cosmo: Cosmology, amplitude: float, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the power kernel.

        The kernel is defined as a function of redshift using a power-law model.

        Args:
            cosmo (Cosmology): The cosmology object providing necessary cosmological parameters.
            amplitude (float): The amplitude (A) of the power kernel.
            alpha (float): The power-law exponent of the kernel.

        Returns:
            The power kernel (np.ndarray) over redshift.
            The corresponding comoving radial distances (np.ndarray).
        """
        inputs = {
            "z_min": self.params[0],
            "z_max": self.params[1],
            "n_z": self.params[2],
        }
        return power_kernel(cosmo, amplitude=amplitude, alpha=alpha, **inputs)


@register_pytree_node_class
class tSZTracer(container):
    r"""
    Represents the tracer associated with the thermal Sunyaev-Zel'dovich (tSZ) Compton-y parameter.

    This tracer computes the radial kernel for the tSZ effect, modeled using the formula:

    $$
    W(\chi) = \dfrac{\sigma_T}{m_e c^2} \dfrac{1}{1+z}
    $$

    Attributes:
        z_max (float): The maximum redshift to be considered. Defaults to 6.0.
        n_z (int): The number of redshift samples. Defaults to 1024.
    """
    def __init__(self, z_max: float = 6.0, n_z: int = 1024):
        super(tSZTracer, self).__init__(z_max, n_z)

    @property
    def zmax(self) -> float:
        """
        Returns the maximum redshift probed by this tracer.

        Returns:
            The maximum redshift.
        """
        return self.params[0]

    @property
    def n_z(self) -> int:
        """
        Returns the number of redshift samples used in the kernel.

        Returns:
            The number of redshift samples.
        """
        return self.params[1]

    def kernel(self, cosmo: Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the thermal Sunyaev-Zel'dovich (tSZ) kernel.

        The kernel is computed using a power-law model with a predefined amplitude.

        Args:
            cosmo (Cosmology): The cosmology object providing necessary cosmological parameters.

        Returns:
            The tSZ kernel (np.ndarray) over redshift.
            The corresponding comoving radial distances (np.ndarray).
        """
        inputs = {
            "z_min": 0.0,
            "z_max": self.params[0],
            "n_z": self.params[1],
        }
        amp = 4.01710079e-06  # Amplitude of the tSZ kernel
        return power_kernel(cosmo, amplitude=amp, alpha=1.0, **inputs)

@register_pytree_node_class
class CMBLensingTracer(container):
    r"""
    A tracer for the CMB lensing convergence, following Equation 31 in the CCL paper.

    $$
    \Delta_{\ell}^{\kappa}(k) = -\dfrac{\ell(\ell+1)}{2}\int_{0}^{\chi_{*}}\dfrac{dz}{H(z)}\,\dfrac{r(\chi_{*}-\chi)}{r(\chi)r(\chi_{*})}T_{\phi+\psi}(k,z)
    $$

    where

    - $\chi_{*}\equiv\chi(z_{*})$

    - $\kappa$ is the convergence of a given source plane at redshift $z_{*}$

    This tracer computes the lensing convergence kernel for the Cosmic Microwave Background (CMB).

    Attributes:
        z_source (float): The redshift of the source plane (CMB).
        n_samples (int): The number of samples used for the comoving radial distance grid. Defaults to 100.
    """

    def __init__(self, z_source: float, n_samples: int = 100):
        super(CMBLensingTracer, self).__init__(z_source, n_samples)

    @property
    def z_source(self) -> float:
        """
        Returns the source redshift for the CMB lensing tracer.

        Returns:
            float: The redshift of the source plane (CMB).
        """
        return self.params[0]

    @property
    def n_samples(self) -> int:
        """
        Returns the number of samples for the comoving radial distance grid.

        Returns:
            int: The number of samples used for integration.
        """
        return self.params[1]

    def kernel(self, cosmo: Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the lensing convergence kernel for the CMB.

        The kernel is computed using the cosmological parameters and the redshift of the source plane.

        Args:
            cosmo (Cosmology): A cosmology object providing the necessary cosmological parameters.

        Returns:
            The kernel values for the CMB lensing convergence.
            The corresponding comoving radial distances.
        """
        inputs = {
            "z_source": self.params[0],
            "n_samples": self.params[1],
        }
        return kappa_kernel(cosmo, **inputs)
