# This module defines kernel functions for various tracers
from typing import Tuple
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

__all__ = ["WeakLensing", "NumberCounts"]


@jit
def weak_lensing_kernel(cosmo, pzs, z, ell):
    """
    Returns a weak lensing kernel

    Note: this function handles differently nzs that correspond to extended redshift
    distribution, and delta functions.
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
def density_kernel(cosmo, pzs, bias, z, ell):
    """
    Computes the number counts density kernel
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


@jit
def nla_kernel(cosmo, pzs, bias, z, ell):
    """
    Computes the NLA IA kernel
    """
    if any(isinstance(pz, rds.delta_nz) for pz in pzs):
        raise NotImplementedError(
            "NLA kernel not properly implemented for delta redshift distributions"
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
    # Apply common A_IA normalization to the kernel
    # Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
    radial_kernel *= (
        -(5e-14 * const.rhocrit) * cosmo.Omega_m / bkgrd.growth_factor(cosmo, z2a(z))
    )
    # Constant factor
    constant_factor = 1.0
    # Ell dependent factor
    ell_factor = np.sqrt((ell - 1) * (ell) * (ell + 1) * (ell + 2)) / (ell + 0.5) ** 2
    return constant_factor * ell_factor * radial_kernel


@register_pytree_node_class
class WeakLensing(container):
    """
    Class representing a weak lensing probe, with a bunch of bins

    Parameters:
    -----------
    redshift_bins: list of nzredshift distributions
    ia_bias: (optional) if provided, IA will be added with the NLA model,
    either a single bias object or a list of same size as nzs
    multiplicative_bias: (optional) adds an (1+m) multiplicative bias, either single
    value or list of same length as redshift bins

    Configuration:
    --------------
    sigma_e: intrinsic galaxy ellipticity
    """

    def __init__(
        self,
        redshift_bins,
        ia_bias=None,
        multiplicative_bias=0.0,
        sigma_e=0.26,
        **kwargs
    ):
        # Depending on the Configuration we will trace or not the ia_bias in the
        # container
        if ia_bias is None:
            ia_enabled = False
            args = (redshift_bins, multiplicative_bias)
        else:
            ia_enabled = True
            args = (redshift_bins, multiplicative_bias, ia_bias)
        if "ia_enabled" not in kwargs.keys():
            kwargs["ia_enabled"] = ia_enabled
        super(WeakLensing, self).__init__(*args, sigma_e=sigma_e, **kwargs)

    @property
    def n_tracers(self):
        """
        Returns the number of tracers for this probe, i.e. redshift bins
        """
        # Extract parameters
        pzs = self.params[0]
        return len(pzs)

    @property
    def zmax(self):
        """
        Returns the maximum redsfhit probed by this probe
        """
        # Extract parameters
        pzs = self.params[0]
        return max([pz.zmax for pz in pzs])

    def kernel(self, cosmo, z, ell):
        """
        Compute the radial kernel for all nz bins in this probe.

        Returns:
        --------
        radial_kernel: shape (nbins, nz)
        """
        z = np.atleast_1d(z)
        # Extract parameters
        pzs, m = self.params[:2]
        kernel = weak_lensing_kernel(cosmo, pzs, z, ell)
        # If IA is enabled, we add the IA kernel
        if self.config["ia_enabled"]:
            bias = self.params[2]
            kernel += nla_kernel(cosmo, pzs, bias, z, ell)
        # Applies measurement systematics
        if isinstance(m, list):
            m = np.expand_dims(np.stack([mi for mi in m], axis=0), 1)
        kernel *= 1.0 + m
        return kernel

    def noise(self):
        """
        Returns the noise power for all redshifts
        return: shape [nbins]
        """
        # Extract parameters
        pzs = self.params[0]
        # retrieve number of galaxies in each bins
        ngals = np.array([pz.gals_per_steradian for pz in pzs])
        if isinstance(self.config["sigma_e"], list):
            sigma_e = np.array([s for s in self.config["sigma_e"]])
        else:
            sigma_e = self.config["sigma_e"]
        return sigma_e**2 / ngals


@register_pytree_node_class
class NumberCounts(container):
    """Class representing a galaxy clustering probe, with a bunch of bins

    Parameters:
    -----------
    redshift_bins: nzredshift distributions

    Configuration:
    --------------
    has_rsd....
    """

    def __init__(self, redshift_bins, bias, has_rsd=False, **kwargs):
        super(NumberCounts, self).__init__(
            redshift_bins, bias, has_rsd=has_rsd, **kwargs
        )

    @property
    def zmax(self):
        """
        Returns the maximum redsfhit probed by this probe
        """
        # Extract parameters
        pzs = self.params[0]
        return max([pz.zmax for pz in pzs])

    @property
    def n_tracers(self):
        """Returns the number of tracers for this probe, i.e. redshift bins"""
        # Extract parameters
        pzs = self.params[0]
        return len(pzs)

    def kernel(self, cosmo, z, ell):
        """Compute the radial kernel for all nz bins in this probe.

        Returns:
        --------
        radial_kernel: shape (nbins, nz)
        """
        z = np.atleast_1d(z)
        # Extract parameters
        pzs, bias = self.params
        # Retrieve density kernel
        kernel = density_kernel(cosmo, pzs, bias, z, ell)
        return kernel

    def noise(self):
        """Returns the noise power for all redshifts
        return: shape [nbins]
        """
        # Extract parameters
        pzs = self.params[0]
        ngals = np.array([pz.gals_per_steradian for pz in pzs])
        return 1.0 / ngals


def power_kernel(
    cosmo: Cosmology, amplitude: float, alpha: float, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Tracer obeying a power law alpha with an amplitude.

    Args:
        cosmo (Cosmology): the cosmology object in JAXCOSMO
        amplitude (float): the amplitude of the kernel
        alpha (float): the power of the kernel

    Returns:
        Tuple[np.ndarray, np.ndarray]: the kernel and the comoving radial distance
    """
    z_min = kwargs["z_min"]
    z_max = kwargs["z_max"]
    n_z = kwargs["n_z"]
    a_arr, chi_arr = bkgrd.scale_of_chi(cosmo, z_min, z_max, n_z)
    w_arr = amplitude * a_arr**alpha
    return w_arr, chi_arr


@register_pytree_node_class
class CIBTracer(container):
    """Class representing the Tracer associated with the cosmic infrared background (CIB)

    z_min (float, optional): The minimum redshift. Defaults to 0.0.
    z_max (float, optional): The maximim redshift. Defaults to 6.0.
    n_z (int, optional): The number of redshifts. Defaults to 1024.
    """

    def __init__(self, z_min=0.0, z_max=6.0, n_z=1024):
        super(CIBTracer, self).__init__(z_min, z_max, n_z)

    @property
    def zmin(self) -> float:
        """Returns the minimum redsfhit probed by this probe

        Returns:
            float: minimum redshift
        """
        return self.params[0]

    @property
    def zmax(self) -> float:
        """Returns the maximum redsfhit probed by this probe

        Returns:
            float: maximum redshift
        """
        return self.params[1]

    @property
    def n_z(self) -> int:
        """Returns the number of redshifts

        Returns:
            int: number of redshifts
        """
        return self.params[2]

    def kernel(self, cosmo: Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the cosmic infrared background kernel.

        Args:
            cosmo (Cosmology): the cosmology object in JAX COSMO

        Returns:
            Tuple[np.ndarray, np.ndarray]: the CIB kernel and the comoving radial distance
        """
        inputs = {
            "z_min": self.params[0],
            "z_max": self.params[1],
            "n_z": self.params[2],
        }
        return power_kernel(cosmo, amplitude=1.0, alpha=1, **inputs)


@register_pytree_node_class
class zPowerTracer(container):
    """Class representing the Tracer associated with the power kernel.

    z_min (float, optional): The minimum redshift. Defaults to 0.0.
    z_max (float, optional): The maximim redshift. Defaults to 6.0.
    n_z (int, optional): The number of redshifts. Defaults to 1024.
    """

    def __init__(self, z_min=0.0, z_max=6.0, n_z=1024):
        super(zPowerTracer, self).__init__(z_min, z_max, n_z)

    @property
    def zmin(self) -> float:
        """Returns the minimum redsfhit probed by this probe

        Returns:
            float: minimum redshift
        """
        return self.params[0]

    @property
    def zmax(self) -> float:
        """Returns the maximum redsfhit probed by this probe

        Returns:
            float: maximum redshift
        """
        return self.params[1]

    @property
    def n_z(self) -> int:
        """Returns the number of redshifts

        Returns:
            int: number of redshifts
        """
        return self.params[2]

    def kernel(
        self, cosmo: Cosmology, amplitude: float, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the cosmic infrared background kernel.

        Args:
            cosmo (Cosmology): the cosmology object in JAX COSMO
            amplitude (float): the aplitude of this kernel
            alpha (float): the power associated with this kernel

        Returns:
            Tuple[np.ndarray, np.ndarray]: the CIB kernel and the comoving radial distance
        """
        inputs = {
            "z_min": self.params[0],
            "z_max": self.params[1],
            "n_z": self.params[2],
        }
        return power_kernel(cosmo, amplitude=amplitude, alpha=alpha, **inputs)


@register_pytree_node_class
class tSZTracer(container):
    """Class representing the Tracer associated with the thermal Sunyaev Zel’dovich Compton-y parameter.

    z_max (float, optional): The maximim redshift. Defaults to 6.0.
    n_z (int, optional): The number of redshifts. Defaults to 1024.
    """

    def __init__(self, z_max=6.0, n_z=1024):
        super(tSZTracer, self).__init__(z_max, n_z)

    @property
    def zmax(self) -> float:
        """Returns the maximum redsfhit probed by this probe

        Returns:
            float: maximum redshift
        """
        return self.params[0]

    @property
    def n_z(self) -> int:
        """Returns the number of redshifts

        Returns:
            int: number of redshifts
        """
        return self.params[1]

    def kernel(self, cosmo: Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the tSZ kernel.

        Args:
            cosmo (Cosmology): the cosmology object in JAX COSMO

        Returns:
            Tuple[np.ndarray, np.ndarray]: the CIB kernel and the comoving radial distance.
        """
        inputs = {
            "z_min": 0.0,
            "z_max": self.params[0],
            "n_z": self.params[1],
        }
        amp = 4.01710079e-06
        return power_kernel(cosmo, amplitude=amp, alpha=1.0, **inputs)


def kappa_kernel(
    cosmo: Cosmology, z_source: float, n_samples: int = 100, ell: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """This convenience function returns the radial kernel for
    CMB-lensing-like tracers.

    Args:
        cosmo (Cosmology): the cosmology object with all cosmological parameters.
        z_source (float): the redshift of the source
        n_samples (int, optional): the number of samples of comoving radial distance. Defaults to 100.
        ell (int, optional): the ell mode. Defaults to 2000.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the kernel and the comoving radial distance
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


@register_pytree_node_class
class CMBLensingTracer(container):
    """A Tracer for CMB lensing convergence. Equation 31 in the CCL paper.

    Args:
        z_source (float, optional): the redshift of the source.
        n_samples (int, optional): the number of samples of the comoving radial distance. Defaults to 100.
    """

    def __init__(self, z_source: float, n_samples: int = 100):
        super(CMBLensingTracer, self).__init__(z_source, n_samples)

    @property
    def z_source(self) -> float:
        """Returns the source redshift.

        Returns:
            float: the source redshift
        """
        return self.params[0]

    @property
    def n_samples(self) -> int:
        """Returns the number of samples on the comoving radial distance grid.

        Returns:
            int: number of samples
        """
        return self.params[1]

    def kernel(self, cosmo: Cosmology) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the CMB kernel.

        Args:
            cosmo (Cosmology): the cosmology object in JAX COSMO

        Returns:
            Tuple[np.ndarray, np.ndarray]: the CMB kernel and the comoving radial distance.
        """
        inputs = {
            "z_source": self.params[0],
            "n_samples": self.params[1],
        }
        return kappa_kernel(cosmo, **inputs)
