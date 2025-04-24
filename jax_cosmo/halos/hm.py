import jax
import jax.numpy as jnp
from jax import lax
from typing import Union, Optional, Callable
from interpax import Interpolator2D
from jax_cosmo.core import Cosmology
from jax_cosmo.cclconstants import PhysicalConstants, CCLSplineParams, CCLGSLParams
from jax_cosmo.halos.hmbase import get_logM_sigM, calculate_mass_function
from jax_cosmo.background import comoving_volume_element
from jax_cosmo.power import linear_matter_power

CCLCST = PhysicalConstants()
CCL_SPLINE_PARAMS = CCLSplineParams()
CCL_GSL_PARAMS = CCLGSLParams()

ArrayLike = Union[float, jnp.ndarray]
Array = jnp.ndarray


class Profile2pt:
    def __init__(self, r_corr=0.0):
        self.r_corr = r_corr

    def update_parameters(self, r_corr=None):
        """Update any of the parameters associated with this 1-halo
        2-point correlator. Any parameter set to ``None`` won't be updated.
        """
        if r_corr is not None:
            self.r_corr = r_corr

    def fourier_2pt(
        self,
        cosmo: Cosmology,
        wavenumber: jnp.ndarray,
        mass: jnp.ndarray,
        scale_factor: jnp.ndarray,
        interpolator: Interpolator2D,
        prof,
        prof2=None,
        diag: bool = True,
    ):

        if prof2 is None:
            prof2 = prof

        uk1 = prof._fourier_analytic(
            cosmo, wavenumber, mass, scale_factor, interpolator
        )

        if prof == prof2:
            uk2 = uk1
        else:
            uk2 = prof2._fourier_analytic(
                cosmo, wavenumber, mass, scale_factor, interpolator
            )

        # TODO: This should be implemented in _fourier_variance
        if (diag is True) or (isinstance(wavenumber, float)):
            output = uk1 * uk2 * (1 + self.r_corr)
        elif isinstance(mass, float):
            output = uk1[None, :] * uk2[:, None] * (1 + self.r_corr)
        else:
            output = uk1[:, None, :] * uk2[:, :, None] * (1 + self.r_corr)

        return output


class JAXHMCalculator:
    """Halo model ingredients computed using JAX.

    Attributes:
        mass_function (Callable): Mass function model callable.
        halo_bias (Callable): Halo bias model callable.
        mass_def (Optional[object]): Mass definition object (optional).
        precision (dict): Dictionary containing mass sampling precision parameters.
        _lmass (jnp.ndarray): Logarithmic mass array.
        _mass (jnp.ndarray): Mass array.
    """

    def __init__(
        self,
        mass_function: Callable,
        halo_bias: Callable,
        mass_def: Optional[object] = None,
        log10M_min: float = 8.0,
        log10M_max: float = 16.0,
        nM: int = 128,
    ):
        """Initialize JAXHaloModel.

        Args:
            mass_function (Callable): Mass function model.
            halo_bias (Callable): Halo bias model.
            mass_def (Optional[object], optional): Mass definition object. Defaults to None.
            log10M_min (float, optional): Minimum log10 mass. Defaults to 8.0.
            log10M_max (float, optional): Maximum log10 mass. Defaults to 16.0.
            nM (int, optional): Number of mass bins. Defaults to 128.
        """
        self.precision = {"log10M_min": log10M_min, "log10M_max": log10M_max, "nM": nM}
        self.mass_function = mass_function
        self.halo_bias = halo_bias
        self.mass_def = mass_def

        self._lmass = jnp.linspace(log10M_min, log10M_max, nM)
        self._mass = 10.0**self._lmass
        self._m0 = self._mass[0]

        self._cosmo_mf = None
        self._a_mf = -1.0
        self._cosmo_bf = None
        self._a_bf = -1.0

    def get_mass_function(
        self,
        cosmo: Cosmology,
        scale_factor: float,
        interpolator: Interpolator2D,
        rho0: float,
    ) -> None:
        """Compute and cache the mass function $n(M,a)$.

        Args:
            cosmo (Cosmology): Cosmology object.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            rho0 (float): Background matter density today.
        """
        if scale_factor != self._a_mf or cosmo != self._cosmo_mf:
            self._mf, _ = calculate_mass_function(
                cosmo, self._mass, scale_factor, self.mass_function, interpolator
            )
            integ = jnp.trapezoid(self._mf * self._mass, self._lmass)
            self._mf0 = (rho0 - integ) / self._m0
            self._cosmo_mf = cosmo
            self._a_mf = scale_factor

    def get_halo_bias(
        self,
        cosmo: Cosmology,
        scale_factor: float,
        interpolator: Interpolator2D,
        rho0: float,
    ) -> None:
        """Compute and cache the halo bias $b(M,a)$.

        Args:
            cosmo (Cosmology): Cosmology object.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            rho0 (float): Background matter density today.
        """
        if scale_factor != self._a_bf or cosmo != self._cosmo_bf:
            log_mass, sigma_mass, dlns_dlogM = get_logM_sigM(
                cosmo, self._mass, scale_factor, interpolator
            )
            self._bf = self.halo_bias.compute_bias(cosmo, sigma_mass, scale_factor)
            integ = jnp.trapezoid(self._mf * self._bf * self._mass, self._lmass)
            self._mbf0 = (rho0 - integ) / self._m0
            self._cosmo_bf = cosmo
            self._a_bf = scale_factor

    def get_ingredients(
        self,
        cosmo: Cosmology,
        scale_factor: float,
        interpolator: Interpolator2D,
        get_bf: bool,
    ) -> None:
        """Ensure mass function (and optionally halo bias) are cached.

        Args:
            cosmo (Cosmology): Cosmology object.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            get_bf (bool): Whether to also compute the halo bias.
        """
        rho0 = CCLCST.RHO_CRITICAL * cosmo.Omega_m * cosmo.h**2
        self.get_mass_function(cosmo, scale_factor, interpolator, rho0)
        if get_bf:
            self.get_halo_bias(cosmo, scale_factor, interpolator, rho0)

    def integrate_over_mf(self, array_2: Array) -> Array:
        """Integral over $n(M,a) \\times f(M)$.

        Args:
            array_2 (Array): Function values $f(M)$ sampled at the model mass grid.

        Returns:
            Array: Result of the integral.
        """
        i1 = jnp.trapezoid(self._mf * array_2, self._lmass)
        return i1 + self._mf0 * array_2[..., 0]

    def integrate_over_mbf(self, array_2: Array) -> Array:
        """Integral over $n(M,a) \\times b(M,a) \\times f(M)$.

        Args:
            array_2 (Array): Function values $f(M)$ sampled at the model mass grid.

        Returns:
            Array: Result of the integral.
        """

        i1 = jnp.trapezoid(self._mf * self._bf * array_2, self._lmass)
        return i1 + self._mbf0 * array_2[..., 0]

    def integrate_over_massfunc(
        self,
        func: Callable,
        cosmo: Cosmology,
        scale_factor: float,
        interpolator: Interpolator2D,
        rho0: float,
    ):
        """Integrate an arbitrary function over the mass function.

        Args:
            func (Callable): Function of mass to integrate.
            cosmo (Cosmology): Cosmology object.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            rho0 (float): Background matter density today.

        Returns:
            Array: Result of the integral.
        """
        fM = func(self._mass)
        self.get_ingredients(cosmo, scale_factor, interpolator, rho0, get_bf=False)
        return self.integrate_over_mf(fM)

    def number_counts(
        self, cosmo: Cosmology, selection: Callable, a_min=None, a_max=1.0, na=128
    ):
        """Compute number counts over a given selection and redshift range.

        Args:
            cosmo (Cosmology): Cosmology object.
            selection (Callable): Selection function over mass and scale factor.
            a_min (float, optional): Minimum scale factor. Defaults to CCL_SPLINE_PARAMS.A_SPLINE_MIN.
            a_max (float, optional): Maximum scale factor. Defaults to 1.0.
            na (int, optional): Number of scale factor bins. Defaults to 128.

        Returns:
            Array: Total number counts.
        """
        if a_min is None:
            a_min = CCL_SPLINE_PARAMS.A_SPLINE_MIN
        agrid = jnp.linspace(a_min, a_max, na)

        dVda = comoving_volume_element(agrid)
        mint = jnp.zeros_like(agrid)

        for i, _a in enumerate(agrid):
            self.get_ingredients(cosmo, _a, get_bf=False)
            _selm = jnp.atleast_2d(selection(self._mass, _a)).T
            mint[i] = jnp.trapezoid(dVda[i] * self._mf * _selm, self._lmass).squeeze()

        return jnp.trapezoid(mint, agrid)

    def I_0_1(
        self,
        cosmo: Cosmology,
        wavenumber: Array,
        scale_factor: float,
        interpolator: Interpolator2D,
        prof: object,
    ) -> Array:
        """Compute the integral I_0_1.

        Args:
            cosmo (Cosmology): Cosmology object.
            wavenumber (Array): Array of wavenumbers.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            prof (object): Profile object with a `_fourier_analytic` method.

        Returns:
            Array: Result of the integral over mass function.
        """
        self.get_ingredients(cosmo, scale_factor, interpolator, get_bf=False)
        uk = prof._fourier_analytic(
            cosmo, wavenumber, self._mass, scale_factor, interpolator
        ).T
        return self.integrate_over_mf(uk)

    def I_1_1(
        self,
        cosmo: Cosmology,
        wavenumber: Array,
        scale_factor: float,
        interpolator: Interpolator2D,
        prof: object,
    ) -> Array:
        """Compute the integral I_1_1.

        Args:
            cosmo (Cosmology): Cosmology object.
            wavenumber (Array): Array of wavenumbers.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            prof (object): Profile object with a `_fourier_analytic` method.

        Returns:
            Array: Result of the integral over mass function weighted by halo bias.
        """
        self.get_ingredients(cosmo, scale_factor, interpolator, get_bf=True)
        uk = prof._fourier_analytic(
            cosmo, wavenumber, self._mass, scale_factor, interpolator
        ).T
        return self.integrate_over_mbf(uk)

    def I_1_3(
        self,
        cosmo: Cosmology,
        wavenumber: Array,
        scale_factor: float,
        interpolator: Interpolator2D,
        prof: object,
        prof_2pt: object,
        prof2: Optional[object] = None,
        prof3: Optional[object] = None,
    ) -> Array:
        """Compute the integral I_1_3.

        Args:
            cosmo (Cosmology): Cosmology object.
            wavenumber (Array): Array of wavenumbers.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            prof (object): Profile object with a `_fourier_analytic` method.
            prof_2pt (object): Profile 2-pt correlation object with `fourier_2pt` method.
            prof2 (Optional[object], optional): Second profile. Defaults to prof.
            prof3 (Optional[object], optional): Third profile. Defaults to prof2.

        Returns:
            Array: Result of the integral over mass function weighted by halo bias.
        """
        if prof2 is None:
            prof2 = prof
        if prof3 is None:
            prof3 = prof2

        self.get_ingredients(cosmo, scale_factor, interpolator, get_bf=True)
        uk1 = prof._fourier_analytic(
            cosmo, wavenumber, self._mass, scale_factor, interpolator
        ).T
        uk23 = prof_2pt.fourier_2pt(
            cosmo,
            wavenumber,
            self._mass,
            scale_factor,
            interpolator,
            prof2,
            prof2=prof3,
        ).T

        uk = uk1[None, :, :] * uk23[:, None, :]
        return self.integrate_over_mbf(uk)

    def I_0_2(
        self,
        cosmo: Cosmology,
        wavenumber: Array,
        scale_factor: float,
        interpolator: Interpolator2D,
        prof: object,
        prof_2pt: object,
        prof2: Optional[object] = None,
    ) -> Array:
        """Compute the integral I_0_2.

        Args:
            cosmo (Cosmology): Cosmology object.
            wavenumber (Array): Array of wavenumbers.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            prof (object): Profile object.
            prof_2pt (object): Profile 2-pt correlation object.
            prof2 (Optional[object], optional): Second profile. Defaults to prof.

        Returns:
            Array: Result of the integral over mass function.
        """
        if prof2 is None:
            prof2 = prof

        self.get_ingredients(cosmo, scale_factor, interpolator, get_bf=False)
        uk = prof_2pt.fourier_2pt(
            cosmo, wavenumber, self._mass, scale_factor, interpolator, prof, prof2=prof2
        ).T
        return self.integrate_over_mf(uk)

    def I_1_2(
        self,
        cosmo: Cosmology,
        wavenumber: Array,
        scale_factor: float,
        interpolator: Interpolator2D,
        prof: object,
        prof_2pt: object,
        prof2: Optional[object] = None,
        diag: bool = True,
    ) -> Array:
        """Compute the integral I_1_2.

        Args:
            cosmo (Cosmology): Cosmology object.
            wavenumber (Array): Array of wavenumbers.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            prof (object): Profile object.
            prof_2pt (object): Profile 2-pt correlation object.
            prof2 (Optional[object], optional): Second profile. Defaults to prof.
            diag (bool, optional): Whether the output is diagonal. Defaults to True.

        Returns:
            Array: Result of the integral over mass function weighted by halo bias.
        """
        if prof2 is None:
            prof2 = prof

        self.get_ingredients(cosmo, scale_factor, interpolator, get_bf=True)
        uk = prof_2pt.fourier_2pt(
            cosmo, wavenumber, self._mass, scale_factor, interpolator, prof, prof2, diag
        )

        if diag:
            uk = uk.T
        else:
            uk = jnp.transpose(uk, axes=[1, 2, 0])

        return self.integrate_over_mbf(uk)

    def I_0_22(
        self,
        cosmo: Cosmology,
        wavenumber: Array,
        scale_factor: float,
        interpolator: Interpolator2D,
        prof: object,
        prof2: Optional[object] = None,
        prof3: Optional[object] = None,
        prof4: Optional[object] = None,
        prof12_2pt: Optional[object] = None,
        prof34_2pt: Optional[object] = None,
    ) -> Array:
        """Compute the integral I_0_22.

        Args:
            cosmo (Cosmology): Cosmology object.
            wavenumber (Array): Array of wavenumbers.
            scale_factor (float): Scale factor.
            interpolator (Interpolator2D): Interpolator for variance quantities.
            prof (object): First profile object.
            prof2 (Optional[object], optional): Second profile. Defaults to prof.
            prof3 (Optional[object], optional): Third profile. Defaults to prof.
            prof4 (Optional[object], optional): Fourth profile. Defaults to prof2.
            prof12_2pt (Optional[object], optional): 2-pt correlation profile between prof and prof2.
            prof34_2pt (Optional[object], optional): 2-pt correlation profile between prof3 and prof4. Defaults to prof12_2pt.

        Returns:
            Array: Result of the integral over mass function.
        """
        if prof2 is None:
            prof2 = prof
        if prof3 is None:
            prof3 = prof
        if prof4 is None:
            prof4 = prof2
        if prof34_2pt is None:
            prof34_2pt = prof12_2pt

        self.get_ingredients(cosmo, scale_factor, interpolator, get_bf=False)
        uk12 = prof12_2pt.fourier_2pt(
            cosmo, wavenumber, self._mass, scale_factor, interpolator, prof, prof2
        ).T

        if (prof, prof2, prof12_2pt) == (prof3, prof4, prof34_2pt):
            uk34 = uk12
        else:
            uk34 = prof34_2pt.fourier_2pt(
                cosmo, wavenumber, self._mass, scale_factor, interpolator, prof3, prof4
            ).T

        return self.integrate_over_mf(uk12[None, :, :] * uk34[:, None, :])


def halomod_power_spectrum(
    cosmo,
    hmc,
    k,
    a,
    interpolator,
    prof,
    prof2=None,
    prof_2pt=None,
    get_1h=True,
    get_2h=True,
):

    a_use = jnp.atleast_1d(a).astype(float)
    k_use = jnp.atleast_1d(k).astype(float)

    if prof2 is None:
        prof2 = prof
    if prof_2pt is None:
        prof_2pt = Profile2pt()

    na = len(a_use)
    nk = len(k_use)
    out = jnp.zeros([na, nk])
    norm = CCLCST.RHO_CRITICAL * cosmo.Omega_m * cosmo.h**2

    for ia, aa in enumerate(a_use):

        if get_2h:
            # bias factors
            i11_1 = hmc.I_1_1(cosmo, k_use, aa, interpolator, prof)

            if prof2 == prof:
                i11_2 = i11_1
            else:
                i11_2 = hmc.I_1_1(cosmo, k_use, aa, interpolator, prof2)

            pk_lin = linear_matter_power(cosmo, k_use / cosmo.h, aa) / cosmo.h**3
            pk_2h = pk_lin * i11_1 * i11_2

        else:
            pk_2h = 0

        if get_1h:
            pk_1h = hmc.I_0_2(
                cosmo, k_use, aa, interpolator, prof, prof_2pt=prof_2pt, prof2=prof2
            )  # 1h term
        else:
            pk_1h = 0

        out = out.at[ia].set(pk_1h + pk_2h)

    if jnp.ndim(a) == 0:
        out = jnp.squeeze(out, axis=0)
    if jnp.ndim(k) == 0:
        out = jnp.squeeze(out, axis=-1)
    return out / norm**2
