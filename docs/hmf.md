---
title: Halos
icon: material/orbit
---

## Concentration

### Bhattacharya et al. (2013)

The concentration parameter depends on the growth factor $g(z)$. The peak height of perturbations is given by:

$$
\nu = \frac{\delta_c}{\sigma_{M}}
$$

where:

- $\nu$ is the peak height.
- $\sigma_{M}$ is the variance of the density field at mass scale $M$.

The concentration parameter is:

$$
c(M, z) = A \cdot g(z)^B \cdot \nu^C
$$

where:

- $c(M, z)$ is the concentration parameter.
- $A, B, C$ are empirical constants, depending on the mass definition:

    - [x] Virial mass, $M_{\text{vir}}$: $A = 7.7, B = 0.9, C = -0.29$
    - [x] 200 mean density, $M_{200m}$: $A = 9.0, B = 1.15, C = -0.29$
    - [x] 200 critical density, $M_{200c}$: $A = 5.9, B = 0.54, C = -0.35$


The final concentration is computed as:

$$
c(M, z) = A \cdot g(z)^B \cdot \left( \frac{\delta_c}{\sigma_{M}} \right)^C
$$

See [Bhattacharya et al. (2013)](https://arxiv.org/pdf/1112.5479) for further details.

### Diemer et al. (2015)

The Lagrangian radius $R$ of a halo is computed from its mass, $M$. The characteristic wavenumber associated with the halo is:

$$
k_R = \frac{2\pi}{R} \cdot \kappa
$$

where:

- $k_R$ is the characteristic wavenumber.
- $\kappa = 1.0$ is a dimensionless parameter (default).


The slope of the logarithmic matter power spectrum is:

$$
  \frac{d\log P}{d\log k}
$$

where $P(k)$ is the linear matter power spectrum. The slope is evaluate at the computed $k_R$.


The peak height of density fluctuations is:

$$
\nu = \frac{\delta_c}{\sigma_{M}}
$$

The baseline concentration is:

$$
\phi = \phi_0 + \frac{d\log P}{d\log k} \cdot \phi_1
$$

The transition peak height $\nu_0$ is:

$$
\nu_0 = \eta_0 + \frac{d\log P}{d\log k} \cdot \eta_1
$$

where:

- $\phi_0 = 6.58$, $\phi_1 = 1.27$ are empirical parameters.
- $\eta_0 = 7.28$, $\eta_1 = 1.56$ are empirical parameters.


The concentration is given by:

$$
c(M, z) = 0.5 \cdot \phi \cdot \left( \left( \frac{\nu_0}{\nu} \right)^\alpha + \left( \frac{\nu}{\nu_0} \right)^\beta \right)
$$

where:

- $\alpha = 1.08$, $\beta = 1.77$ are power-law exponents.
- The function interpolates between low-$\nu$ and high-$\nu$ behavior.

See [Diemer et al. (2015)](https://arxiv.org/pdf/1407.4730) for further details.

### Duffy et al. (2008)

The concentration-mass relation from [Duffy et al. (2008)](https://arxiv.org/pdf/0804.2486) is given by:

$$
c(M, z) = A \cdot \left(\frac{M}{M_{\text{pivot}}} \right)^B \cdot a^{-C}
$$

where:

- $c(M, z)$ is the halo concentration.
- $M_{\text{pivot}}$ is a reference mass set to $2 \times 10^{12} h^{-1} M_{\odot}$.
- $a$ is the scale factor.
- $A, B, C$ are empirical parameters that depend on the mass definition:
    + [x] Virial mass ($M_{\text{vir}}$): $A = 7.85, B = -0.081, C = -0.71$
    + [x] 200 mean density ($M_{200m}$): $A = 10.14, B = -0.081, C = -1.01$
    + [x] 200 critical density ($M_{200c}$): $A = 5.71, B = -0.084, C = -0.47$

### Klypin et al. (2011)

The concentration-mass relation from [Klypin et al. (2011)](https://arxiv.org/pdf/1002.3660) is given by:

$$
c(M, z) = 9.6 \cdot \left(\frac{M}{M_{\text{pivot}}} \right)^{-0.075}
$$

where:

- $c(M, z)$ is the halo concentration.
- $M$ is the halo mass (in solar masses).
- $M_{\text{pivot}}$ is a reference mass:

$$
M_{\text{pivot}} = \frac{10^{12} M_{\odot}}{h}
$$

- h is the dimensionless Hubble parameter.

This equation describes how the concentration of dark matter halos varies with mass, but **it does not explicitly depend on redshift**.


### Prada et al. (2012)

The concentration-mass relation from [Prada et al. (2012)](https://arxiv.org/pdf/1104.5130) is given by:

$$
c(M, z) = b_0(x) \cdot c_{\text{eff}}(\sigma_M)
$$

where

$$
c_{\text{eff}}(\sigma_M) = 2.881\,\text{exp}\left(\frac{0.060}{\sigma_{P}^{2}}\right)\left[\left( \frac{\sigma_P}{1.257} \right)^{1.022} + 1 \right],
$$


and the modified variance is:

$$
\sigma_P = b_1(x) \cdot \sigma_M.
$$

where

$$
x = a \times \left( \frac{\Omega_{\text{de}}}{\Omega_m} \right)^{1/3}
$$

and

- $a = \frac{1}{1+z}$ is the scale factor,
- $\Omega_{\text{de}}$ and $\Omega_m$ are the dark energy and matter density parameters,
- $\sigma_M$ is the mass variance,
- $b_0(x)$ and $b_1(x)$ are computed using:

$$
b_0(x) = \frac{c_{\min}(x, x_0, c_0, c_1, \alpha)}{c_{\min}(1.393, x_0, c_0, c_1, \alpha)}
$$

$$
b_1(x) = \frac{c_{\min}(x, x_1, i_0, i_1, \beta)}{c_{\min}(1.393, x_1, i_0, i_1, \beta)}
$$

The expression for $c_{\min}$ is:

$$
c_{\min}(x, x_0, v_0, v_1, v_2) = v_0 + (v_1 - v_0) \times \left( \frac{\text{tan}^{-1}(v_2 \cdot (x - x_0))}{\pi} + 0.5 \right).
$$

The model parameters are:

- $c_0 = 3.681$, $c_1 = 5.033$, $\alpha = 6.948$, $x_0 = 0.424$
- $i_0 = 1.047$, $i_1 = 1.646$, $\beta = 7.386$, $x_1 = 0.526$

This formulation accounts for the **evolution of halo concentration** based on mass variance and redshift.

### Ishiyama et al. (2021)

The concentration-mass relation from [Ishiyama et al. (2021)](https://arxiv.org/pdf/2007.14720) is given by:

$$
c(M, z) = C_{\text{factor}} \cdot G^{-1}(\nu)
$$

where:

$$
G(x, n_{\text{eff}}) = \frac{x}{\left( \ln(1 + x) - \frac{x}{1 + x} \right)^{(5 + n_{\text{eff}}) / 6}}
$$

The peak height is given by:

$$
\nu = \frac{\delta_c}{\sigma_M}
$$

where:

- $\delta_c$ is the critical overdensity for collapse and
- $\sigma_M$ is the mass variance.

The effective spectral index is:

$$
n_{\text{eff}} = -2 \frac{d \ln \sigma}{d \ln M} - 3.
$$


$\alpha_{\text{eff}}$ is the growth rate factor and the scaling factors are given by:

$$
A_{\text{factor}} = a_0 \left( 1 + a_1 (n_{\text{eff}} + 3) \right)
$$

$$
B_{\text{factor}} = b_0 \left( 1 + b_1 (n_{\text{eff}} + 3) \right)
$$

$$
C_{\text{factor}} = 1 - c_{\alpha} (1 - \alpha_{\text{eff}})
$$

$G^{-1}(\nu)$ is the inverse of $G(x)$, obtained by root finding, given $n_{\text{eff}}$ and $V$:

$$
V = \frac{A_{\text{factor}}}{\nu} \left( 1 + \frac{\nu^2}{B_{\text{factor}}} \right)
$$

where $a_0, a_1, b_0, b_1, c_{\alpha}$ are **model parameters** based on mass definition, relaxation state, and method. This formulation accounts for the **mass variance, spectral index, and structure growth** in determining the halo concentration.


## Halo Mass Function

The halo mass function is defined as

$$
\dfrac{dn}{dM} = f(\sigma_{M})\dfrac{\bar{\rho}_{m}}{M}\dfrac{d\textrm{ln}\sigma^{-1}}{dM}
$$

where $n$ is the number density of halos of a give mass $M$ associated with the rms variance of the matter density field, $\sigma^{2}_{M}$, at a given redshift. $f$ is a fitting function. The most important function is to precompute $\sigma_{M}$ as function of mass and scale factor. To initialise a mass function, we can simply do

```python
from jax_cosmo.halos.hmfunc import JAXMassFuncAngulo12

jax_mf = JAXMassFuncAngulo12()
```

---

A list of the following mass functions are supported.

### Angulo et al. (2012)
The expression as derived by [Angulo et al. (2012)](https://arxiv.org/pdf/1203.3216v1) is

$$
f(\sigma_{M}) = 0.201 \times \left(\dfrac{2.08}{\sigma_{M}}\right)^{1.7}\textrm{exp}\left(\dfrac{-1.172}{\sigma^{2}_{M}}\right)
$$

where $\sigma_{M}$ is the variance of the linear density field within a top-hat filter containing mass $M$.


### Bocquet et al. (2016)
As described by [Bocquet et al. (2016)](https://arxiv.org/pdf/1502.07357), the halo mass function is given by:

$$
f(\sigma_{M}) = A \left( \left( \frac{\sigma_{M}}{b} \right)^{-a} + 1 \right) \exp\left(-\frac{c}{\sigma_{M}^{2}}\right)
$$

where:

- $\sigma_{M}$ is the standard deviation of the linear density field at mass scale $M$.
- $A, a, b, c$ are redshift-dependent parameters.

**Redshift Dependence of Parameters**

The parameters evolve with redshift as:

$$
A = A_0 \cdot (1+z)^{A_z}
$$

$$
a = a_0 \cdot (1+z)^{a_z}
$$

$$
b = b_0 \cdot (1+z)^{b_z}
$$

$$
c = c_0 \cdot (1+z)^{c_z}
$$

where $(A_0, a_0, b_0, c_0, A_z, a_z, b_z, c_z)$ depend on the overdensity definition. See Table 2 in [Bocquet et al. (2016)](https://arxiv.org/pdf/1502.07357) for the fitted parameters.

**Converting $M_{200c}$ to $M_{200m}$**

The correction factors are:

$$
\gamma = \gamma_0 + \gamma_1 \exp\left(-\left(\frac{\gamma_2 - z}{\gamma_3}\right)^2\right)
$$

$$
\delta = \delta_0 + \delta_1 z
$$

where:

$$
\gamma_0 = 3.54 \times 10^{-2} + \Omega_m^{0.09},
$$

$$
\gamma_1 = 4.56 \times 10^{-2} + \frac{2.68 \times 10^{-2}}{\Omega_m}
$$

$$
\gamma_2 = 0.721 + \frac{3.50 \times 10^{-2}}{\Omega_m}
$$

$$
\gamma_3 = 0.628 + \frac{0.164}{\Omega_m}
$$

$$
\delta_0 = -1.67 \times 10^{-2} + 2.18 \times 10^{-2} \Omega_m
$$

$$
\delta_1 = 6.52 \times 10^{-3} - 6.86 \times 10^{-3} \Omega_m
$$

Applying this correction:

$$
f(\sigma) \to f(\sigma) \cdot (\gamma + \delta \log M)
$$

**Converting $M_{500c}$ to $M_{200m}$**

The correction factors are:

$$
\alpha = \alpha_0 \frac{\alpha_1 z + \alpha_2}{z + \alpha_2}
$$

$$
\beta = -1.7 \times 10^{-2} + 3.74 \times 10^{-3} \Omega_m
$$

where:

$$
\alpha_0 = 0.880 + 0.329 \Omega_m
$$

$$
\alpha_1 = 1.00 + \frac{4.31 \times 10^{-2}}{\Omega_m}
$$

$$
\alpha_2 = -0.365 + \frac{0.254}{\Omega_m}
$$

Applying this correction:

$$
f(\sigma) \to f(\sigma) \cdot (\alpha + \beta \log M)
$$


### Despali et al. (2016)

The function $f(\sigma_{M})$ is computed as:

$$
f(\sigma_{M}) = 2 A(x) \sqrt{\frac{\nu'}{2\pi}} \exp\left(-\frac{\nu'}{2} \right) \left(1 + \nu'^{-p(x)}\right)
$$

where:

- $\nu'$ is the ellipsoidal collapse correction factor:

$$
\nu' = a(x) \left(\frac{\delta_c}{\sigma_{M}}\right)^2
$$

- $\delta_c$ is the critical density threshold for collapse, computed using the [Nakamura & Suto (1997)](https://arxiv.org/pdf/astro-ph/9612074) prescription.
- $\sigma_{M}$ is the variance of the linear density field at mass $M$.
- $A(x)$, $a(x)$, and $p(x)$ are polynomial functions of the logarithmic ratio of halo overdensity to virial overdensity:

$$
x = \log_{10} \left(\frac{\Delta}{\Delta_{\text{virial}}} \right)
$$

The polynomial functions are given by:

$$
A(x) = A_1 x + A_0
$$

$$
a(x) = a_2 x^2 + a_1 x + a_0
$$

$$
p(x) = p_2 x^2 + p_1 x + p_0
$$

The coefficients $A_0, A_1, a_0, a_1, a_2, p_0, p_1, p_2$ depend on whether ellipsoidal collapse corrections are applied.

**Ellipsoidal collapse corrections**

  - $A_0 = 0.3953$, $A_1 = -0.1768$
  - $a_0 = 0.7057$, $a_1 = 0.2125$, $a_2 = 0.3268$
  - $p_0 = 0.2206$, $p_1 = 0.1937$, $p_2 = -0.04570$

**No ellipsoidal collapse corrections**

  - $A_0 = 0.3292$, $A_1 = -0.1362$
  - $a_0 = 0.7665$, $a_1 = 0.2263$, $a_2 = 0.4332$
  - $p_0 = 0.2488$, $p_1 = 0.2554$, $p_2 = -0.1151$


### Tinker et al. (2008)

The multiplicity function $f(\sigma_{M})$ in the [Tinker et al. (2008)](https://arxiv.org/pdf/0803.2706) mass function is given by:

$$
f(\sigma_{M}) = \alpha \left( \left(\frac{\phi}{\sigma_{M}}\right)^\beta + 1 \right) \exp\left( -\frac{\Phi}{\sigma_{M}^{2}} \right)
$$

where:

- $\sigma_{M}$ is the variance of the density field on the mass scale.
- $\Delta$ is the matter overdensity corresponding to a given mass definition.
- $\hat{\delta}\equiv\log_{10} \Delta$ is used for interpolating fitting parameters.

The fitting parameters are interpolated based on the logarithm of the overdensity:

$$
\alpha = \alpha(\hat{\delta}) \cdot a^{0.14}
$$

$$
\beta = \beta(\hat{\delta}) \cdot a^{0.06}
$$

$$
\gamma = 10^{-\left(\frac{0.75}{\hat{\delta} - 1.875}\right)^{1.2}}
$$

$$
\phi = \gamma(\hat{\delta}) \cdot a^\gamma
$$

$$
\Phi = \Phi(\hat{\delta})
$$

where:

- $a$ is the scale factor.
- $\alpha, \beta, \gamma, \Phi$ are interpolated from tabulated values in the original paper.

### Tinker et al. (2010)

The function $f(\sigma_{M})$ in the [Tinker et al. (2010)](https://arxiv.org/pdf/1001.3162) mass function is given by:

$$
f(\sigma_{M}) = \nu f(\nu)
$$

where the functional form of $f(\nu)$ is:

$$
f(\nu) = A \left( 1 + ( \beta \nu )^{-2\phi} \right) \nu^{2\alpha} \exp \left( -\frac{\gamma \nu^2}{2} \right)
$$

**Definitions of Parameters**:

- $\nu$ is the peak height, defined as:

$$
\nu = \frac{\delta_c}{\sigma_{M}}
$$

where $\delta_c$ is the critical density for spherical collapse, and $\sigma_{M}$ is the variance of the density field.

- The overdensity $\Delta$ is converted to the matter overdensity $\Delta_m$, and its logarithm is used for interpolating fitting parameters:

$$
\hat{\delta}\equiv\log_{10} \Delta_m
$$

- The fitting parameters are interpolated based on the logarithm of the overdensity:

$$
\alpha = \eta(\hat{\delta}) \cdot a^{-0.27}
$$

$$
\beta = \beta(\hat{\delta}) \cdot a^{-0.20}
$$

$$
\gamma = \gamma(\hat{\delta}) \cdot a^{0.01}
$$

$$
\phi = \phi(\hat{\delta}) \cdot a^{0.08}
$$

$$
A = \alpha(\hat{\delta})
$$

where $a$ is the scale factor.

**Redshift Evolution Correction**

If redshift evolution normalization is applied, the amplitude $A$ is corrected by:

$$
A \rightarrow A \exp \left( z ( p + q z ) \right)
$$

where:

- $z = \frac{1}{a} - 1$ is the redshift,
- $p = p(\hat{\delta})$,
- $q = q(\hat{\delta})$.

**Final Expression**

$$
f(\sigma_{M}) = \nu A \left( 1 + (\beta \nu)^{-2\phi} \right) \nu^{2\alpha} \exp \left( -\frac{\gamma \nu^2}{2} \right)
$$


### Watson et al. (2013)

The [Watson et al. (2013)](https://arxiv.org/pdf/1212.0095) mass function provides separate parameterizations for **Friends-of-Friends (FoF)** and **Spherical Overdensity (S.O.)** mass definitions.

**Friends-of-Friends (FoF) Mass Function**

For the FoF mass function, the function $f(\sigma_{M})$ is given by:

$$
f(\sigma_{M}) = A \left( \left(\frac{b}{\sigma_{M}}\right)^a + 1 \right) \exp\left(-\frac{c}{\sigma_{M}^{2}} \right)
$$

where the fitting parameters are:

- $A = 0.282$
- $a = 2.163$
- $b = 1.406$
- $c = 1.210$

**Spherical Overdensity (S.O.) Mass Function**

For the S.O. mass function, the parameters evolve with redshift, and an additional correction factor is applied for overdensities other than $178$. The base mass function is:

$$
f(\sigma_{M}) = A \left( \left(\frac{b}{\sigma_{M}}\right)^a + 1 \right) \exp\left(-\frac{c}{\sigma_{M}^{2}} \right)
$$

where the parameters are:

For $z = 0$ ($a = 1$):

- $A = 0.194$
- $a = 1.805$
- $b = 2.267$
- $c = 1.287$

For $z > 6$ ($a < \frac{1}{7}$):

- $A = 0.563$
- $a = 3.810$
- $b = 0.874$
- $c = 1.453$

For intermediate redshifts:

$$
A = \Omega_m(a) \left( 1.097 a^{3.216} + 0.074 \right)
$$

$$
a = \Omega_m(a) \left( 5.907 a^{3.058} + 2.349 \right)
$$

$$
b = \Omega_m(a) \left( 3.136 a^{3.599} + 2.344 \right)
$$

$$
c = 1.318
$$

where $\Omega_m(a)$ is the matter density fraction at scale factor $a$.

**Overdensity Correction Factor**

For overdensities $\Delta \neq 178$, a correction factor $\Gamma$ is applied:

$$
\Gamma = \exp\left[0.023 (\hat{\Delta} - 1) \right] \Delta^d \exp\left( \frac{0.072 (1 - \hat{\Delta})}{\sigma_{M}^{2.130}} \right)
$$

where $\hat{\Delta} \equiv\frac{\Delta}{178}$ and

$$
d = -0.456 \Omega_m(a) - 0.139
$$

The final S.O. mass function is:

$$
f(\sigma_{M}) = f_{\text{base}}(\sigma_{M}) \times \Gamma
$$

where $f_{\text{base}}(\sigma_{M})$ is the base mass function.

**Final Expression**

For a given overdensity definition:

$$
f(\sigma_{M}) =
\begin{cases}
f_{\text{FoF}}(\sigma_{M}) & \text{if FoF mass definition} \\
f_{\text{SO}}(\sigma_{M}) & \text{if S.O. mass definition}
\end{cases}
$$

This function models the fraction of mass collapsed into halos at a given mass scale.


### Sheth et al. (1999)

The [Sheth et al. (1999)](https://arxiv.org/pdf/astro-ph/9901122) mass function is given by:

$$
f(\sigma_{M}) = \nu A \left( 1 + (\lambda \nu^2)^{-p} \right) \exp\left(-\frac{\lambda \nu^2}{2}\right)
$$

where:

- $\nu$ is the peak height:

$$
\nu = \frac{\delta_c}{\sigma_{M}}
$$

- $\delta_c$ is the critical overdensity for collapse.
- $\sigma_{M}$ is the mass variance.
- $A = 0.21615998645$ is the normalization constant.
- $p = 0.3$ is the slope parameter.
- $\lambda = 0.707$ is the scaling factor.

If `use_custom_delta_c` is set to `True`, then $\delta_c$ follows the fit from [Nakamura & Suto (1997)](https://arxiv.org/pdf/astro-ph/9612074). Otherwise, it uses the Einstein-de Sitter (EdS) approximation.

### Press & Schechter (1974)

The [Press & Schechter (1974)](https://articles.adsabs.harvard.edu/pdf/1974ApJ...187..425P) mass function is given by:

$$
f(\sigma_{M}) = \frac{\nu}{\sqrt{2 \pi}} \exp\left(-\frac{\nu^2}{2}\right)
$$

where:

- $\nu$ is the peak height, defined as:

$$
\nu = \frac{\delta_c}{\sigma_{M}}
$$

- $\delta_c$ is the critical overdensity for collapse, which can be obtained from the Einstein-de Sitter (EdS) approximation.

### Jenkins et al. (2001)

The [Jenkins et al. (2001)](https://arxiv.org/pdf/astro-ph/0005260) mass function is given by:

$$
f(\sigma_{M}) = A \cdot \exp\left( - \left| \log(\sigma_{M}) + \gamma \right|^\alpha \right)
$$

where:

- $A = 0.315$ is the amplitude.
- $\gamma = 0.61$ is the shape parameter.
- $\alpha = 3.8$ is the steepness parameter.
- $\sigma_{M}$ is the standard deviation of the linear density field.

## Halo Bias Function

Similar to CCL, we implement the following four halo bias functions:

- Sheth & Tormen (1999)
- Sheth et al. (2001)
- Tinker et al. (2010)
- Bhattacharya et al. (2011)

### Sheth & Tormen (1999)
The halo bias equation used in the [Sheth & Tormen (1999)](https://arxiv.org/pdf/astro-ph/9901122) model is given by:

$$
b(\nu) = 1 + \dfrac{a\nu^{2}-1+B(\nu,a)}{\delta_{c}}
$$

where

$$
B(\nu,a) = \dfrac{2p}{1+(a\nu^{2})^{p}}
$$

- $\nu=\frac{\delta_{c}}{\sigma_{M}}$ is the peak height,
- $\delta_{c}$ is the critical density contrast for spherical collapse,
- $\sigma_{M}$ is the variance of mass fluctuations,
- $a=0.707$ is a scaling factor and
- $p=0.3$ is a power-law index.

### Sheth et al. (2001)

The halo bias equation used in the [Sheth et al. (2001)](https://arxiv.org/pdf/astro-ph/9907024) model is given by:

$$
b(\nu) = 1 + \dfrac{\sqrt{a\nu^{2}}P(\nu,a) - Q(\nu,a)}{\sqrt{a}\delta_{c}}
$$

where

$$
P(a,\nu) = 1 + \dfrac{b}{(a\nu^{2})^{c}}
$$

and

$$
Q(a,\nu) = \dfrac{(a\nu^{2})^{c}}{(a\nu^{2})^{c}+b(1-c)(1-0.5c)}
$$

- $\nu=\frac{\delta_{c}}{\sigma_{M}}$ is the peak height,
- $\delta_{c}$ is the critical density contrast for spherical collapse,
- $\sigma_{M}$ is the mass variance,
- $a=0.707$ is a scaling factor,
- $b=0.5$ and $c=0.6$ are fitting parameters.

### Tinker et al. (2010)
The halo bias formula from [Tinker et al. 2010](https://arxiv.org/pdf/1001.3162) is based on the following equation:

$$
b(\nu) = 1 - A\dfrac{\nu^{a}}{\nu^{a}+\delta_{c}^{a}} + B\nu^{b} + C\nu^{c}
$$

where

- $\delta_{c}$ is the critical density contrast,
- $\sigma_{M}$ is the mass variance for the halo of mass $M$,
- $A$, $B$, $C$, $a$, $b$, and $c$ are parameters derived from fitting to simulations. See Table 2 in the paper for the definitions of these quantities.

### Bhattacharya et al. (2011)
The halo bias equation used in Bhattacharya et al. (2011) model is given by

$$
b(\nu,a) = 1 + \dfrac{A(a)\nu^{2}-q+B(\nu,a)}{\delta_{c}}
$$

where
- $\nu = \frac{\delta_{c}}{\sigma_{M}}$ is the peak height,
- $\delta_{c}$ is the critical density contrast for spherical collapse,
- $\sigma_{M}$ is the variance of mass functions,
- $A(a)$ is the scale dependent factor:
$$
A(a) = A_{0}a^{A_{z}}
$$
with $A_{0}=0.788$ and $A_{z}=0.01$.
- the quantity $B(\nu, a)$ is given by:

$$
B(\nu,a) = \frac{2p}{1+(A(a)\nu^{2})^{p}}
$$

- $p=0.807$ and $q=1.795$ are fitting parameters.

See Equation 18 in the [paper](https://arxiv.org/pdf/1005.2239).



---
::: jax_cosmo.halos.hmbase.sigmaR
::: jax_cosmo.halos.hmbase.sigmaM
::: jax_cosmo.halos.hmbase.d_sigmaM_dlogM
::: jax_cosmo.halos.hmbase.compute_sigma

<!-- ::: jax_cosmo.halos.hmfunc.JAXMassFuncAngulo12
::: jax_cosmo.halos.hmfunc.JAXMassFuncBocquet16
::: jax_cosmo.halos.hmfunc.JAXMassFuncDespali16
::: jax_cosmo.halos.hmfunc.JAXMassFuncTinker08
::: jax_cosmo.halos.hmfunc.JAXMassFuncTinker10
::: jax_cosmo.halos.hmfunc.JAXMassFuncWatson13
::: jax_cosmo.halos.hmfunc.JAXMassFuncSheth99
::: jax_cosmo.halos.hmfunc.JAXMassFuncPress74
::: jax_cosmo.halos.hmfunc.JAXMassFuncJenkins01 -->
