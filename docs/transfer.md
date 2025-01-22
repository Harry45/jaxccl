---
title: Transfer Function
icon: material/chart-donut
---
In this section, we briefly discuss the methods implemented for computing the transfer function. In particular, in addition to the existing EH method, we have added the BBKS method in the code. See [Chisari et al. 2019](https://arxiv.org/abs/1812.05995) for further details.

The analytical expression for the BBKS approximation to the transfer function is given by:

$$
T(q) = \dfrac{\textrm{ln}[1+2.34q]}{2.34q}[1+3.89q+(16.2q)^{2}+(5.47q)^{3}+(6.71q)^{4}]^{-0.25}
$$

where $q$ is measured in $\textrm{Mpc}^{-1}$ and is given by:

$$
q\equiv \dfrac{k}{\Omega_{m}h^{2}e^{-\Omega_{b}[1+\sqrt{2h}/\Omega_{m}]}}
$$

**Code**

::: jax_cosmo.transfer.BBKS

# Emulator

We also have an emulator build to compute the linear matter power spectrum directly. It takes advantage of the fact that the linear matter power spectrum at any redshift can be computed as the product of the growth factor and the linear matter power spectrum at a fixed redshift, $z_{0}$, that is,

$$
P_{\text{lin}}(\boldsymbol{\theta},k,z) = A(\boldsymbol{\theta},z)\,P_{\textrm{lin}}(\boldsymbol{\theta},k,z_{0}).
$$

In this case, $\boldsymbol{\theta}$ is the set of cosmological parameters for the emulator,

$$
\boldsymbol{\theta} = [\sigma_{8},\,\Omega_{\text{cdm}},\,\Omega_{b},\,h,\,n_{s}]
$$

without the $h^{2}$ factor for $\Omega_{\text{cdm}}$ and $\Omega_{b}$. The emulator is built over the following prior range:


| \(\boldsymbol{\theta}\) | Distribution | Minimum | Scale  | Fiducial |
|--------------------------------------------|-------------------|--------------------------|--------------------------|--------------------------------|
| \(\sigma_8\)                               | Uniform           | 0.6                      | 0.4                      | 0.8                            |
| \(\Omega_{\text{cdm}}\)                    | Uniform           | 0.07                     | 0.43                     | 0.2                            |
| \(\Omega_b\)                               | Uniform           | 0.028                    | 0.027                    | 0.04                           |
| \(h\)                                      | Uniform           | 0.64                     | 0.18                     | 0.7                            |
| \(n_s\)                                    | Uniform           | 0.87                     | 0.2                      | 1.0                            |

Neutrino mass in fixed, that is, $\sum m_{\nu}=0.06\,\text{eV}$ and the settings for generating the training set using `classy` is as follows:

- $z_{\text{min}}=0.0$
- $z_{\text{max}}=3.0$
- $k_{\text{min}}=10^{-4}\;\text{Mpc}^{-1}$
- $k_{\text{max}}=50\;\text{Mpc}^{-1}$
- $\Omega_{k}=0$.

A minimal example of how we can use the emulator is as follows:

```python
from jax_cosmo.core import Cosmology
from jax_cosmo.power import linear_matter_power_emu

# define the cosmology
cosmo_jax = Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8 = 0.8, n_s=0.96,
                      Omega_k=0., w0=-1., wa=0., Neff = 3.044)

# compute the linear matter power spectrum
plin_emu = linear_matter_power_emu(cosmo, k=0.01, a=1.0)
```
