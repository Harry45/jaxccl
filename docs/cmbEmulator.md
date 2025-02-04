---
title: CMB Emulator
icon: material/rocket-outline
---
We also build an emulator for the CMB power spectra (TT, TE and EE), up to $\ell=2500$. The priors of the cosmological parameters are roughly centred on the recent Planck 2018 results and are as follows:

| \(\boldsymbol{\theta}\) | Distribution | Minimum | Scale  | Fiducial |
|--------------------------------------------|-------------------|--------------------------|--------------------------|--------------------------------|
| \(\sigma_8\)                               | Uniform           | 0.7                      | 0.2                      | 0.8                            |
| \(\Omega_{\text{cdm}}\)                    | Uniform           | 0.20                     | 0.15                     | 0.2                            |
| \(\Omega_b\)                               | Uniform           | 0.04                    | 0.02                    | 0.04                           |
| \(h\)                                      | Uniform           | 0.62                     | 0.12                     | 0.7                            |
| \(n_s\)                                    | Uniform           | 0.90                     | 0.2                      | 1.0                            |


A minimal example of how we can use the CMB emulator is as follows:

```python
from jax_cosmo.emulator import EMUCMBdata, prediction_cmb_cls
from cmbrun.cmbcls import get_config

cfg = get_config('planck')
emudata = EMUCMBdata(cfg)
cosmology = np.array([0.8120, 0.265, 0.04938, 0.6732, 0.96605])

# using the emulator to predict the power spectra
mean_tt_jax = prediction_cmb_cls(cosmology, emudata, 'tt')
mean_te_jax = prediction_cmb_cls(cosmology, emudata, 'te')
mean_ee_jax = prediction_cmb_cls(cosmology, emudata, 'ee')
```
