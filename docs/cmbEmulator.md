---
title: CMB Emulator
icon: material/rocket-outline
---

A minimal example of how we can use the CMB emulator is as follows:

```python
from jax_cosmo.emulator import EMUCMBdata, prediction_cmb_cls
from cmbrun.cmbcls import get_config

emudata = EMUCMBdata()
cfg = get_config('planck')
cosmology = np.array([0.8120, 0.265, 0.04938, 0.6732, 0.96605])

# using the emulator to predict the power spectra
mean_tt_jax = prediction_cmb_cls(cosmology, emudata, 'tt')
mean_te_jax = prediction_cmb_cls(cosmology, emudata, 'te')
mean_ee_jax = prediction_cmb_cls(cosmology, emudata, 'ee')
```
