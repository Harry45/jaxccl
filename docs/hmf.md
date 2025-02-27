---
title: Halo Mass Function
icon: material/orbit
---
The most important function is to precompute $\sigma_{M}$ as function of mass and scale factor. A list of the following mass functions are supported:

- Angulo12
- Bocquet16
- Despali16
- Tinker08
- Tinker10
- Watson13
- Sheth99
- Press74
- Jenkins01

To initialise a mass function, we can simply do

```python
from jax_cosmo.halos.hmfunc import JAXMassFuncAngulo12

jax_mf = JAXMassFuncAngulo12()
```

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