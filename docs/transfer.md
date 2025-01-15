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