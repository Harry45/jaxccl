---
title: Growth Factor and Growth Rate
icon: material/telescope
---

We will follow the explanation in the CCL paper ([Chisari et al. 2019](https://arxiv.org/abs/1812.05995)) to compute the growth rate and the growth factor. In short, we want to solve the following differential equation:

$$
\dfrac{d}{da}\left(a^{3}H(a)\dfrac{dD}{da}\right) = \dfrac{3}{2}\Omega_{m}(a)aH(a)D(a)
$$

where

- $a$ is the scale factor,
- $H(a)$ is the Hubble parameter at scale factor $a$,
- $D(a)$ is the linear growth factor of matter perturbation and
- $\Omega_{m}(a)$ is the matter density at scale factor $a$.

Following the CCL paper, we define $D(a) = ag(a)$ and hence, $D'(a) = ag'(a) + g(a)$. The notation $'$ denotes the derivative of the quantity with respect to $a$. Working through the maths, we can show:

$$
g''(a) + \left(\dfrac{5}{a}+\dfrac{H'(a)}{H(a)}\right)g'(a) - \dfrac{1}{a^{2}}\left(\dfrac{3}{2}\Omega_{m}(a)-a\dfrac{H'(a)}{H(a)}-3\right)g(a) = 0
$$

We can also write the above expression in terms of $E(a)$, that is,

$$
\dfrac{H'(a)}{H(a)} = \dfrac{E'(a)}{E(a)}
$$

Next, we define $\alpha(a)$ and $\beta(a)$ as follows:

$$
\begin{split}
\alpha(a) &=\dfrac{5}{a} + \dfrac{H'(a)}{H(a)} \\
\beta(a) &=-\dfrac{1}{a^{2}}\left(\dfrac{3}{2}\Omega_{m}(a) - a\dfrac{H'(a)}{H(a)}-3\right)
\end{split}
$$

and we now have

$$
g''(a) + \alpha(a)g'(a) + \beta(a)g(a) = 0.
$$

At very high redshift, $g(a=0)=1$ and $g'(a=0)=0$. If we define:

$$
\boldsymbol{y} =\left(\begin{array}{c}
y_{0}(a)\\
y_{1}(a)\\
\end{array}\right)=
\left(\begin{array}{c}
g(a)\\
g'(a)\\
\end{array}\right)
$$

and

$$
\boldsymbol{y}' =\left(\begin{array}{c}
y'_{0}(a)\\
y'_{1}(a)\\
\end{array}\right)=\left(\begin{array}{cc}
0 & 1\\
-\beta(a) & -\alpha(a)
\end{array}\right)\,\left(\begin{array}{c}
y_{0}(a)\\
y_{1}(a)
\end{array}\right)
$$

The above differential equation can then be solved used various numerical methods such as the Euler's method, Runge-Kutta $4^{th}$ order method, Cash-Karp method and others. Once we have $g(a)$, we can compute:

- the **normalised linear growth factor**:

$$
D(a) = \dfrac{ag(a)}{g(a=1)}
$$

- the **logarithmic growth rate**:

$$
f(a)\equiv \dfrac{d\,\textrm{ln}D}{d\,\textrm{ln}a}=\dfrac{aD'}{D} = 1+a\dfrac{g'(a)}{g(a)}
$$
