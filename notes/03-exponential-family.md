# 03 · The Gaussian in exponential family form

**Source:** Bayesian Series Part 1 §6
**Implements:** `core/distributions.py` — `Gaussian`
**Handwritten working:** [scans/03-exponential-family.pdf](scans/03-exponential-family.pdf)

## Result

The 1-D Gaussian is a member of the exponential family. In canonical form

$$p(x \mid \theta) = h(x)\exp\!\big(\eta(\theta)^\top T(x) - A(\theta)\big)$$

it has

$$
T(x) = \begin{bmatrix} x \\ x^2 \end{bmatrix}, \qquad
\eta(\theta) = \begin{bmatrix} \dfrac{\mu}{\sigma^2} \\ -\dfrac{1}{2\sigma^2} \end{bmatrix}, \qquad
A(\theta) = \frac{\mu^2}{2\sigma^2} + \tfrac{1}{2}\ln(2\pi\sigma^2), \qquad
h(x) = 1 .
$$

## Derivation

**1 · The standard density**

$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**2 · Expand the quadratic exponent**, separating terms by their power of $x$:

$$-\frac{(x-\mu)^2}{2\sigma^2} = -\frac{x^2 - 2\mu x + \mu^2}{2\sigma^2}
= \left(\frac{\mu}{\sigma^2}\right)x + \left(-\frac{1}{2\sigma^2}\right)x^2 - \frac{\mu^2}{2\sigma^2}$$

**3 · Move the normalising factor inside the exponential:**

$$\frac{1}{\sqrt{2\pi\sigma^2}} = \exp\!\left(-\tfrac{1}{2}\ln(2\pi\sigma^2)\right)$$

Combining,

$$p(x \mid \mu, \sigma^2) = \exp\Big(
\underbrace{\left(\tfrac{\mu}{\sigma^2}\right)x + \left(-\tfrac{1}{2\sigma^2}\right)x^2}_{\eta(\theta)^\top T(x)}
- \underbrace{\left[\tfrac{\mu^2}{2\sigma^2} + \tfrac{1}{2}\ln(2\pi\sigma^2)\right]}_{A(\theta)}
\Big)$$

**4 · Match terms** against the canonical form to read off $T$, $\eta$, $A$, $h$ as stated above.

In natural parameters, substituting $\sigma^2 = -1/(2\eta_2)$ and $\mu = -\eta_1/(2\eta_2)$:

$$A(\eta) = -\frac{\eta_1^2}{4\eta_2} + \tfrac{1}{2}\ln\!\left(-\frac{\pi}{\eta_2}\right)$$

## Why it matters

**Sufficient statistics.** The data enters only through $T(x)$. To infer $\mu$ and
$\sigma^2$ from a stream, you need two running totals — $\sum x_i$ and $\sum x_i^2$ —
and can discard the raw observations without losing information. This is what makes
online updating possible at constant memory.

**Natural parameters add.** Because the data appears solely inside the inner product
$\eta^\top T(x)$, multiplying a Gaussian prior by a Gaussian likelihood adds their
natural parameters. Since $\eta_2 = -\tfrac{1}{2\sigma^2}$, adding natural parameters
*is* adding precisions — which is the algebraic reason behind the precision-weighted
mean update in note 02, and behind the Kalman measurement update in Stage 3.

**The log-partition generates moments.** $A(\eta)$ is a cumulant generating function:
differentiating it with respect to $\eta$ returns $\mathbb{E}[T(x)]$.

## Checks

- $\partial A/\partial \eta_1 = \mathbb{E}[x] = \mu$ — verify numerically by finite
  differences against the analytic mean
- $\partial^2 A/\partial \eta_1^2 = \mathrm{Var}[x] = \sigma^2$
- round-trip $(\mu, \sigma^2) \to \eta \to (\mu, \sigma^2)$ is the identity
- the canonical form evaluates to the same density as `scipy.stats.norm.pdf` across a
  grid of $x$
- combining two Gaussians by adding natural parameters gives the same posterior as
  the precision-weighted update in note 02
