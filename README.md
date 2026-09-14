# Active Digital Twin

Probabilistic inference, active inference, and digital twins — built from first
principles with the derivations alongside the code.

> **Status:** Stage 1 (inference core) in progress. Nothing here is stable yet.

---

## Why this exists

Generative models are fluent but ungrounded. In any domain where a recommendation has
to be *defended* — a regulator asking why a forecast is calibrated, an engineer asking
why the system chose to inspect rather than act — the useful architecture treats the
model as an interface to a probabilistic backend that carries its own uncertainty.

This repository builds that backend from the bottom up: exact inference over discrete
graphs, then approximate inference when exactness stops scaling, then an agent that
selects actions by expected free energy, then a twin that assimilates observations and
reports calibrated forecasts with coverage guarantees.

Everything is implemented from scratch. The point is not to compete with PyMC or
pgmpy — it is to understand, and to be able to show the understanding. Each module has
a corresponding derivation in [`notes/`](notes/).

---

## Stages

| Stage | Contents | Status |
| --- | --- | --- |
| **1 · Inference core** | Factors, discrete Bayesian networks, d-separation, exact inference (enumeration, variable elimination, belief propagation), MCMC, mean-field VI | in progress |
| **2 · Active inference** | Discrete generative model (A, B, C, D), variational state estimation, expected free energy, policy selection | planned |
| **3 · Digital twin** | State-space models, Kalman and particle filters, data assimilation, calibration, conformal prediction, V&V | planned |

---

## The running example

A **machine health twin**: a component with a hidden degradation state, noisy sensor
readings, and a choice each step between inspecting (buying information) and
intervening (acting on belief).

This single example carries through all three stages — a diagnosis network in Stage 1,
an agent deciding when to inspect in Stage 2, and a twin tracking degradation with
calibrated remaining-life forecasts in Stage 3.

The structure is deliberately domain-general. Hidden state inferred from noisy
indicators, with a test-or-treat decision under uncertainty, is the same problem
whether the subject is a bearing or a patient; only the variable names change. All data
here is synthetic.

---

## Layout

```
src/adt/
  core/         factors, distributions, log-space arithmetic, DAGs
  inference/    exact (enumeration, VE, BP) and approximate (MCMC, CAVI)
  agent/        generative model, expected free energy, policy selection
  twin/         filters, assimilation, calibration, conformal prediction
  viz/          graphs, posteriors, reliability diagrams, traces

tests/          mirrors src/ — correctness against closed-form results
examples/       runnable end-to-end notebooks
notes/          derivations underpinning each module
experiments/    scratch work; not maintained
```

---

## On correctness

A probabilistic library is only worth anything if its numbers are right, so
correctness here is demonstrated rather than asserted:

- `inference/exact/enumeration.py` is a deliberately naive brute-force implementation.
  It is slow and obviously correct, and every faster algorithm is tested against it.
- Conjugate updates are checked against their analytic posteriors.
- The CAVI implementation asserts the ELBO increases monotonically, and converges to
  the closed-form conjugate solution where one exists.
- Conformal predictors are checked for empirical coverage at the nominal rate.

---

## Notes

Derivations live in [`notes/`](notes/), one per topic, each stating a result, deriving
it, and linking to the module that implements it. They are worked by hand first and
written up afterwards — the implementation follows the derivation, not the reverse.

`notes/tutorials/` contains worked exercises following the official
[pymdp](https://github.com/infer-actively/pymdp) tutorials by the Infer Actions Lab.
That material is the work of the pymdp authors and retains their licence; the notes on
it are mine.

---

## Licence

MIT. See [LICENSE](LICENSE).
