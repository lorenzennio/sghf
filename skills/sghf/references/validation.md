# Validating an SGHF result

SGHF compiles a workspace into a **linear response with a quadratic Gaussian
objective**. It does not evaluate the HistFactory likelihood. Every number it
produces is an approximation whose error is unknown until measured.

The library reports no approximation error. Nothing in `FitResult`,
`HypothesisTestResult` or `LimitResult` tells you how far the answer is from
pyhf's. That has to be established separately.

## The gate

**Do not present an SGHF number as a final scientific result without a pyhf
cross-check of that specific number.** Fast iteration, scans, sensitivity
studies and exploration are what SGHF is for. The published limit, significance
or interval is not.

State plainly which one you are producing. "The 95% CL upper limit is 5.98"
claims more than "SGHF gives 5.98; this needs a pyhf cross-check before it goes
in a paper".

## Cheap checks, in order

Run these first; they cost seconds and catch most trouble.

**1. What compiled.**

```python
import numpy as np
import sghf

model = sghf.load("workspace.json")
print(model.approximation)     # modifier types, parameters, warnings
```

Non-empty `warnings` means something nonstandard was accepted.

**2. Boundaries and pulls.**

```python
fit = model.fit()
print("at bound:", fit.at_bound)
print("max |pull|:", np.nanmax(np.abs(fit.pulls)))
print("converged:", fit.converged)
```

Anything in `at_bound` invalidates the Gaussian error for that parameter. Large
pulls mean the fit is far from the expansion point, which is exactly where a
linearization degrades.

**3. Nominal versus iterated.**

```python
nominal = model.fit(variance="nominal")
iterated = model.fit(variance="iterated")
```

The two differ only in how the diagonal variance is chosen. A material shift
between them is a direct signal that the quadratic approximation is being
pushed, and it must be reported. This is the single cheapest diagnostic
available.

**4. Toy resolution**, whenever an empirical p-value is involved. A p-value at
`1/(ntoys+1)` is a bound, not a measurement. Repeat with a different `rng` and
see whether the answer moves.

## The pyhf cross-check

pyhf is a test-time dependency of this project and is not needed at runtime.
Install it separately for validation.

```python
import pyhf, json

spec = json.load(open("workspace.json"))
workspace = pyhf.Workspace(spec)
exact_model = workspace.model()
data = workspace.data(exact_model)

exact = pyhf.infer.hypotest(5.98, data, exact_model, test_stat="qtilde")
```

Compare the quantity you actually intend to report. Agreement of `mu_hat` does
not imply agreement of a tail probability, and a CLs value far from the
crossing says little about the limit.

### Budget it: cross-check points, not scans

**A full pyhf limit scan on a large workspace is not feasible.** Measured on a
232-parameter model with the NumPy backend and pyhf's default minimizer: a
single `pyhf.infer.hypotest` took **274 seconds**, and a 41-point observed plus
expected scan produced no result before it was abandoned. SGHF's equivalent
calls are sub-second. That ratio is why SGHF exists, and it is also why "just
run it in pyhf" is not a plan.

Pick two or three points and check those:

1. the reported limit itself — the number going in the paper
2. the median expected limit
3. optionally one point in the bulk, as a sanity check

Do not bother cross-checking a point below `mu_hat`: SGHF's toy CLs is
degenerate there by construction (see [pitfalls.md](pitfalls.md)), so the
comparison tests nothing about the limit.

To make more points affordable, switch pyhf to a faster backend and minimizer.
Both are optional pyhf extras and are **not** installed by default —
`python -m pip install 'pyhf[jax,minuit]'` first, or this raises
`ModuleNotFoundError`:

```python
pyhf.set_backend("jax", pyhf.optimize.minuit_optimizer(verbose=0))
```

Note that pyhf 0.7.6's **toy** calculator is incompatible with NumPy 2
(`np.percentile(interpolation=)` was removed), so its asymptotic path may be the
only one available to you.

## Escalation

| Observation | Minimum response |
|---|---|
| nominal or derivative mismatch against pyhf | do not use the result until the translation is understood |
| material curvature over the fitted or tested range | use pyhf for the affected result |
| influential asymmetric modifier with large displacement | compare the complete test or limit with pyhf |
| sparse or far-tail result | calibrate with toys and compare statistic ordering with pyhf |
| active POI or nuisance bound | prefer toy or pyhf validation; the Gaussian error is undefined there |
| `ModelDomainError`, or a fitted yield near zero | the linear model is outside its domain; do not extrapolate |
| rank failure or unstable covariance | reparameterize, fix the degeneracy, or use pyhf |
| material nominal-versus-iterated shift | report the sensitivity and validate against pyhf |
| empirical result near the toy resolution floor | raise `ntoys` and check repeat-seed stability |

## What local agreement does not establish

Matching nominal yields and response slopes against pyhf is necessary, not
sufficient. It does **not** establish agreement of nonlinear yields away from
the expansion point, equality of Poisson and Gaussian auxiliary constraints,
equality of the quadratic objective and pyhf's likelihood ratio, regularity at
boundaries, asymptotic or empirical coverage, or limit agreement over a scan far
from the expansion point.

Full treatment:
https://github.com/lorenzennio/sghf/blob/main/docs/validation-and-applicability.md
