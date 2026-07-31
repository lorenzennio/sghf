---
name: sghf
description: Use when fitting a pyhf or HistFactory workspace, extracting a signal strength and its uncertainty, computing discovery significance or p0, running CLs exclusion tests, deriving upper limits with expected bands, or generating toy pseudo-experiments - especially when pyhf is too slow for the iteration you need. Also use when SGHF output looks wrong: CLs pinned at 1.0, a parameter uncertainty of exactly zero, toy p-values stuck at a floor, or the asymptotic and toy calculators disagreeing.
---

# SGHF

## Overview

SGHF compiles a pyhf workspace into a **linear response with a quadratic
Gaussian objective**, then reuses NumPy/SciPy matrices for fits, profile tests,
limits and vectorized pseudo-experiments. pyhf is not a runtime dependency.

**Core principle: it is fast because it is approximate.** It does not evaluate
the HistFactory likelihood, and it reports no approximation error. Use it to
iterate; validate against pyhf before reporting.

## When to use

- Fitting a HistFactory workspace, or scanning many of them
- Signal strength with uncertainties, nuisance pulls, correlations
- Discovery significance (`q0`), CLs exclusion (`q`, `qtilde`), upper limits
- Large toy ensembles, where per-toy fitting in pyhf is impractical

## When to use pyhf instead

- The number is going in a paper (cross-check it — see below)
- Low counts, strongly nonlinear or asymmetric modifiers, active bounds
- Far-tail results, or a modifier SGHF cannot compile

## Workflow

```python
import sghf

model = sghf.load("workspace.json")   # accepts path, mapping, or stream
print(model.approximation)            # what compiled; check .warnings

fit = model.fit()
print(fit.value(model.poi_name), fit.error(model.poi_name), fit.at_bound)

discovery = model.hypotest(0.0, test_stat="q0")     # .significance, .p0
exclusion = model.hypotest(1.0, test_stat="qtilde") # .cls, .expected_cls
limit = model.upper_limit(level=0.05)               # .observed, .expected
```

Everything is addressed **by name**, never by position. There is no
`sghf.infer.*`; pyhf-style calls raise `SGHFUsageError` naming the right method.

## Quick reference

| Task | Call |
|---|---|
| Compile | `sghf.load(src, measurement=..., strict=False)` |
| Inspect | `model.approximation`, `model.parameter_names`, `model.poi_name` |
| Fit | `model.fit(data=None, fixed=..., bounds=..., variance="nominal")` |
| Discovery | `model.hypotest(0.0, test_stat="q0")` |
| Exclusion | `model.hypotest(mu, test_stat="qtilde")` |
| Toy version of either | add `calculator="toys", ntoys=..., rng=...` |
| Upper limit | `model.upper_limit(level=0.05)` |
| Pseudo-data | `model.toys(size, parameters=fit, rng=...)` |

Expected tuples run `(-2σ, -1σ, median, +1σ, +2σ)`. Preconditions: `q0` requires
`poi_test=0`; `qtilde` requires the POI lower bound to be exactly zero.

## Behaviours that look like bugs

Read [references/pitfalls.md](references/pitfalls.md) before concluding
something is broken. The four that catch people:

- **`CLs == 1.0` for every μ below the best fit** is correct. The observed
  statistic is zero there and every toy satisfies `q >= 0`, so the tail
  fraction is 1. It does not affect the limit.
- **`error(name) == 0.0`** means the parameter is at a bound. Check
  `fit.at_bound`; the Gaussian error is undefined there, not zero.
- **Toy p-values floor at `1/(ntoys+1)`.** A p-value equal to the floor is a
  bound, not a measurement, and a significance quoted from it is wrong. Size
  `ntoys` from the p-value you need to resolve.
- **`variance="nominal"` (default) versus `"iterated"`** shifts the best fit by
  several percent. Comparing them is the cheapest approximation diagnostic.

## Before you report a result

Do not present an SGHF number as final without a pyhf cross-check of **that
number** — agreement of the best fit does not imply agreement of a tail
probability. Say which you are giving: an SGHF estimate, or a validated result.

Cheap checks first: `model.approximation.warnings`, `fit.at_bound`,
`fit.converged`, large `fit.pulls`, and the nominal-versus-iterated shift.

Budget the cross-check. On a 232-parameter workspace a single
`pyhf.infer.hypotest` took 274 seconds and a full limit scan never finished, so
check the two or three points you actually report — the limit, the median
expected — not a scan. Full procedure and escalation rules:
[references/validation.md](references/validation.md).

## References

- [references/api.md](references/api.md) — signatures, result fields, exceptions
- [references/workflows.md](references/workflows.md) — runnable recipes
- [references/pitfalls.md](references/pitfalls.md) — counterintuitive behaviour
- [references/validation.md](references/validation.md) — the pyhf cross-check
