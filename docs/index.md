[Documentation index](index.md) | Previous: [References](references.md) | Next: [Simplified likelihood](simplified-likelihood.md)

# SGHF documentation

SGHF compiles a pyhf workspace into a local linear response with a quadratic
fitting objective. It is designed for fast fitting, asymptotic inference, and
large toy ensembles. It does not evaluate the exact nonlinear HistFactory
likelihood.

Start with the theory pages to understand what is being approximated. Before a
final scientific result, apply the
[validation and applicability checklist](validation-and-applicability.md).

## Five-minute workflow

```python
import sghf

# 1. Compile a pyhf workspace.
model = sghf.load("workspace.json")

# 2. Inspect what was compiled.
print(model.approximation)
print(model.parameter_names)

# 3. Fit the observed data.
fit = model.fit()
print(fit.value(model.poi_name), fit.error(model.poi_name))

# 4a. Use the asymptotic calculator.
asymptotic = model.hypotest(1.0, test_stat="qtilde")
print(asymptotic.cls, asymptotic.expected_cls)

# 4b. Or calibrate the SGHF statistic with pseudo-experiments.
toy_discovery = model.hypotest(
    0.0,
    test_stat="q0",
    calculator="toys",
    ntoys=100_000,
    rng=12345,
)
print(toy_discovery.p0, toy_discovery.significance)

# 5. Invert the asymptotic CLs curve.
limit = model.upper_limit()
print(limit.observed, limit.expected)
```

The [API reference](api-reference.md) contains a complete runnable workspace
and documents all arguments and return fields.

## Documentation map

1. [The simplified likelihood](simplified-likelihood.md) derives the compiled
   expectation, quadratic objective, approximations, and validity conditions.
2. [Workspace compilation](workspace-compilation.md) gives the exact
   workspace-to-array mapping for every supported modifier.
3. [Fitting](fitting.md) derives the weighted solve, iterated variances,
   covariance, pulls, bounds, and failures.
4. [Hypothesis testing](hypothesis-testing.md) defines $q_0$, $q_\mu$,
   $\tilde q_\mu$, conditional Asimov data, and asymptotic probabilities.
5. [Toy inference](toy-inference.md) derives the vectorized nominal estimator,
   empirical exclusion, and toy-based $q_0$ discovery.
6. [Upper limits](upper-limits.md) documents $CL_s$ inversion: asymptotic
   root finding with bracketing, and the toy grid scan.
7. [API reference](api-reference.md) lists every public method, result field,
   default, and exception.
8. [Validation and applicability](validation-and-applicability.md) gives the
   decision checklist for scientific use.
9. [References](references.md) collects the primary statistical and model
   sources and points to the SGHF implementation files.

## Fixed notation

| Symbol | Meaning |
|---|---|
| $\mathbf n$ | flattened main observations |
| $\boldsymbol{\nu}_0$ | nominal expected yields at the expansion point |
| $A$ | response matrix, bins by scalar parameters |
| $\boldsymbol{\theta}$ | model parameters, including the POI |
| $\boldsymbol{\theta}_0$ | configured initial values and expansion point |
| $\mathbf a$ | Gaussian auxiliary observations |
| $P$ | diagonal auxiliary-constraint precision |
| $V$ | diagonal main-measurement variance |
| $\mu$ | parameter of interest |
| $\hat{\boldsymbol{\theta}}$ | free fit |
| $\hat{\hat{\boldsymbol{\theta}}}(\mu)$ | conditional fit at fixed $\mu$ |

## Calculator and variance choices

| Operation | Default | What it computes |
|---|---|---|
| `model.fit()` | nominal variance | constrained quadratic fit in a single solve |
| `model.hypotest(..., calculator="asymptotic")` | nominal variance | profiled statistic plus Wald/Asimov mapping |
| `model.hypotest(..., calculator="toys")` | nominal variance | vectorized empirical exclusion or discovery |
| any of the above with `variance="iterated"` | explicit opt-in | reweighted solves; per-toy fits for the toy calculator |
| `model.upper_limit()` | nominal asymptotic | roots of observed and expected $CL_s$ curves |
| `model.upper_limit(..., calculator="toys")` | nominal variance | grid scan of toy $CL_s$ curves with interpolated crossings |

Every entry point defaults to `variance="nominal"` so that the asymptotic and
toy calculators evaluate the same test statistic and can be compared directly.
`"iterated"` is the more faithful approximation and remains available
everywhere as an explicit opt-in, but it is not affordable as a toy default:
the per-toy fit path costs seconds per toy on a large workspace.

Nominal variance fixes
$V=\operatorname{diag}(\boldsymbol{\nu}_0)$. Iterated variance updates the
diagonal from fitted positive expectations. Both use the same fixed linear
response matrix.

## Model boundary

SGHF supports `normfactor`, `normsys`, `histosys`, `staterror`, `shapesys`,
and `lumi` in workspace schema version `"1.0.0"`. Loading a supported
workspace establishes only that SGHF can compile it. Accuracy must be assessed
over the parameter and tail region used by the analysis.

The central distinction is:

$$
\text{toy calibration of the SGHF statistic}
\ne
\text{exact inference for the nonlinear pyhf likelihood}.
$$

Poisson toys can accurately represent discrete fluctuations around the
compiled mean at finite Monte Carlo resolution. They do not restore discarded
modifier curvature or the original auxiliary likelihood terms.

---

[Documentation index](index.md) | Previous: [References](references.md) | Next: [Simplified likelihood](simplified-likelihood.md)
