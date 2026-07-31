# SGHF pitfalls

Behaviours that look wrong and are not, plus the ones that silently mislead.
Every number below is reproducible on a 232-parameter workspace.

## CLs is exactly 1.0 for every mu below the best fit

Not a bug. For `q` and `qtilde` the observed statistic is defined as zero when
`mu_hat > mu`: the data are more signal-like than the hypothesis, so an upper
test cannot reject. Every toy also satisfies `q >= 0`, so the tail fraction
`P(q >= 0)` is exactly 1, and about half the signal-plus-background ensemble
sits precisely at zero.

```
mu=0.5  cls=1.00000  q=0.0000  fraction of s+b toys with q==0: 0.498
mu=2.0  cls=1.00000  q=0.0000  fraction of s+b toys with q==0: 0.501
mu=4.0  cls=1.00000  q=0.0000  fraction of s+b toys with q==0: 0.489
```

This matches pyhf's toy convention (`sum(samples >= value) / n`). The asymptotic
calculator instead returns `CLs+b = 0.5` there, because the asymptotic formula
splits the atom at zero rather than counting it. Neither is wrong; they are
different treatments of the same degenerate point, and **neither affects the
upper limit**, whose crossing always lies above `mu_hat`.

Two practical consequences:

- **Below `mu_hat`, quote the asymptotic value.** The toy number is
  `1.0` by construction and carries no information. More toys will not change
  it; the degeneracy is in the estimator at the boundary of the statistic's
  support, not in the sample size.
- **Do not compare calculators there.** A disagreement at such a point is
  expected and tells you nothing about whether the approximation is sound.
  Compare them near the crossing instead, where both are informative.

## The asymptotic discovery p-value is `clsb`, not `cls`

`HypothesisTestResult` has no `p0` field. For `test_stat="q0"` the p-value is
`clsb`. The `cls` and `clb` fields are still populated, but they are exclusion
quantities and mean nothing for a discovery test — and `cls` is close enough to
a plausible p-value to be read as one by mistake:

```
q0 = 13.363
clsb        = 0.00012830   <- the p0 you want, = norm.sf(3.656)
cls         = 0.04566      <- NOT a p-value here; 356x too large
significance = 3.656 sigma
```

Reporting `cls` would give 1.7 sigma instead of 3.66 sigma. Use `clsb`, or
`norm.sf(result.significance)`. The toy path does name the field `p0`.

## Toy p-values cannot go below 1/(ntoys+1)

The empirical tail uses add-one smoothing, so the smallest p-value obtainable is
`1/(ntoys+1)`. At the default `ntoys=10_000` that is 1e-4.

```
ntoys=1000   clsb=0.000999   floor=1/(n+1)=0.000999   <- pinned at the floor
ntoys=10000  clsb=0.000200   floor=1/(n+1)=0.000100
```

**Size the ensemble from the p-value you need**, not from a habit. A p-value
pinned exactly at the floor means the toys never reached the observed statistic
and the true tail is smaller — it is a lower bound, not a measurement. A toy
significance quoted from a floored p-value is wrong: 1000 toys can calibrate to
roughly 3 sigma, no further.

Compare a toy p-value to `1/(ntoys+1)` before quoting it.

## variance="nominal" versus "iterated"

`"nominal"` is the default everywhere. It freezes the variance at the workspace
nominal point and solves once. `"iterated"` re-estimates the variance from the
fitted yields and is the more faithful approximation.

```
nominal   mu_hat=4.1241  err=1.1282  iterations=1
iterated  mu_hat=4.3272  err=1.2055  iterations=7
```

That is a 5% shift in the central value, and it propagates to limits. The
default is `"nominal"` so that the asymptotic and toy calculators evaluate the
same statistic and are directly comparable — with `"iterated"` they are not,
because the toy path would need a fit per toy.

Comparing the two is the cheapest approximation diagnostic there is. A large
gap means the linearization is being pushed.

## Toy upper limits scan; they do not root-find

The toy CLs curve is stochastic — it changes sign several times across a narrow
mu window even at a fixed seed, because each tested mu conditions its ensembles
on a different fit. A root finder would converge on an arbitrary crossing and
report it to machine precision, which would be fiction.

`upper_limit(calculator="toys")` therefore evaluates a grid and interpolates the
last crossing in log CLs. Two resolutions bound the answer and **neither appears
in the result**: Monte Carlo noise on the crossing (about 1% seed-to-seed at
20k toys) and the grid spacing (`points=21` over an auto-seeded range). Raise
`points`, raise `ntoys`, or pass an explicit `scan` when either matters.

Without an explicit `scan`, the toy path first solves the asymptotic limit to
seed its grid — so it is not purely toy-based.

## error() returns 0.0 for a parameter at a bound

A parameter resting on a bound has zero covariance entries, so its reported
error is `0.0`. That is a flag, not an uncertainty.

```
deficit fit: mu=0.0000  err=0.0000  at_bound=('mu',)
```

**Check `fit.at_bound` before quoting any uncertainty.** A zero error with the
parameter named in `at_bound` means the fit is on the boundary and the Gaussian
error is undefined there; quote an interval from `upper_limit` instead.

## Band ordering differs between CLs and significance

`expected_cls` and `LimitResult.expected` run `(-2σ, -1σ, median, +1σ, +2σ)`,
so the values increase. But `expected_significance` from a toy discovery runs
the other way — most significant first:

```
expected_cls:          (0.0, 0.0, 0.0, 1e-05, 0.00046)     increasing
expected_significance: (2.7944, 1.9334, 0.8977, 0.0, 0.0)  decreasing
```

Both order "most optimistic first". Label band entries explicitly rather than
assuming a direction.

## keep_results retains every fit

`upper_limit(..., keep_results=True)` hands back every evaluated test, and each
carries four `FitResult` objects with full covariance matrices — of order
200 MiB on a 232-parameter model. It defaults to `False` for that reason; the
tests are computed either way.

For `calculator="asymptotic"` those points are wherever Brent's method looked:
clustered at the six crossings with wide gaps between. Plot them if useful, but
the gaps are an artefact of the solver, not structure in the curve. The toy path
returns its even grid instead.

## Reported uncertainty is the Wald error

`fit.error(name)` is the square root of the covariance diagonal — a Wald
uncertainty from the quadratic objective, symmetric by construction. It is not
a profile-likelihood interval and will not reflect asymmetry or boundary
effects. For an interval that respects the boundary, use `upper_limit`.

## Parameter order is not pyhf's

`model.parameter_names` groups by modifier type and sorts by name within each
group; pyhf does not sort within a group. The two orders coincide only when a
workspace happens to declare its modifiers alphabetically. Everything in SGHF is
addressed by name through `parameter_index`, so this only matters if you index
positionally against pyhf output — do not.
