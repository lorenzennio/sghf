[Documentation index](index.md) | Previous: [Toy inference](toy-inference.md) | Next: [API reference](api-reference.md)

# Upper limits

`Model.upper_limit` inverts the asymptotic $CL_s$ calculation to find an
observed upper limit and five expected limits. The current method does not
invert toy-calibrated tests.

The statistical origin of the ratio is given in the
[modified-frequentist reference](references.md#modified-frequentist-cls);
this page documents SGHF's implemented inversion.

## Operational summary

```python
limit = model.upper_limit()

print(limit.observed)
print(limit.expected)
print(limit.bracket)
```

The public signature is:

```python
model.upper_limit(
    data=None,
    *,
    level=0.05,
    test_stat="qtilde",
    variance="nominal",
    bracket=None,
)
```

`test_stat` is `q` or `qtilde`. The statistic, variance behavior, and
asymptotic probabilities are defined in
[hypothesis testing](hypothesis-testing.md).

## Limit definition

For a confidence-level threshold $\alpha$ with

$$
0<\alpha<1,
$$

the observed upper limit is a root

$$
\boxed{
CL_s^\mathrm{obs}(\mu_\mathrm{up})=\alpha
}.
$$

The default is $\alpha=0.05$. For each expected-band index $r$, SGHF similarly
solves

$$
CL_{s,r}^\mathrm{exp}(\mu_{\mathrm{up},r})=\alpha.
$$

The expected tuple is ordered as

```text
(-2 sigma, -1 sigma, median, +1 sigma, +2 sigma).
```

Each function evaluation calls the asymptotic calculator with the selected
test statistic and variance mode. Results at an exactly repeated floating
point $\mu$ are cached during one `upper_limit` call, so the observed and
expected root searches reuse a computed `HypothesisTestResult`.

## Automatic bracket

With `bracket=None`, SGHF starts from

$$
\mu_\mathrm{low}
=
\max(0,\ell_\mu),
\qquad
\mu_\mathrm{high}=u_\mu,
$$

where $\ell_\mu$ and $u_\mu$ are the model POI bounds. Both endpoints must be
finite and increasing; in particular, automatic bracketing requires a finite
model upper bound.

At the high endpoint, SGHF evaluates the observed and all five expected
$CL_s$ values. If every value is finite and at or below $\alpha$, the endpoint
is accepted. Otherwise it doubles the bracket width:

$$
\mu_\mathrm{high}
\leftarrow
\mu_\mathrm{low}
+
2(\mu_\mathrm{high}-\mu_\mathrm{low}).
$$

The loop permits 20 high-end checks. If no endpoint brackets all six curves,
SGHF raises `LimitNotFound`.

After the high endpoint is selected, each of the six curves is checked
separately at both endpoints. A value exactly equal to $\alpha$ returns that
endpoint directly.

## Explicit bracket

With `bracket=(low, high)`, SGHF uses exactly those finite increasing
endpoints. It does not expand an explicit bracket.

For each curve, the endpoint values of

$$
f(\mu)=CL_s(\mu)-\alpha
$$

must be finite and have opposite signs, unless one endpoint is already a
root. Otherwise SGHF raises `LimitNotFound`.

Supplying an explicit bracket is useful when the model's configured POI upper
bound is not an appropriate search scale. It does not change the POI bounds
used in the free fit; the conditional tested value is made admissible by the
hypothesis-test implementation.

## Root solve

Once a sign change is present, SGHF uses Brent's scalar root method with
absolute tolerance

$$
\texttt{xtol}=10^{-10}.
$$

The method assumes that the selected $CL_s$ curve is sufficiently continuous
over the bracket. SGHF checks endpoint signs but does not prove monotonicity
or search for multiple crossings. Strong non-monotonic behavior can therefore
make the selected crossing bracket dependent.

## `LimitResult`

The immutable result contains:

| Field | Meaning |
|---|---|
| `observed` | root of the observed $CL_s=\alpha$ curve |
| `expected` | five expected roots in `(-2 sigma, -1 sigma, median, +1 sigma, +2 sigma)` order |
| `level` | requested $\alpha$ |
| `test_stat` | `"q"` or `"qtilde"` |
| `bracket` | final bracket used for all six solves, including automatic expansion |
| `results` | every `HypothesisTestResult` evaluated, sorted by `poi_test` |

### Plotting the evaluated curve

`results` costs nothing extra: it is the set of tests the bracket expansion
and the six root finds already required. Each entry is a full
`HypothesisTestResult`, so the observed curve, the expected band, and the
underlying statistics are all available.

```python
import numpy as np

limit = model.upper_limit()

poi = np.array([test.poi_test for test in limit.results])
observed = np.array([test.cls for test in limit.results])
band = np.array([test.expected_cls for test in limit.results])

plt.plot(poi, observed, label="observed")
plt.fill_between(poi, band[:, 0], band[:, 4], alpha=0.3)
plt.plot(poi, band[:, 2], linestyle="--", label="expected")
plt.axhline(limit.level, color="red")
```

These points are where the solver chose to look, not an even scan. Brent's
method clusters them tightly around each of the six crossings and skips the
intervals between, so the sampling density is highly non-uniform and the
gaps carry no meaning. On a 232-parameter workspace a default solve visited
60 values of $\mu$ whose spacing ranged from $5\times10^{-15}$ to $2$.

Do not read structure into that spacing, and do not use these points to
integrate or to locate a crossing other than the six already reported. When
an evenly sampled curve is required, call `Model.hypotest` over a grid of
your own choosing and accept the extra cost.

Retaining a `LimitResult` retains every fit it evaluated. Each
`HypothesisTestResult` carries four `FitResult` objects, so on a
232-parameter workspace the 60 retained results held roughly 200 MiB. Peak
memory during the solve is unchanged, since these objects already existed;
what changes is that they stay alive as long as the result does.

## Failures

`ValueError` is raised when:

- `level` is not strictly between zero and one;
- the bracket is non-finite, non-increasing, or otherwise invalid;
- an invalid test statistic or variance mode reaches the hypothesis-test
  implementation.

`LimitNotFound` is raised when:

- an endpoint $CL_s$ is non-finite;
- a curve has no sign change inside the requested bracket;
- automatic expansion does not bracket all six curves within its loop.

## Interpretation and applicability

The root solver can be numerically accurate while the statistical
approximation is inaccurate. Limit interpretation depends on:

- validity of the compiled linear response over the range up to the crossing;
- stability of nominal versus iterated weighting;
- regularity of the free and conditional fits;
- accuracy of the Wald/Asimov mapping;
- absence of problematic active bounds or sparse-tail effects.

`Model.upper_limit` is asymptotic only. To investigate sampling behavior at
selected $\mu$ values, use `Model.hypotest(..., calculator="toys")`; SGHF does
not currently provide automatic root inversion of those empirical results.

---

[Documentation index](index.md) | Previous: [Toy inference](toy-inference.md) | Next: [API reference](api-reference.md)
