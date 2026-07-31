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
    calculator="asymptotic",
    variance="nominal",
    bracket=None,
    scan=None,
    points=21,
    ntoys=10_000,
    batch_size=None,
    sampling="poisson",
    rng=None,
    keep_results=False,
)
```

`test_stat` is `q` or `qtilde`. The statistic, variance behavior, and
asymptotic probabilities are defined in
[hypothesis testing](hypothesis-testing.md).

`calculator` selects `"asymptotic"` or `"toys"`, exactly as in
`Model.hypotest`. The asymptotic calculator ignores `scan`, `points`,
`ntoys`, `batch_size`, `sampling`, and `rng`; the toy calculator ignores
nothing but reinterprets `bracket` as the range of its scan. The two use
different crossing algorithms, described below.

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
| `calculator` | `"asymptotic"` or `"toys"` |
| `scan_values` | POI grid the toy calculator scanned; empty for the asymptotic one |
| `results` | every test evaluated, sorted by `poi_test`; empty unless `keep_results=True` |

### Plotting the evaluated curve

`results` costs nothing extra: it is the set of tests the solve already
required. Each entry is a full `HypothesisTestResult` (or
`ToyHypothesisTestResult`), so the observed curve, the expected band, and
the underlying statistics are all available. It is returned only when
asked, because retaining it also retains its fits.

```python
import numpy as np

limit = model.upper_limit(keep_results=True)

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

`keep_results` defaults to `False` because retaining a `LimitResult` would
otherwise retain every fit it evaluated. Each `HypothesisTestResult`
carries four `FitResult` objects, so on a 232-parameter workspace the 60
retained results held roughly 200 MiB. The flag costs nothing either way:
the tests are computed regardless, and it only controls whether they are
handed back.

## The toy calculator

`calculator="toys"` scans a grid of POI values, runs a toy hypothesis test
at each, and interpolates the crossings. It does not use a root finder,
and this is not an efficiency choice.

The toy $CL_s$ curve is stochastic. On a 232-parameter workspace at
`ntoys=10_000` it changes sign ten times across $\mu\in[5.90,6.10]$ and
carries 2 to 3 percent relative noise, and it does so even at a fixed
seed, because each tested $\mu$ conditions its ensembles on a different
fit. Brent's method requires a continuous function with one sign change in
the bracket. Applied here it would converge to an arbitrary member of a
cluster of crossings and report it to `xtol=1e-10`, a precision the
underlying curve does not possess.

So each of the six curves takes its **last** downward crossing of
$\alpha$, which keeps the reported limit conservative when noise straddles
the threshold, interpolated linearly in $\log CL_s$. The corrected tail
fractions of [toy inference](toy-inference.md) are strictly positive, so
those logarithms are always finite. A curve with no crossing anywhere in
the scanned range raises `LimitNotFound` naming the curve and reporting the
$CL_s$ range it actually spanned.

Without an explicit `scan`, the grid is seeded from the asymptotic
crossings: an asymptotic limit is solved first, and `points` values are
spaced evenly from $0.7\times$ the smallest crossing to $1.3\times$ the
largest, clipped below at the POI lower bound. This spends toys where the
limits are instead of across a mostly empty bracket. Pass `scan` to
override the grid entirely, or `bracket` to keep the automatic point count
over a range you choose.

```python
limit = model.upper_limit(
    calculator="toys",
    ntoys=20_000,
    rng=1234,
)

limit.observed      # interpolated crossing of the observed curve
limit.scan_values   # the grid actually evaluated
```

Cost is one toy hypothesis test per grid point, plus one asymptotic solve
when the grid is seeded automatically. On the 232-parameter workspace a
default 21-point scan at `ntoys=10_000` took 35 seconds and returned an
observed limit of 6.13 against the asymptotic 5.98.

Two resolutions bound the result and neither is reported as an
uncertainty. Monte Carlo noise on $CL_s$ propagates to the crossing:
repeating a one-bin limit across six seeds at `ntoys=20_000` moved the
observed value by a standard deviation of about 1 percent. Grid spacing
bounds the interpolation: the default 21 points over a seeded range gave
a spacing of 0.35 on the reference workspace, which the $-2\sigma$
crossing near 1.3 resolves only coarsely. Raise `points` or pass a
targeted `scan` when either matters.

## Failures

`ValueError` is raised when:

- `level` is not strictly between zero and one;
- the bracket is non-finite, non-increasing, or otherwise invalid;
- `calculator` is neither `"asymptotic"` nor `"toys"`;
- a toy `scan` holds fewer than two distinct values, or any non-finite one;
- `points` is not an integer of at least two;
- an invalid test statistic or variance mode reaches the hypothesis-test
  implementation.

`LimitNotFound` is raised when:

- an endpoint $CL_s$ is non-finite;
- a curve has no sign change inside the requested bracket;
- automatic expansion does not bracket all six curves within its loop;
- a toy curve never crosses $\alpha$ within the scanned grid.

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
