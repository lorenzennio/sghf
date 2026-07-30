[Documentation index](index.md) | Previous: [Fitting](fitting.md) | Next: [Toy inference](toy-inference.md)

# Hypothesis testing

`Model.hypotest` tests a specified value $\mu$ of the parameter of interest
using one-sided profiled objective differences. SGHF can map those statistics
to asymptotic probabilities or calibrate them with pseudo-experiments. This
page describes the asymptotic path; the ensemble path is in
[toy inference](toy-inference.md).

The conventions follow the
[profile-likelihood and Asimov framework](references.md#profile-likelihood-and-asymptotic-tests)
and the
[modified-frequentist construction](references.md#modified-frequentist-cls),
but every equation below is written for the specific quantities returned by
SGHF.

## Operational summary

```python
result = model.hypotest(
    1.0,
    test_stat="qtilde",
    calculator="asymptotic",
)

print(result.q)
print(result.cls)
print(result.expected_cls)
```

The relevant public arguments are:

```python
model.hypotest(
    poi_test,
    data=None,
    *,
    test_stat="qtilde",
    calculator="asymptotic",
    variance="nominal",
    ...
)
```

Both calculators default to `variance="nominal"`, so the asymptotic and toy
dispatches evaluate the same test statistic and their results can be compared
directly. Pass `variance="iterated"` explicitly to opt into the reweighted
statistic. Toy-only arguments are ignored by the asymptotic dispatch.

Use:

- `q0` with `poi_test=0` for discovery against the background-only value;
- `q` for a one-sided upper test with an unconstrained-below free POI fit;
- `qtilde` for a one-sided upper test with a physical POI lower bound exactly
  equal to zero.

## Free and conditional fits

Let

$$
\hat{\boldsymbol{\theta}}
=
\underset{\boldsymbol{\theta}}{\arg\min}\,
\chi^2(\boldsymbol{\theta})
$$

be the free fit under the bounds used by the selected statistic, and let

$$
\hat{\hat{\boldsymbol{\theta}}}(\mu)
=
\underset{\boldsymbol{\theta}:\,\theta_\mathrm{POI}=\mu}{\arg\min}\,
\chi^2(\boldsymbol{\theta})
$$

be the conditional fit. SGHF extends the conditional fit's POI bounds when
necessary so that the requested finite test value can be fixed.

The free-fit bounds depend on the statistic:

| Statistic | Free POI bounds |
|---|---|
| `q0` | model bounds |
| `q` | $(-\infty, u_\mu]$, where $u_\mu$ is the model upper bound |
| `qtilde` | model bounds, with the additional requirement that the model lower bound is exactly zero |

Thus `q` can represent a negative unconstrained estimator even when the loaded
model normally has a non-negative POI. `qtilde` uses the bounded denominator
fit.

Define the non-negative profiled objective difference

$$
\Delta\chi^2_\mu
=
\max\!\left[
0,\,
\chi^2\!\left(\hat{\hat{\boldsymbol{\theta}}}(\mu)\right)
-
\chi^2(\hat{\boldsymbol{\theta}})
\right].
$$

The explicit maximum suppresses small negative differences from numerical
effects.

### Nominal versus iterated interpretation

With `variance="nominal"`, the free and conditional fits minimize the same
fixed quadratic objective, so $\Delta\chi^2_\mu$ is a genuine profiled
objective difference within the compiled model.

With `variance="iterated"`, each fit updates its variance from its own fitted
expectation. The two reported `chi2` values can therefore use different
diagonal weights. Their difference is the implemented reweighted statistic;
it is not the difference of one parameter-independent likelihood or one fixed
quadratic objective. This distinction matters when fitted expectations move
substantially.

## One-sided statistics

Let $\hat\mu$ be the POI value in the relevant free fit.

### Discovery statistic $q_0$

`q0` requires `poi_test=0` and uses

$$
q_0
=
\begin{cases}
\Delta\chi^2_0, & \hat\mu\ge0,\\
0, & \hat\mu<0.
\end{cases}
$$

If the model lower bound is already zero, the negative branch cannot normally
be reached. SGHF does not require that bound for `q0`, so the clipping rule is
still explicit.

### Upper-test statistic $q_\mu$

`q` uses the free fit with no finite lower POI bound:

$$
q_\mu
=
\begin{cases}
\Delta\chi^2_\mu, & \hat\mu\le\mu,\\
0, & \hat\mu>\mu.
\end{cases}
$$

An upward fluctuation beyond the tested value is not counted as evidence
against that value.

### Bounded upper-test statistic $\tilde q_\mu$

`qtilde` requires the model's POI lower bound to be exactly zero. Its free fit
therefore has $\hat\mu\ge0$, and SGHF applies

$$
\tilde q_\mu
=
\begin{cases}
\Delta\chi^2_\mu, & \hat\mu\le\mu,\\
0, & \hat\mu>\mu.
\end{cases}
$$

The distinction from $q_\mu$ is the bounded denominator fit. In the nominal
quadratic toy shortcut, the same bounded statistic is written as an explicit
piecewise function of an *unbounded* affine estimate; see
[toy inference](toy-inference.md).

## Conditional Asimov construction

SGHF constructs an Asimov data set conditionally from the supplied
observations rather than always using the workspace initial point.

First it fits the observed main data with the POI fixed to

$$
\mu_A
=
\begin{cases}
1, & q_0,\\
0, & q_\mu\ \text{or}\ \tilde q_\mu.
\end{cases}
$$

Call the result $\hat{\hat{\boldsymbol{\theta}}}_A(\mu_A)$. The deterministic
Asimov main and auxiliary data are

$$
\mathbf n_A
=
\boldsymbol{\nu}\!\left(
\hat{\hat{\boldsymbol{\theta}}}_A(\mu_A)
\right),
\qquad
\mathbf a_A
=
\hat{\hat{\boldsymbol{\theta}}}_A(\mu_A).
$$

SGHF then evaluates the same selected statistic at $\mathbf n_A,\mathbf a_A$
to obtain $q_A$ and returns the Asimov free and conditional fits for
inspection.

Consequences of this construction are:

- exclusion uses nuisance values conditionally fitted at $\mu=0$;
- discovery expectation uses nuisance values conditionally fitted at
  $\mu=1$;
- $q_A$ depends on the supplied observed data through those conditional
  nuisance estimates;
- the configured POI bounds must admit the Asimov truth value used by the
  initial conditional fit.

## Asymptotic transformation

Define

$$
\sqrt q=\sqrt{\max(q,0)},
\qquad
\sqrt{q_A}=\sqrt{\max(q_A,0)}.
$$

For `q0` and `q`, SGHF uses

$$
t=\sqrt q-\sqrt{q_A}.
$$

For `qtilde`, it uses

$$
t
=
\begin{cases}
\sqrt q-\sqrt{q_A},
& \sqrt q\le\sqrt{q_A},\\[4pt]
\dfrac{q-q_A}{2\sqrt{q_A}},
& \sqrt q>\sqrt{q_A}\ \text{and}\ \sqrt{q_A}>0.
\end{cases}
$$

If $\sqrt{q_A}=0$, the first expression is used. The transformed statistic is
bounded below at $-\sqrt{q_A}$ for the clipped `qtilde` construction.

Let $\Phi$ be the standard-normal cumulative distribution function. SGHF
defines

$$
CL_{s+b}
=
\Phi\!\left[-\left(t+\sqrt{q_A}\right)\right],
$$

$$
CL_b=\Phi(-t),
$$

and, when $CL_b>0$,

$$
\boxed{
CL_s=\frac{CL_{s+b}}{CL_b}
}.
$$

If the transformed value is outside the implemented clipped domain, or if
$CL_b=0$, the affected probabilities are `NaN`.

For `q0`, the first expression simplifies to

$$
CL_{s+b}=\Phi(-\sqrt{q_0}),
$$

which is the implemented asymptotic background-only tail probability. The
reported discovery significance is

$$
Z_0=\sqrt{q_0}.
$$

For `q` and `qtilde`, `significance` is `NaN`; the exclusion outputs are
`clsb`, `clb`, and `cls`.

## Expected asymptotic bands

SGHF evaluates the same $CL_s$ mapping at transformed values

$$
t\in(2,1,0,-1,-2).
$$

For `qtilde`, each value is clipped below at $-\sqrt{q_A}$. Because larger
$t$ gives a stronger exclusion at fixed $\mu$, the returned tuple is labeled
in expected-limit order:

```text
(-2 sigma, -1 sigma, median, +1 sigma, +2 sigma)
```

This ordering describes the expected upper-limit bands obtained when the
curves are inverted. It should not be inferred solely from the numerical
ordering of five $CL_s$ values at an arbitrary test point.

## `HypothesisTestResult`

The asymptotic calculator returns an immutable `HypothesisTestResult`.

| Field | Meaning |
|---|---|
| `poi_test` | tested POI value |
| `test_stat` | `"q0"`, `"q"`, or `"qtilde"` |
| `q` | observed one-sided statistic |
| `q_asimov` | statistic evaluated on the conditional Asimov data |
| `transformed` | $t$ used by the normal-CDF mapping |
| `clsb` | implemented $CL_{s+b}$ |
| `clb` | implemented $CL_b$ |
| `cls` | ratio $CL_{s+b}/CL_b$ |
| `expected_cls` | five asymptotic $CL_s$ values in expected-limit band order |
| `significance` | $\sqrt{q_0}$ for `q0`, otherwise `NaN` |
| `free_fit` | fit to observed data under the statistic's denominator bounds |
| `fixed_fit` | observed conditional fit at the tested POI |
| `asimov_free_fit` | free fit to conditional Asimov data |
| `asimov_fixed_fit` | tested-POI fit to conditional Asimov data |

The fit objects are important diagnostics. In particular, compare their
expected yields, active bounds, and variance-mode stability before treating
the asymptotic result as reliable.

## Assumptions specific to the asymptotic mapping

The asymptotic result inherits all assumptions of the
[simplified likelihood](simplified-likelihood.md) and adds:

1. The local profiled statistic is adequately described by a Wald
   approximation.
2. The conditional Asimov statistic represents the scale of fluctuations
   relevant to the tested hypothesis.
3. The normal-CDF formula and the implemented boundary clipping adequately
   describe the statistic's sampling distribution.
4. Nuisance estimators are sufficiently regular and identifiable near the
   free and conditional fits.
5. For iterated variance, using differences of separately reweighted
   objectives does not materially alter the intended calibration.

These conditions can fail in sparse data, far tails, weakly identified
models, models with active bounds, or tests that force large nonlinear
modifier shifts. Toy calibration evaluates the SGHF statistic under generated
pseudo-data and is often the direct diagnostic for asymptotic behavior, but it
does not repair the compiled-model approximations.

---

[Documentation index](index.md) | Previous: [Fitting](fitting.md) | Next: [Toy inference](toy-inference.md)
