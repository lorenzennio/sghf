[Documentation index](index.md) | Previous: [Workspace compilation](workspace-compilation.md) | Next: [Hypothesis testing](hypothesis-testing.md)

# Fitting

`Model.fit` minimizes the quadratic objective of the
[simplified likelihood](simplified-likelihood.md), subject to fixed parameters
and box bounds. With nominal variances this is one constrained weighted
least-squares problem. With iterated variances SGHF repeatedly solves such a
problem while updating the diagonal main-measurement variance.

## Operational summary

```python
fit = model.fit()

print(fit[model.poi_name])
print(fit.error(model.poi_name))
print(fit.chi2)
print(fit.at_bound)
```

The public signature is:

```python
model.fit(
    data=None,
    *,
    auxdata=None,
    fixed=None,
    bounds=None,
    variance="nominal",
    rtol=1.0e-6,
    max_iter=25,
)
```

The default `variance="nominal"` uses one fixed quadratic approximation and
is the fastest option; it is also the default everywhere else in the library,
so fits, asymptotic tests, and toy tests stay mutually consistent. Pass
`variance="iterated"` when the fitted expectations can move enough that
updating the expected-count weights is important. Comparing the two is also a
useful approximation diagnostic.

## Inputs

### Main data

`data=None` uses `model.observations`. A flat sequence must have one value per
flattened bin. A mapping must contain exactly the names in
`model.channel_names`; SGHF concatenates its arrays in that deterministic
order.

Main data must be finite. The quadratic fit accepts non-integer values, which
is required for Asimov data and can also represent preprocessed measurements.

### Auxiliary data

`auxdata=None` uses `model.constraint_mean`. A full sequence must have one
value per scalar parameter, including entries whose precision is zero. A
mapping updates selected constrained parameters by name. Supplying a mapped
auxiliary value for a parameter with zero precision raises `ValueError`.

The auxiliary values enter only through

$$
(\boldsymbol{\theta}-\mathbf a)^\mathsf T
P(\boldsymbol{\theta}-\mathbf a).
$$

### Fixed values and bounds

`fixed={"name": value}` fixes named parameters for this fit. Parameters marked
fixed by the workspace remain fixed at their configured initial values unless
the `fixed` mapping supplies another explicit value. Hypothesis testing uses
this same mechanism for conditional fits.

`bounds={"name": (low, high)}` replaces the model bounds for named parameters
in this fit. Bounds must be increasing; infinite endpoints are accepted, but
`NaN` is not. Every fixed value must lie inside the effective bounds.

## The augmented weighted-least-squares system

Partition scalar parameters into free indices $F$ and fixed indices $G$.
Write the parameter displacement as

$$
\Delta\boldsymbol{\theta}
=
\boldsymbol{\theta}-\boldsymbol{\theta}_0.
$$

For fixed values, define the main-measurement residual target

$$
\mathbf r
=
\mathbf n-\boldsymbol{\nu}_0
-A_G\Delta\boldsymbol{\theta}_G.
$$

The free response is $A_F$. Given a positive diagonal variance $V$, define

$$
B
=
\begin{bmatrix}
V^{-1/2}A_F\\
P_F^{1/2}
\end{bmatrix},
\qquad
\mathbf y
=
\begin{bmatrix}
V^{-1/2}\mathbf r\\
P_F^{1/2}
(\mathbf a_F-\boldsymbol{\theta}_{0,F})
\end{bmatrix}.
$$

Here $P_F$ is the diagonal precision restricted to free parameters. The
quadratic terms that depend on the free displacement are

$$
\|B\Delta\boldsymbol{\theta}_F-\mathbf y\|_2^2.
$$

The information matrix and right-hand side are

$$
I=B^\mathsf TB
=
A_F^\mathsf TV^{-1}A_F+P_F,
$$

$$
\mathbf b=B^\mathsf T\mathbf y.
$$

If the unconstrained solution lies inside every bound, SGHF solves

$$
\boxed{
\Delta\hat{\boldsymbol{\theta}}_F=I^{-1}\mathbf b
}.
$$

The implementation uses a linear solve; it does not form $I^{-1}$ to obtain
the parameter values.

### Identifiability

Before solving, SGHF checks

$$
\operatorname{rank}(I)=|F|.
$$

If the information matrix is rank deficient, it raises
`NonIdentifiableModel`. A parameter can be identifiable through its main
response, a positive auxiliary precision, or a combination of both. Fixing a
degenerate parameter can restore identifiability of the remaining free
subspace.

This is a numerical matrix-rank check using the current weighted design. SGHF
does not report a separate condition number, so a full-rank but poorly
conditioned covariance still requires user inspection.

## Bounds

SGHF first computes the unconstrained normal-equation solution. If any free
parameter violates a bound, it minimizes the same augmented residual with
SciPy's bounded linear least-squares solver:

$$
\min_{\Delta\boldsymbol{\theta}_F}
\|B\Delta\boldsymbol{\theta}_F-\mathbf y\|_2^2
$$

subject to

$$
\boldsymbol{\ell}_F-\boldsymbol{\theta}_{0,F}
\le
\Delta\boldsymbol{\theta}_F
\le
\mathbf u_F-\boldsymbol{\theta}_{0,F}.
$$

An active bound is recognized when the fitted value is within an absolute
tolerance of $10^{-8}$ of either endpoint. `FitResult.at_bound` contains the
names of active free parameters. Fixed parameters are not included in this
tuple.

If the bounded solver fails, SGHF raises `FitConvergenceError`.

## Nominal variance fit

For `variance="nominal"`,

$$
V=\operatorname{diag}(\boldsymbol{\nu}_0).
$$

SGHF performs exactly one solve, so `FitResult.iterations == 1`. The nominal
expectations must all be positive. Since $A$, $V$, and $P$ are fixed, the
objective is globally quadratic over the box-bounded compiled parameter
domain.

This exactness applies to the compiled objective, not to the original
nonlinear HistFactory likelihood.

## Iterated variance fit

For `variance="iterated"`, SGHF starts from

$$
\mathbf v^{(1)}=\boldsymbol{\nu}_0,
\qquad
V^{(1)}=\operatorname{diag}(\mathbf v^{(1)}).
$$

At iteration $k$, it solves the weighted problem and computes

$$
\boldsymbol{\nu}^{(k)}
=
\boldsymbol{\nu}_0
+
A(\hat{\boldsymbol{\theta}}^{(k)}
-\boldsymbol{\theta}_0).
$$

Every component must be finite and strictly positive. Otherwise SGHF raises
`ModelDomainError`.

The convergence measure is

$$
\rho^{(k)}
=
\max_i
\frac{
\left|\nu_i^{(k)}-v_i^{(k)}\right|
}{
\max\!\left(\left|v_i^{(k)}\right|,1\right)
}.
$$

The fit converges when

$$
\rho^{(k)}\le\texttt{rtol}.
$$

Defaults are `rtol=1e-6` and `max_iter=25`.

### Two-cycle damping

Without damping, expected-variance updates can alternate between two vectors.
From the second update onward, SGHF compares the new candidate expectation
with the variance vectors from the current and previous solves. If the
candidate is closer to the previous vector than to the current vector in the
same normalized maximum metric, damping is enabled.

Once enabled, subsequent updates are

$$
\mathbf v^{(k+1)}
=
\frac12
\left(
\mathbf v^{(k)}+\boldsymbol{\nu}^{(k)}
\right).
$$

Damping remains enabled for the rest of that fit. If the convergence
criterion is not met within `max_iter`, SGHF raises `FitConvergenceError`.

### What iteration does and does not mean

The iteration changes only the diagonal variance. It does not:

- recompute the response matrix;
- restore modifier curvature;
- minimize the exact Poisson deviance;
- include derivatives of the variance in the normal equations.

On the convergence iteration, SGHF records the fitted expectation as the final
variance used in the reported `chi2`. The returned covariance is the one
computed in that iteration's solve, using the variance vector at the start of
the iteration. Those vectors agree only to the requested tolerance, so the
covariance is not a separately recomputed Hessian at the exact fixed point.

## Covariance, errors, and correlations

Without active bounds, the nominal quadratic covariance of the free
parameters is

$$
C_F=I^{-1}.
$$

Rows and columns for fixed parameters are zero in the full $K\times K$
reported covariance.

When some free parameters are active at bounds, SGHF removes their rows and
columns from $I$, inverts the remaining inactive block, and embeds that
inverse into the full covariance. Active-bound rows and columns are zero:

$$
C_{\mathrm{active},j}=C_{j,\mathrm{active}}=0.
$$

This is the conditional local covariance obtained by treating active
parameters as fixed. A zero reported error at a bound is not a statement of
zero physical uncertainty and is not a one-sided interval.

Errors are

$$
\sigma_j=\sqrt{\max(C_{jj},0)}.
$$

For pairs with nonzero errors, correlations are

$$
\rho_{jk}=\frac{C_{jk}}{\sigma_j\sigma_k}.
$$

If either error is zero, SGHF reports correlation zero for that entry.

For iterated fits, this covariance also treats the last solve's variance as
fixed. It does not propagate uncertainty from the reweighting rule.

## Pulls and reported chi-square

For a constrained parameter,

$$
\operatorname{pull}_j
=
(\hat\theta_j-a_j)\sqrt{P_{jj}}.
$$

For $P_{jj}=0$, the pull is `NaN` because there is no auxiliary standard
deviation against which to normalize the displacement.

The reported objective value is

$$
\chi^2
=
\sum_i
\frac{(n_i-\hat\nu_i)^2}{v_i}
+
\sum_j
P_{jj}(\hat\theta_j-a_j)^2.
$$

In nominal mode, $\mathbf v=\boldsymbol{\nu}_0$. In a converged iterated fit,
$\mathbf v=\hat{\boldsymbol{\nu}}$ as recorded by the final update.

No nominal goodness-of-fit distribution or number of degrees of freedom is
attached to this value. It is used directly to form profiled objective
differences in [hypothesis testing](hypothesis-testing.md).

## `FitResult`

`Model.fit` returns an immutable `FitResult`.

| Field | Meaning |
|---|---|
| `parameter_names` | scalar parameter order |
| `values` | fitted $\hat{\boldsymbol{\theta}}$ |
| `errors` | square roots of covariance diagonal entries |
| `covariance` | full local covariance with zero fixed/active rows and columns |
| `correlation` | covariance normalized by nonzero errors |
| `pulls` | Gaussian auxiliary pulls; `NaN` when precision is zero |
| `expected` | fitted linearized expectation $\hat{\boldsymbol{\nu}}$ |
| `chi2` | final quadratic objective value |
| `converged` | always `True` for a returned result; failures raise |
| `iterations` | number of weighted solves performed |
| `at_bound` | names of free parameters active at a bound |

Lookup helpers are:

```python
fit.parameter_index("mu")
fit.value("mu")
fit.error("mu")
fit["mu"]  # same value as fit.value("mu")
```

Unknown names raise `KeyError`.

## Failures and diagnostics

| Condition | Result |
|---|---|
| invalid data, auxiliary shape, variance name, tolerance, iteration limit, fixed value, or bounds | `ValueError` |
| nominal or fitted expected yield is non-positive or non-finite | `ModelDomainError` |
| free information matrix is rank deficient | `NonIdentifiableModel` |
| bounded solver failure or exhausted variance iterations | `FitConvergenceError` |

For scientific use, inspect at least:

- `fit.at_bound`;
- nuisance pulls;
- the smallest fitted expected yields;
- covariance scale and correlations;
- stability between nominal and iterated variance;
- stability against comparison with the original pyhf model when shifts are
  not demonstrably local.

The broader decision procedure is in
[validation and applicability](validation-and-applicability.md).

---

[Documentation index](index.md) | Previous: [Workspace compilation](workspace-compilation.md) | Next: [Hypothesis testing](hypothesis-testing.md)
