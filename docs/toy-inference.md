[Documentation index](index.md) | Previous: [Hypothesis testing](hypothesis-testing.md) | Next: [Upper limits](upper-limits.md)

# Toy inference

SGHF can generate pseudo-observations and use empirical ensembles to calibrate
one-sided exclusion and discovery statistics. In the default nominal-variance
mode, a single information-matrix solve produces an affine estimator of the
parameter of interest; all toy statistics are then evaluated with vectorized
array operations. This is the main fast toy-calculation path.

Toy calibration measures the sampling distribution of the statistic defined
by SGHF under pseudo-data generated from the compiled model. It does not
remove the linear response approximation or make the result identical to a
toy study of the original nonlinear pyhf model.

## Operational summary

Generate pseudo-observations directly:

```python
toys = model.toys(
    100_000,
    parameters={"mu": 0.0},
    sampling="poisson",
    rng=12345,
)
```

Run a toy-calibrated upper test:

```python
exclusion = model.hypotest(
    1.0,
    test_stat="qtilde",
    calculator="toys",
    ntoys=100_000,
    rng=12345,
)
```

Run a toy-calibrated discovery test:

```python
discovery = model.hypotest(
    0.0,
    test_stat="q0",
    calculator="toys",
    ntoys=100_000,
    rng=12345,
)

print(discovery.p0, discovery.significance)
```

For the toy calculator, `variance=None` selects `variance="nominal"`.
Supplying `variance="iterated"` enables the slower per-toy fit path.

## Pseudo-observation generation

The public generator signature is:

```python
model.toys(
    size,
    *,
    parameters=None,
    sampling="poisson",
    fluctuate_auxiliary=True,
    rng=None,
)
```

`size` must be a positive integer. The truth vector can be a full sequence, a
partial name-to-value mapping, a `FitResult`, or `None`. A mapping starts from
`model.initial` and replaces the named entries; `None` uses
`model.initial`.

For truth $\boldsymbol{\theta}^\ast$, SGHF first evaluates the positive
compiled expectation

$$
\boldsymbol{\nu}^\ast
=
\boldsymbol{\nu}_0
+
A(\boldsymbol{\theta}^\ast-\boldsymbol{\theta}_0).
$$

### Main measurement

With `sampling="poisson"`, toys are independent by bin and ensemble entry:

$$
n_{ri}^\ast
\sim
\operatorname{Pois}(\nu_i^\ast),
\qquad
r=1,\ldots,N_\mathrm{toys}.
$$

This retains discrete Poisson fluctuations around the *compiled* mean even
though SGHF fits each toy with a quadratic statistic.

With `sampling="gaussian"`,

$$
n_{ri}^\ast
\sim
N(\nu_i^\ast,\nu_i^\ast).
$$

Gaussian draws can be non-integer or negative. The fitting data validation
requires finiteness but does not reject a negative observed fluctuation.

### Auxiliary measurement

For every parameter with $P_{jj}>0$ and
`fluctuate_auxiliary=True`, SGHF draws

$$
a_{rj}^\ast
\sim
N\!\left(\theta_j^\ast,\frac{1}{P_{jj}}\right).
$$

Parameters with zero precision have
$a_{rj}^\ast=\theta_j^\ast$. With
`fluctuate_auxiliary=False`, all auxiliary entries are fixed to the truth
vector.

The auxiliary generation mean is the requested truth value, not
`model.constraint_mean`. This is the standard conditional pseudo-experiment
construction within SGHF's Gaussian auxiliary model.

### Random-number control

`rng` accepts:

- an integer seed, passed to `numpy.random.default_rng`;
- an existing `numpy.random.Generator`, whose state is advanced;
- `None`, which creates a generator without a user-fixed seed.

An integer seed makes a repeated call reproducible for the same model,
arguments, NumPy behavior, and call order. Exclusion and discovery calculators
use one generator sequentially for their two ensembles.

## The nominal affine POI estimator

The nominal shortcut uses

$$
V=\operatorname{diag}(\boldsymbol{\nu}_0)
$$

and removes parameters marked fixed in the model. Let $F$ denote the remaining
free parameters and let $A_F$ and $P_F$ be the corresponding response and
diagonal precision.

Define the nominal information matrix

$$
I
=
A_F^\mathsf TV^{-1}A_F
+
P_F.
$$

Let $\mathbf e_\mu$ select the POI coordinate in the free parameter vector.
SGHF solves

$$
I\mathbf c=\mathbf e_\mu.
$$

Because $I$ is symmetric, $\mathbf c$ is the POI column of $I^{-1}$. The
nominal POI variance is

$$
\sigma_\mu^2=c_\mu.
$$

The main-data and auxiliary weights are

$$
\mathbf w_n
=
V^{-1}A_F\mathbf c,
\qquad
\mathbf w_a
=
P_F\mathbf c.
$$

For each pseudo-experiment, the POI estimator is

$$
\boxed{
\hat\mu
=
\mu_0
+
\mathbf w_n^\mathsf T
(\mathbf n-\boldsymbol{\nu}_0)
+
\mathbf w_a^\mathsf T
(\mathbf a_F-\boldsymbol{\theta}_{0,F})
}.
$$

The dense information-matrix rank check and solve are independent of the
ensemble size. Once the two weight vectors are known, a batch of toys is
evaluated by matrix-vector products.

### Exact scope of the shortcut

This estimator is the unbounded affine POI estimator of SGHF's nominal
quadratic model. It:

- includes Gaussian nuisance constraints through $P_F$;
- excludes parameters marked fixed by the model;
- ignores all free-parameter box bounds, including nuisance bounds;
- uses a closed-form POI-bound correction only in the piecewise
  $\tilde q_\mu$ statistic below.

The resulting statistics are exact for the nominal quadratic model when the
unbounded nuisance solution is the relevant solution. They are not exact for
the nonlinear HistFactory likelihood, and they can differ from
`Model.fit(variance="nominal")` when nuisance or POI bounds become active.

SGHF raises `NonIdentifiableModel` if $I$ is rank deficient or if
$\sigma_\mu^2$ is not finite and positive. The POI must be among the model's
free parameters.

## Vectorized nominal statistics

### Discovery $q_0$

For `poi_test=0`,

$$
q_0
=
\begin{cases}
\hat\mu^2/\sigma_\mu^2, & \hat\mu\ge0,\\
0, & \hat\mu<0.
\end{cases}
$$

### Upper test $q_\mu$

$$
q_\mu
=
\begin{cases}
(\mu-\hat\mu)^2/\sigma_\mu^2, & \hat\mu\le\mu,\\
0, & \hat\mu>\mu.
\end{cases}
$$

### Bounded upper test $\tilde q_\mu$

`qtilde` requires the loaded POI lower bound to be exactly zero and uses

$$
\tilde q_\mu
=
\begin{cases}
0,
& \hat\mu>\mu,\\[3pt]
(\mu-\hat\mu)^2/\sigma_\mu^2,
& 0\le\hat\mu\le\mu,\\[3pt]
(\mu^2-2\mu\hat\mu)/\sigma_\mu^2,
& \hat\mu<0.
\end{cases}
$$

The last branch is the quadratic difference between a numerator fixed at
$\mu$ and a denominator constrained to the physical point zero, expressed in
terms of the unbounded affine estimate.

All returned statistics are clipped at zero against floating-point
round-off.

## Iterated per-toy path

With `variance="iterated"`, SGHF does not use the affine shortcut. For every
toy it calls the profile-statistic fit path with that toy's main and auxiliary
observations. The variance iteration, bounds, and failures are therefore
those described in [fitting](fitting.md) and
[hypothesis testing](hypothesis-testing.md).

This path scales with the number of toys times the fit cost. It is provided to
study the effect of reweighting, not as the default fast calculation.

For observed `q0` in this path, the discovery result stores

$$
\max\!\left[
0,\chi^2(\hat{\hat{\boldsymbol{\theta}}}(0))
-\chi^2(\hat{\boldsymbol{\theta}})
\right].
$$

The observed-result assignment does not reapply the negative-$\hat\mu$
clipping after this difference. With the usual physical POI lower bound of
zero this is equivalent to the defined $q_0$. For a model that permits a
negative free POI, use nominal toy mode or verify the iterated observed
statistic explicitly.

## Empirical tail probabilities

Let $T_1,\ldots,T_N$ be an ensemble of statistics and $t_\mathrm{obs}$ the
threshold. SGHF counts the inclusive upper tail,

$$
k
=
\#\{r:T_r\ge t_\mathrm{obs}\},
$$

and returns

$$
\boxed{
\hat p
=
\frac{k+1}{N+1}
}.
$$

The correction prevents a zero estimate when no simulated statistic reaches
the threshold; see the Monte Carlo calibration discussion in the
[references](references.md). Its immediate consequences are:

- the minimum returned tail probability is $1/(N+1)$;
- the maximum is one;
- ties at the observed threshold are included;
- SGHF does not report a binomial Monte Carlo interval.

For scale, the ordinary binomial standard error
$\sqrt{p(1-p)/N}$ is a useful approximation away from the boundaries, but the
two tails entering a $CL_s$ ratio and the expected-band construction are
correlated. Repeat seeds or larger ensembles are needed when Monte Carlo
variation is scientifically relevant.

## Toy exclusion

For `test_stat="q"` or `"qtilde"`, SGHF first fits the supplied observed data.

### Ensemble truths

The signal-plus-background truth is the observed conditional fit at the
tested value:

$$
\boldsymbol{\theta}_{s+b}^\ast
=
\hat{\hat{\boldsymbol{\theta}}}(\mu).
$$

The background truth is a separate observed conditional fit at zero:

$$
\boldsymbol{\theta}_{b}^\ast
=
\hat{\hat{\boldsymbol{\theta}}}(0).
$$

Each truth supplies both the main expectation and the center of the Gaussian
auxiliary fluctuations for its ensemble. The background fit requires zero to
be inside its effective model bounds.

### Observed probabilities

Let $q_\mathrm{obs}$ be the selected observed statistic and let
$\{q_{s+b,r}\}$ and $\{q_{b,r}\}$ be the two toy ensembles. SGHF returns

$$
\widehat{CL}_{s+b}
=
\frac{
1+\#\{r:q_{s+b,r}\ge q_\mathrm{obs}\}
}{
N+1
},
$$

$$
\widehat{CL}_{b}
=
\frac{
1+\#\{r:q_{b,r}\ge q_\mathrm{obs}\}
}{
N+1
},
$$

and

$$
\boxed{
\widehat{CL}_s
=
\frac{\widehat{CL}_{s+b}}{\widehat{CL}_b}
}.
$$

The ratio is not clipped to $[0,1]$.

In nominal mode, `q` in the returned result is recomputed with the vectorized
affine estimator and `model.constraint_mean` as the observed auxiliary vector.
The stored `free_fit` and `fixed_fit` are diagnostic bounded fits, so their
`chi2` difference can disagree with `q` when a bound is active.

### Expected exclusion bands

Each background statistic is treated in turn as a threshold. SGHF computes
its corrected signal-plus-background tail, its corrected background tail, and
their ratio. The returned expected tuple contains percentiles

$$
(2.27501319,\ 15.86552539,\ 50,\
84.13447461,\ 97.72498681)
$$

of those background-ensemble $CL_s$ values. It is reported in conventional
expected-limit order:

```text
(-2 sigma, -1 sigma, median, +1 sigma, +2 sigma).
```

The bands include finite-ensemble discreteness and correlations from reusing
the same background ensemble as both thresholds and reference distribution.

## Toy discovery with $q_0$

Toy-based discovery is explicitly supported. It requires
`poi_test=0` and returns `ToyDiscoveryTestResult`.

### Null and alternative truths

The null truth is the observed conditional fit at $\mu=0$:

$$
\boldsymbol{\theta}_0^\ast
=
\hat{\hat{\boldsymbol{\theta}}}(0).
$$

The expected-discovery alternative is the observed conditional fit at
$\mu=1$:

$$
\boldsymbol{\theta}_1^\ast
=
\hat{\hat{\boldsymbol{\theta}}}(1).
$$

SGHF generates one `q0` ensemble under each conditional truth. The value one
is fixed by the implementation; it is not currently a configurable discovery
alternative.

### Observed $p_0$ and significance

With observed statistic $q_{0,\mathrm{obs}}$ and sorted null statistics
$\{q_{0,r}^{(0)}\}$, SGHF returns

$$
\boxed{
\hat p_0
=
\frac{
1+\#\{r:q_{0,r}^{(0)}\ge q_{0,\mathrm{obs}}\}
}{
N+1
}
}.
$$

The empirical Gaussian-equivalent significance is

$$
\boxed{
Z_0
=
\max\!\left[\Phi^{-1}(1-\hat p_0),0\right]
}.
$$

The non-negative clipping means a downward fluctuation returns zero rather
than a negative discovery significance.

In nominal mode, the observed `q` is computed by the same unbounded affine
estimator with `model.constraint_mean` as auxiliary data. The stored
`free_fit` and `fixed_fit` remain bounded diagnostic fits, so—as for
exclusion—their `chi2` difference need not reproduce the returned `q` when a
bound is active.

### Expected discovery distribution

For every alternative statistic $q_{0,r}^{(1)}$, SGHF computes its corrected
tail probability against the same null ensemble. `expected_p0` contains the
percentiles

$$
(2.27501319,\ 15.86552539,\ 50,\
84.13447461,\ 97.72498681)
$$

of those alternative $p_0$ values. `expected_significance` applies
$\max[\Phi^{-1}(1-p_0),0]$ to the tuple in the same order.

Because $p_0$ decreases as significance increases, this tuple is ordered from
the smallest $p_0$ and largest $Z$ to the largest $p_0$ and smallest $Z$. If
Gaussian fluctuation labels are attached, the significance order is

```text
(+2 sigma, +1 sigma, median, -1 sigma, -2 sigma).
```

This is intentionally different from the expected-limit ordering of the
exclusion tuple.

## Batching and memory

The calculator generates and evaluates each ensemble in batches. An explicit
positive integer `batch_size` is capped at `ntoys`. With `batch_size=None`,
SGHF estimates

$$
B_\mathrm{toy}
=
8\left(
N_\mathrm{bins}+N_\mathrm{parameters}
\right)\ \text{bytes}
$$

for the two principal temporary toy rows and selects

$$
N_\mathrm{batch}
=
\min\!\left[
N_\mathrm{toys},
\max\!\left(
1,
\left\lfloor
\frac{128\ \mathrm{MiB}}{B_\mathrm{toy}}
\right\rfloor
\right)
\right].
$$

This is a target based on main and auxiliary toy matrices, not an exact
process-memory bound. Generator temporaries, sorting, fits, and library
overhead require additional memory.

The final result retains two length-`ntoys` statistic arrays:

- `sb_statistics` and `b_statistics` for exclusion;
- `null_statistics` and `alternative_statistics` for discovery.

Batching therefore bounds the large temporary observation matrices but does
not make the result memory independent of `ntoys`.

## Result objects

### `ToyResult`

| Field | Meaning |
|---|---|
| `observations` | shape `(size, nbins)`; integer for Poisson, float for Gaussian |
| `auxiliary` | shape `(size, nparameters)` |
| `parameters` | truth vector used for generation |
| `expected` | compiled expected yields at that truth |
| `sampling` | `"poisson"` or `"gaussian"` |

### `ToyHypothesisTestResult`

| Field | Meaning |
|---|---|
| `poi_test`, `test_stat`, `q` | tested value, `q`/`qtilde`, and observed statistic |
| `clsb`, `clb`, `cls` | corrected empirical tails and ratio |
| `expected_cls` | five background-ensemble percentiles in expected-limit order |
| `ntoys`, `batch_size`, `variance` | ensemble and calculation configuration |
| `sb_statistics`, `b_statistics` | retained statistic ensembles |
| `free_fit`, `fixed_fit` | observed diagnostic free and tested-POI fits |

The background conditional fit used to generate `b_statistics` is not stored
as a public field.

### `ToyDiscoveryTestResult`

| Field | Meaning |
|---|---|
| `poi_test`, `test_stat`, `q` | always zero, `"q0"`, and observed statistic |
| `p0`, `significance` | corrected null tail and non-negative Gaussian-equivalent $Z_0$ |
| `expected_p0`, `expected_significance` | alternative-ensemble percentiles in the ordering described above |
| `ntoys`, `batch_size`, `variance` | ensemble and calculation configuration |
| `null_statistics`, `alternative_statistics` | retained `q0` ensembles |
| `free_fit`, `fixed_fit` | observed free and background-only fits |
| `alternative_fit` | observed conditional fit at $\mu=1$ |

## Applicability

A toy result is directly calibrated only for:

- the compiled SGHF mean and Gaussian auxiliary model;
- the selected Poisson or Gaussian main generator;
- the implemented nominal or iterated statistic;
- the conditional truth construction used by the calculator;
- the finite number of generated toys.

It is not a coverage statement for the nonlinear pyhf workspace. For final
results, compare the relevant response range and, when tails or boundaries
matter, the statistic distribution against the original model. The structured
checks are listed in
[validation and applicability](validation-and-applicability.md).

---

[Documentation index](index.md) | Previous: [Hypothesis testing](hypothesis-testing.md) | Next: [Upper limits](upper-limits.md)
