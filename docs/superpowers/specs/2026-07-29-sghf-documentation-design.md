# SGHF Theory-First Documentation Design

## Objective

Create a layered, multi-page documentation set that explains the statistical
model SGHF actually implements, derives its inference procedures with correct
notation, and documents every public method and result type. The documentation
must make the boundary between exact pyhf/HistFactory semantics and SGHF's
linear-Gaussian approximation explicit.

## Audience and layering

The primary readers are HEP analysts familiar with pyhf or HistFactory and
statistically literate readers who may not know those tools.

Every conceptual page uses the same three layers:

1. An operational summary stating what SGHF does and when a reader would use
   the feature.
2. A mathematical derivation defining all symbols and connecting each equation
   to the corresponding statistical object.
3. Implementation and validity notes identifying the relevant SGHF behavior,
   numerical choices, approximations, diagnostics, and failure modes.

Pages may include short code examples after the mathematical definition. Code
examples must use the public object API and must not imply exact pyhf parity.

## Documentation architecture

### `docs/index.md`

The documentation landing page provides navigation, scope, a five-minute
workflow, and a notation table. It distinguishes workspace compilation,
fitting, asymptotic inference, and pseudo-experiment inference.

### `docs/simplified-likelihood.md`

This page starts from the binned HistFactory model and derives SGHF's compiled
model

$$
\boldsymbol{\nu}(\boldsymbol{\theta})
\approx
\boldsymbol{\nu}_0
+ A(\boldsymbol{\theta}-\boldsymbol{\theta}_0).
$$

It then defines the quadratic objective

$$
\chi^2(\boldsymbol{\theta})
=
(\mathbf{n}-\boldsymbol{\nu}(\boldsymbol{\theta}))^\mathsf{T}
V^{-1}
(\mathbf{n}-\boldsymbol{\nu}(\boldsymbol{\theta}))
+
(\boldsymbol{\theta}-\mathbf{a})^\mathsf{T}
P
(\boldsymbol{\theta}-\mathbf{a}),
$$

where the documentation must define the observations, nominal expectations,
response matrix, auxiliary observations, constraint precision, and variance
matrix. It covers nominal and iterated diagonal variances and explicitly lists
all approximations:

- local first-order response in every parameter;
- diagonal Gaussian approximation to the main Poisson measurement in the
  fitted objective;
- Gaussian approximation to all supported auxiliary constraints;
- absence of multiplicative cross terms between simultaneous modifier shifts;
- local treatment of asymmetric modifiers through their central derivative;
- use of a parameter-independent response matrix;
- boundary handling by constrained least squares rather than the exact
  nonlinear likelihood;
- positive expected-yield requirement for evaluated linearized models.

The page gives validity conditions and failure indicators for each
approximation, rather than relying on a generic "large counts" statement.

### `docs/workspace-compilation.md`

This page documents loading, measurement selection, channel ordering, bin
flattening, parameter ordering, parameter metadata, and response construction.
For each supported modifier (`normfactor`, `normsys`, `histosys`, `staterror`,
`shapesys`, and `lumi`), it states:

- the pyhf object being approximated;
- the nominal contribution;
- the derivative stored in the response matrix;
- the constraint precision used by SGHF;
- any symmetry, interpolation, or correlation assumption;
- cases where the approximation can differ materially from pyhf.

It also documents unknown metadata behavior, strict mode, unsupported
modifiers, multiple measurements, and the contents of `Model` and
`ApproximationReport`.

### `docs/fitting.md`

This page derives the augmented weighted least-squares system, normal matrix,
right-hand side, covariance matrix, errors, correlations, pulls, and reported
chi-square. It distinguishes:

- `variance="nominal"`: one quadratic solve with
  $V=\operatorname{diag}(\boldsymbol{\nu}_0)$;
- `variance="iterated"`: iteratively reweighted solves with the diagonal
  variance updated from the fitted expectation.

It documents damping of two-cycle behavior, convergence tolerance, iteration
limit, fixed parameters, user bounds, active bounds, covariance treatment at
active constraints, identifiability checks, and typed failures.

### `docs/hypothesis-testing.md`

This page defines the profiled objective difference and the one-sided
statistics $q_0$, $q_\mu$, and $\tilde q_\mu$. It documents the exact clipping
rules used by SGHF, the physical lower-bound requirement for
$\tilde q_\mu$, conditional fits, and Asimov data construction.

It then derives the transformed statistic used by the implementation and
defines $CL_{s+b}$, $CL_b$, $CL_s$, discovery significance, and the five
expected bands. The page separates asymptotic assumptions from the earlier
linear-Gaussian model approximations.

### `docs/toy-inference.md`

This page documents:

- Poisson and Gaussian main-measurement generation;
- Gaussian auxiliary fluctuations;
- conditional null, signal-plus-background, background-only, and
  $\mu=1$ discovery-alternative ensembles;
- empirical tail probabilities with the finite-sample
  $(k+1)/(N+1)$ correction;
- expected exclusion and discovery bands;
- empirical $p_0$ and non-negative discovery significance;
- the finite resolution $1/(N+1)$;
- deterministic random generators and batching.

For the default nominal mode, it derives the affine POI estimator

$$
\hat{\mu}
=
\mu_0
+ \mathbf{w}_n^\mathsf{T}(\mathbf{n}-\boldsymbol{\nu}_0)
+ \mathbf{w}_a^\mathsf{T}
(\mathbf{a}-\boldsymbol{\theta}_0)
$$

from the inverse normal matrix and derives the vectorized Wald-equivalent
$q_0$, $q_\mu$, and $\tilde q_\mu$ formulas used in code. It clearly states
that these formulas are exact for SGHF's nominal quadratic model, not for the
original nonlinear HistFactory likelihood. It separately documents the
`variance="iterated"` per-toy profiling fallback.

### `docs/upper-limits.md`

This page defines observed and expected $CL_s=\alpha$ crossings, automatic
bracket expansion, the scalar root solve, caching, expected-band ordering,
explicit brackets, and `LimitNotFound` conditions. It states that the current
upper-limit method is asymptotic only.

### `docs/api-reference.md`

This page documents public functions, methods, arguments, defaults, return
objects, and exceptions for:

- `sghf.load`;
- `Model.expected`;
- `Model.fit`;
- `Model.hypotest`;
- `Model.toys`;
- `Model.upper_limit`;
- lookup helpers such as `parameter_index` and `response_for`;
- all public result dataclasses and their fields;
- pyhf-style compatibility guidance exposed by `sghf`.

The reference must match the current signatures and calculator-dependent
defaults, including the nominal toy variance and iterated asymptotic variance.

### `docs/validation-and-applicability.md`

This page provides a decision checklist covering response linearity, modifier
asymmetry, expected counts, nuisance pulls, active bounds, fitted-yield
positivity, covariance conditioning, nominal-versus-iterated stability, and
SGHF-versus-pyhf comparison. It explains what nominal and local response parity
does and does not validate.

It gives concrete escalation rules: final scientific results should be checked
with pyhf when the approximation diagnostics are unfavorable or the result is
sensitive to tails, boundaries, or nonlinear modifier behavior.

### `docs/references.md`

This page lists primary sources for HistFactory, pyhf, profile-likelihood and
Wald asymptotics, $CL_s$, and toy calibration. Conceptual pages link directly
to the relevant references.

## Navigation and README integration

Every page begins with links to the documentation index and the logically
previous and next pages. `docs/index.md` contains the complete table of
contents. The root README remains a concise installation and quick-start page
and links prominently to the theory, inference, toy, applicability, and API
pages.

GitHub-compatible Markdown and display math are used. No documentation
framework or runtime dependency is introduced.

## Mathematical consistency requirements

The notation is fixed across all pages:

- $\mathbf{n}$: flattened main observations;
- $\boldsymbol{\nu}_0$: nominal expected yields at
  $\boldsymbol{\theta}_0$;
- $A$: response matrix with bins as rows and scalar parameters as columns;
- $\boldsymbol{\theta}$: model parameters, including the POI;
- $\mathbf{a}$: Gaussian auxiliary observations;
- $P$: diagonal auxiliary-constraint precision;
- $V$: diagonal main-measurement variance;
- $\mu$: parameter of interest;
- $\hat{\boldsymbol{\theta}}$: free fit;
- $\hat{\hat{\boldsymbol{\theta}}}(\mu)$: conditional fit at fixed $\mu$.

Equations must be checked against the current implementation, not copied from
generic likelihood references. Whenever SGHF differs from a standard formula,
the implementation's behavior and the standard convention are both stated.

## Testing and review

Documentation verification includes:

1. A local-link and page-presence test implemented with the Python standard
   library.
2. Execution of all Python examples that are intended to be runnable.
3. A manual equation-to-code audit against `workspace.py`, `fit.py`,
   `asymptotic.py`, and `toys.py`.
4. The full existing test suite to ensure documentation-related changes do not
   alter package behavior.
5. A package build to ensure the README remains valid distribution metadata.

## Non-goals

- No package behavior or public API changes.
- No attempt to reproduce the complete pyhf or HistFactory documentation.
- No claim that SGHF gives exact frequentist coverage for the original
  nonlinear likelihood.
- No new documentation generator, theme, or dependency.
- No benchmark claims beyond measurements already verified and accompanied by
  their conditions.

## Acceptance criteria

The documentation is complete when:

- all ten pages exist and are reachable from `docs/index.md`;
- all public methods and result fields are documented;
- every implemented test statistic and calculator path has its defining
  equations;
- every compiler modifier has a documented local-response construction;
- all approximation and validity conditions are explicit;
- the README links to the documentation set;
- documentation tests, package tests, link checks, and package build pass;
- documentation changes are split into logical commits and pushed to `main`
  only after local verification.
