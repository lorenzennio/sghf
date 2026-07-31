# SGHF API reference

Complete signatures, result fields and the exception tree. Defaults shown are
the real ones. Full derivations: https://github.com/lorenzennio/sghf/tree/main/docs

## Loading

```python
sghf.load(source, *, measurement=None, strict=False) -> Model
```

`source` is a decoded mapping, a filesystem path, a path-like object, or a
readable text stream. `measurement` selects by name when a workspace defines
several; the default takes the first and emits `WorkspaceWarning`. `strict=True`
rejects unknown top-level fields instead of warning.

Supported modifiers: `normfactor`, `normsys`, `histosys`, `staterror`,
`shapesys`, `lumi`, in workspace schema `"1.0.0"`. Anything else raises
`UnsupportedModifier`.

## Model

Frozen dataclass. Arrays it holds are not read-only; treat them as immutable.

| Field | Meaning |
|---|---|
| `channel_names` | channel order (sorted) |
| `channel_slices` | flattened slice per channel |
| `parameter_names` | scalar parameter order; vector parameters appear as `name[i]` |
| `poi_name` | parameter of interest |
| `nominal_expected` | expected yields at the expansion point |
| `response` | response matrix, bins by parameters |
| `observations` | flattened observed data |
| `initial`, `lower_bounds`, `upper_bounds`, `fixed` | parameter configuration |
| `constraint_mean`, `constraint_precision` | Gaussian constraint, in parameter space |
| `modifier_types`, `warnings` | what compiled, and accepted loading warnings |

Methods:

```python
model.approximation                  -> ApproximationReport
model.parameter_index(name)          -> int          # KeyError if unknown
model.response_for(name)             -> ndarray      # copy of one response column
model.expected(parameters)           -> ndarray      # parameters: sequence, mapping, or FitResult
```

`model.parameter_names` order is SGHF's own; it does not reproduce pyhf's.
Always address parameters by name.

## Fitting

```python
model.fit(
    data=None,           # None, flat bin sequence, or {channel: [...]} mapping
    *,
    auxdata=None,        # None, full parameter-length sequence, or {name: value}
    fixed=None,          # {name: value}
    bounds=None,         # {name: (low, high)}
    variance="nominal",  # or "iterated"
    rtol=1.0e-6,
    max_iter=25,
) -> FitResult
```

`FitResult`:

| Field | Meaning |
|---|---|
| `values`, `errors` | arrays in `parameter_names` order |
| `value(name)`, `error(name)`, `result[name]` | scalar access by name |
| `covariance`, `correlation` | full matrices |
| `pulls` | `(value - constraint_mean) * sqrt(precision)`; `nan` for unconstrained |
| `expected` | post-fit expected yields |
| `chi2` | objective at the minimum |
| `converged`, `iterations` | `iterations == 1` for `variance="nominal"` |
| `at_bound` | names sitting on a bound |

Parameters in `at_bound` have zero covariance entries, so `error(name)` returns
`0.0` — that is not a real uncertainty. See [pitfalls.md](pitfalls.md).

## Hypothesis tests

```python
model.hypotest(
    poi_test,
    data=None,
    *,
    test_stat="qtilde",        # "q0" | "q" | "qtilde"
    calculator="asymptotic",   # or "toys"
    variance="nominal",
    ntoys=10_000,              # toys only
    batch_size=None,
    sampling="poisson",        # or "gaussian"
    rng=None,
)
```

Preconditions: `q0` requires `poi_test=0`; `qtilde` requires the POI lower bound
to be exactly zero. Both raise `ValueError` otherwise.

`HypothesisTestResult` (asymptotic): `poi_test`, `test_stat`, `q`, `q_asimov`,
`transformed`, `clsb`, `clb`, `cls`, `expected_cls`, `significance`, `free_fit`,
`fixed_fit`, `asimov_free_fit`, `asimov_fixed_fit`.

**There is no `p0` field on the asymptotic result.** For `test_stat="q0"` the
discovery p-value is `clsb`, which satisfies `clsb == norm.sf(significance)`
exactly. `cls` and `clb` are populated but are exclusion quantities and are
meaningless for `q0` — reading `cls` as the discovery p-value is a silent
error of orders of magnitude. Only `ToyDiscoveryTestResult` names `p0`
directly.

| `q0` p-value from | Field |
|---|---|
| `calculator="asymptotic"` | `result.clsb` (or `norm.sf(result.significance)`) |
| `calculator="toys"` | `result.p0` |

`ToyHypothesisTestResult` (`calculator="toys"`, `q`/`qtilde`): `poi_test`,
`test_stat`, `q`, `clsb`, `clb`, `cls`, `expected_cls`, `ntoys`, `batch_size`,
`variance`, `sb_statistics`, `b_statistics`, `free_fit`, `fixed_fit`.

`ToyDiscoveryTestResult` (`calculator="toys"`, `q0`): `p0`, `significance`,
`expected_p0`, `expected_significance`, `null_statistics`,
`alternative_statistics`, plus `free_fit`, `fixed_fit`, `alternative_fit`.

`expected_cls` runs `(-2σ, -1σ, median, +1σ, +2σ)`. `expected_significance` runs
the opposite way — most significant first.

## Upper limits

```python
model.upper_limit(
    data=None,
    *,
    level=0.05,
    test_stat="qtilde",        # or "q"
    calculator="asymptotic",   # or "toys"
    variance="nominal",
    bracket=None,              # (low, high); toys read it as the scan range
    scan=None,                 # explicit toy grid
    points=21,                 # toy grid size when scan is absent
    ntoys=10_000,
    batch_size=None,
    sampling="poisson",
    rng=None,
    keep_results=False,
) -> LimitResult
```

`LimitResult`: `observed`, `expected` (five entries, `-2σ` first), `level`,
`test_stat`, `bracket`, `calculator`, `scan_values`, `results`.

`results` is empty unless `keep_results=True`. The two calculators locate
crossings differently — see [pitfalls.md](pitfalls.md).

## Pseudo-experiments

```python
model.toys(
    size,
    *,
    parameters=None,            # None, sequence, {name: value}, or FitResult
    sampling="poisson",         # or "gaussian"
    fluctuate_auxiliary=True,
    rng=None,                   # int seed or numpy Generator
) -> ToyResult
```

`ToyResult`: `observations` `(size, nbins)`, `auxiliary` `(size, nparameters)`,
`parameters`, `expected`, `sampling`. An integer seed makes generation
reproducible. `sampling="gaussian"` can produce negative counts, which the
quadratic fitter accepts without clipping.

## Exceptions

```
SGHFError
├── SGHFUsageError (also AttributeError)  pyhf-style call; use the object API
├── InvalidWorkspace (also ValueError)    workspace cannot be compiled safely
│   └── UnsupportedModifier              modifier outside the supported set
├── FitError (also RuntimeError)
│   ├── NonIdentifiableModel             free parameters not identifiable
│   ├── ModelDomainError                 linearized yields left the valid domain
│   └── FitConvergenceError              iterated variances did not converge
└── InferenceError (also RuntimeError)
    └── LimitNotFound                    no crossing bracketed
```

`WorkspaceWarning` (a `UserWarning`) is emitted for accepted but nonstandard
workspace content, such as an unselected multi-measurement workspace.

| Raised | Usual cause | Action |
|---|---|---|
| `UnsupportedModifier` | modifier outside the six supported | use pyhf for this workspace |
| `NonIdentifiableModel` | degenerate parameters, rank deficiency | fix or remove the degeneracy |
| `ModelDomainError` | a linearized yield went non-positive | the linear model is outside its domain; do not extrapolate |
| `FitConvergenceError` | `variance="iterated"` oscillating | raise `max_iter`, or use `variance="nominal"` |
| `LimitNotFound` | no CLs crossing in range | widen `bracket`, or `scan` for toys |
| `SGHFUsageError` | `pyhf.infer.hypotest(...)` style call | use `model.hypotest(...)` |
