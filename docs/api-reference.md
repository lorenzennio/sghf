[Documentation index](index.md) | Previous: [Upper limits](upper-limits.md) | Next: [Validation and applicability](validation-and-applicability.md)

# API reference

SGHF exposes one loading function and an object-oriented inference API. Load a
workspace with `sghf.load`, then call methods on the returned `Model`.
Functions in `sghf.fit`, `sghf.asymptotic`, and `sghf.toys` that begin with an
underscore are implementation details and are not part of this public
contract.

## Complete example workspace

The examples on this page use this one-bin model:

```python
import json
from pathlib import Path

import numpy as np
import sghf

workspace = {
    "version": "1.0.0",
    "channels": [
        {
            "name": "single_bin",
            "samples": [
                {
                    "name": "signal",
                    "data": [5.0],
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "data": None}
                    ],
                },
                {
                    "name": "background",
                    "data": [20.0],
                    "modifiers": [],
                },
            ],
        }
    ],
    "observations": [{"name": "single_bin", "data": [23.0]}],
    "measurements": [
        {
            "name": "measurement",
            "config": {
                "poi": "mu",
                "parameters": [
                    {"name": "mu", "bounds": [[0.0, 10.0]], "inits": [1.0]}
                ],
            },
        }
    ],
}

model = sghf.load(workspace)
```

To load the same object from JSON:

```python
workspace_path = Path("/tmp/sghf-workspace.json")
workspace_path.write_text(json.dumps(workspace), encoding="utf-8")
model_from_path = sghf.load(workspace_path)
```

## `sghf.load`

```python
sghf.load(source, *, measurement=None, strict=False) -> Model
```

Parameters:

- `source`: decoded mapping, filesystem path, path-like object, or readable
  text stream containing a pyhf workspace;
- `measurement`: measurement name, or `None` to select the first;
- `strict`: if `True`, reject unknown top-level fields instead of warning.

The loader supports workspace version `"1.0.0"` and the modifiers listed in
[workspace compilation](workspace-compilation.md). It returns a compiled
`Model`.

Principal failures are `InvalidWorkspace`, `UnsupportedModifier`, and
`WorkspaceWarning`.

## `Model`

`Model` is a frozen dataclass container. Attribute reassignment is blocked,
but contained NumPy arrays and dictionaries are not made read-only; callers
should treat them as immutable to avoid invalidating model consistency.

### Fields

| Field | Type | Meaning |
|---|---|---|
| `channel_names` | `tuple[str, ...]` | sorted channel order |
| `channel_slices` | `dict[str, slice]` | flattened slice for each channel |
| `parameter_names` | `tuple[str, ...]` | scalar parameter order |
| `poi_name` | `str` | parameter-of-interest name |
| `nominal_expected` | float array | $\boldsymbol{\nu}_0$ |
| `response` | float array | response matrix $A$ |
| `observations` | float array | flattened observed main data |
| `initial` | float array | expansion point $\boldsymbol{\theta}_0$ |
| `lower_bounds`, `upper_bounds` | float arrays | scalar parameter bounds |
| `fixed` | Boolean array | workspace-level fixed flags |
| `constraint_mean` | float array | Gaussian auxiliary means |
| `constraint_precision` | float array | diagonal Gaussian precisions |
| `modifier_types` | `tuple[str, ...]` | encountered modifier types |
| `warnings` | `tuple[str, ...]` | accepted loading warnings |

### `Model.approximation`

```python
model.approximation -> ApproximationReport
```

Returns a concise report with modifier types, scalar parameter names,
`variance_default="nominal"`, and loading warnings. It does not estimate
approximation error.

### `Model.parameter_index`

```python
model.parameter_index(name: str) -> int
```

Returns the scalar column index for `name`. Unknown names raise `KeyError`.

### `Model.response_for`

```python
model.response_for(name: str) -> numpy.ndarray
```

Returns a copy of the expected-yield response column for one scalar parameter.

```python
mu_index = model.parameter_index("mu")
mu_response = model.response_for("mu")

assert mu_index == 0
np.testing.assert_allclose(mu_response, [5.0])
```

### `Model.expected`

```python
model.expected(parameters) -> numpy.ndarray
```

`parameters` can be:

- a full scalar sequence;
- a partial name-to-value mapping, applied to a copy of `model.initial`;
- a `FitResult`.

The result is

$$
\boldsymbol{\nu}_0
+
A(\boldsymbol{\theta}-\boldsymbol{\theta}_0).
$$

Wrong shapes or non-finite parameter values raise `ValueError`.
Non-finite or non-positive expected yields raise `ModelDomainError`.

```python
np.testing.assert_allclose(model.expected({"mu": 0.0}), [20.0])
np.testing.assert_allclose(model.expected({"mu": 2.0}), [30.0])
```

### `Model.fit`

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
) -> FitResult
```

Arguments:

- `data`: `None`, flat bin sequence, or exact channel-name mapping;
- `auxdata`: `None`, full parameter-length sequence, or constrained-name
  mapping;
- `fixed`: name-to-value mapping for this fit;
- `bounds`: name-to-`(low, high)` mapping for this fit;
- `variance`: `"iterated"` or `"nominal"`;
- `rtol`: positive expected-variance convergence tolerance;
- `max_iter`: positive maximum number of weighted solves.

See [fitting](fitting.md) for the objective, bounds, covariance, and failures.

```python
fit = model.fit(variance="nominal")
assert isinstance(fit, sghf.FitResult)
np.testing.assert_allclose(fit.value("mu"), 0.6)
np.testing.assert_allclose(model.expected(fit), fit.expected)
```

### `Model.hypotest`

```python
model.hypotest(
    poi_test,
    data=None,
    *,
    test_stat="qtilde",
    calculator="asymptotic",
    variance="nominal",
    ntoys=10_000,
    batch_size=None,
    sampling="poisson",
    rng=None,
) -> HypothesisTestResult | ToyHypothesisTestResult | ToyDiscoveryTestResult
```

Arguments:

- `poi_test`: finite tested POI value; must be zero for `q0`;
- `data`: `None`, flat bin sequence, or exact channel mapping;
- `test_stat`: `"q0"`, `"q"`, or `"qtilde"`;
- `calculator`: `"asymptotic"` or `"toys"`;
- `variance`: `"iterated"` or `"nominal"`;
- `ntoys`: positive integer toys per generated ensemble;
- `batch_size`: positive toy batch size or `None` for automatic selection;
- `sampling`: `"poisson"` or `"gaussian"` main-toy generation;
- `rng`: integer seed, `numpy.random.Generator`, or `None`.

Both calculators default to `variance="nominal"`, so the observed `q` agrees
between them and their $CL_s$ values are directly comparable. Passing
`variance="iterated"` is supported by both, but for `calculator="toys"` it
selects the per-toy fit path and is orders of magnitude slower.

The asymptotic calculator does not use `ntoys`, `batch_size`, `sampling`, or
`rng`. See [hypothesis testing](hypothesis-testing.md) and
[toy inference](toy-inference.md).

```python
asymptotic = model.hypotest(
    1.0,
    test_stat="qtilde",
    calculator="asymptotic",
    variance="nominal",
)
assert isinstance(asymptotic, sghf.HypothesisTestResult)

toy_exclusion = model.hypotest(
    1.0,
    test_stat="qtilde",
    calculator="toys",
    ntoys=200,
    batch_size=50,
    rng=12345,
)
assert isinstance(toy_exclusion, sghf.ToyHypothesisTestResult)

toy_discovery = model.hypotest(
    0.0,
    test_stat="q0",
    calculator="toys",
    ntoys=200,
    batch_size=50,
    rng=12345,
)
assert isinstance(toy_discovery, sghf.ToyDiscoveryTestResult)
```

### `Model.toys`

```python
model.toys(
    size,
    *,
    parameters=None,
    sampling="poisson",
    fluctuate_auxiliary=True,
    rng=None,
) -> ToyResult
```

Arguments:

- `size`: positive integer number of pseudo-experiments;
- `parameters`: `None`, full sequence, partial mapping, or `FitResult`;
- `sampling`: `"poisson"` or `"gaussian"`;
- `fluctuate_auxiliary`: whether constrained auxiliary observations fluctuate;
- `rng`: integer seed, `numpy.random.Generator`, or `None`.

```python
generated = model.toys(
    10,
    parameters={"mu": 0.0},
    rng=12345,
)
assert generated.observations.shape == (10, 1)
assert generated.auxiliary.shape == (10, 1)
```

### `Model.upper_limit`

```python
model.upper_limit(
    data=None,
    *,
    level=0.05,
    test_stat="qtilde",
    variance="nominal",
    bracket=None,
) -> LimitResult
```

Arguments:

- `data`: `None`, flat bin sequence, or exact channel mapping;
- `level`: target $CL_s$ strictly between zero and one;
- `test_stat`: `"q"` or `"qtilde"`;
- `variance`: `"iterated"` or `"nominal"`;
- `bracket`: finite `(low, high)` pair or `None` for automatic expansion.

This method is asymptotic only. See [upper limits](upper-limits.md).

```python
limit = model.upper_limit(variance="nominal")
assert isinstance(limit, sghf.LimitResult)
assert np.isfinite(limit.observed)
```

## Result types

All result classes are frozen dataclasses, but any NumPy arrays they contain
remain mutable. Treat returned arrays as read-only analysis records unless a
copy is made.

### `ApproximationReport`

| Field | Type |
|---|---|
| `modifier_types` | `tuple[str, ...]` |
| `parameter_names` | `tuple[str, ...]` |
| `variance_default` | literal `"nominal"` |
| `warnings` | `tuple[str, ...]` |

### `FitResult`

| Field | Type |
|---|---|
| `parameter_names` | `tuple[str, ...]` |
| `values`, `errors`, `covariance`, `correlation`, `pulls`, `expected` | float arrays |
| `chi2` | `float` |
| `converged` | `bool` |
| `iterations` | `int` |
| `at_bound` | `tuple[str, ...]` |

Methods:

```python
fit.parameter_index(name) -> int
fit.value(name) -> float
fit.error(name) -> float
fit[name] -> float
```

### `HypothesisTestResult`

| Field | Type or meaning |
|---|---|
| `poi_test` | `float` |
| `test_stat` | `"q0"`, `"q"`, or `"qtilde"` |
| `q`, `q_asimov`, `transformed`, `clsb`, `clb`, `cls`, `significance` | `float` |
| `expected_cls` | five-float tuple |
| `free_fit`, `fixed_fit`, `asimov_free_fit`, `asimov_fixed_fit` | `FitResult` |

### `LimitResult`

| Field | Type |
|---|---|
| `observed` | `float` |
| `expected` | five-float tuple |
| `level` | `float` |
| `test_stat` | `"q"` or `"qtilde"` |
| `bracket` | two-float tuple |

### `ToyResult`

| Field | Type or meaning |
|---|---|
| `observations` | integer or float array |
| `auxiliary`, `parameters`, `expected` | float arrays |
| `sampling` | `"poisson"` or `"gaussian"` |

### `ToyHypothesisTestResult`

| Field | Type or meaning |
|---|---|
| `poi_test` | `float` |
| `test_stat` | `"q"` or `"qtilde"` |
| `q`, `clsb`, `clb`, `cls` | `float` |
| `expected_cls` | five-float tuple |
| `ntoys`, `batch_size` | `int` |
| `variance` | `"nominal"` or `"iterated"` |
| `sb_statistics`, `b_statistics` | float arrays |
| `free_fit`, `fixed_fit` | `FitResult` |

### `ToyDiscoveryTestResult`

| Field | Type or meaning |
|---|---|
| `poi_test` | `float`, always zero |
| `test_stat` | `"q0"` |
| `q`, `p0`, `significance` | `float` |
| `expected_p0`, `expected_significance` | five-float tuples |
| `ntoys`, `batch_size` | `int` |
| `variance` | `"nominal"` or `"iterated"` |
| `null_statistics`, `alternative_statistics` | float arrays |
| `free_fit`, `fixed_fit`, `alternative_fit` | `FitResult` |

Field semantics and tuple orderings are detailed on the relevant method pages.

## Exceptions and warnings

The package-root exports are:

| Name | Base and use |
|---|---|
| `SGHFUsageError` | `AttributeError`; pyhf-style call guidance |
| `InvalidWorkspace` | `ValueError`; workspace cannot be compiled safely |
| `UnsupportedModifier` | `InvalidWorkspace`; modifier type is unsupported |
| `WorkspaceWarning` | `UserWarning`; accepted workspace ambiguity or metadata |
| `FitError` | `RuntimeError`; base for exported fit failures |
| `NonIdentifiableModel` | `FitError`; free information matrix is rank deficient |
| `ModelDomainError` | `FitError`; linear expectation is invalid |
| `FitConvergenceError` | `FitError`; bounded or iterated fit did not converge |
| `InferenceError` | `RuntimeError`; base for exported inference failures |
| `LimitNotFound` | `InferenceError`; requested $CL_s$ root was not bracketed |

Input-contract errors not represented by a more specific SGHF exception use
standard `TypeError`, `ValueError`, or `KeyError`.

## Guidance for pyhf-style calls

SGHF intentionally uses a smaller object API. Common pyhf-style entry points
are intercepted to raise `SGHFUsageError` with a replacement:

| Attempted style | SGHF replacement |
|---|---|
| `sghf.Workspace(workspace)` | `model = sghf.load(workspace)` |
| `sghf.infer.mle.fit(...)` | `model.fit(...)` |
| `sghf.infer.hypotest(...)` | `model.hypotest(...)` |
| a path ending in `.upper_limit(...)` | `model.upper_limit(...)` |

```python
try:
    sghf.infer.hypotest(1.0)
except sghf.SGHFUsageError as error:
    assert "model.hypotest" in str(error)
```

These objects provide migration guidance only. SGHF does not present a
drop-in pyhf namespace, tensor backend, or complete HistFactory API.

The installed version string is available as `sghf.__version__`.

---

[Documentation index](index.md) | Previous: [Upper limits](upper-limits.md) | Next: [Validation and applicability](validation-and-applicability.md)
