# SGHF

SGHF is a small linear-Gaussian inference engine for statistical models stored
as [pyhf workspace JSON](https://scikit-hep.org/pyhf/likelihood.html). It
compiles the workspace once, then reuses NumPy/SciPy matrices for fits,
profile tests, limits, and pseudo-experiments. pyhf is not a runtime
dependency.

SGHF is approximate. It does not evaluate the exact HistFactory likelihood.
Use it for rapid inference and validate final results with pyhf when counts
are low, responses are nonlinear or asymmetric, or fitted parameters approach
their bounds.

## Installation

```console
python -m pip install sghf
```

From a source checkout:

```console
python -m pip install .
```

Python 3.10 or newer is required.

## Quick start

```python
import sghf

model = sghf.load("workspace.json")

fit = model.fit()
print(fit.value(model.poi_name), fit.error(model.poi_name))
print(fit.covariance, fit.correlation)

test = model.hypotest(1.0)
print(test.cls, test.expected_cls)

limit = model.upper_limit()
print(limit.observed, limit.expected)
```

Expected `CLs` and upper-limit tuples are ordered as
`(-2 sigma, -1 sigma, median, +1 sigma, +2 sigma)`.

Loading accepts a path, a text file object, or an already-decoded mapping.
When a workspace has multiple measurements, select one explicitly:

```python
model = sghf.load(workspace, measurement="Measurement")
```

Unknown top-level metadata is ignored with `sghf.WorkspaceWarning`. Use
`strict=True` to reject it:

```python
model = sghf.load(workspace, strict=True)
```

## Documentation

The detailed documentation is organized from theory to methods:

- [documentation index](docs/index.md)
- [simplified likelihood and approximations](docs/simplified-likelihood.md)
- [workspace and modifier compilation](docs/workspace-compilation.md)
- [fitting](docs/fitting.md)
- [hypothesis testing](docs/hypothesis-testing.md)
- [toy inference, including `q0` discovery](docs/toy-inference.md)
- [upper limits](docs/upper-limits.md)
- [API reference](docs/api-reference.md)
- [validation and applicability checklist](docs/validation-and-applicability.md)

## Fits

The default fit freezes the diagonal Poisson variance at the workspace's
nominal parameter point, giving a reproducible one-step Gaussian calculation.
Pass `variance="iterated"` to instead update the variance from the current
expected counts:

```python
fit = model.fit(
    data={"signal_region": [12, 8], "control_region": [101, 97]},
    fixed={"calibration": 0.0},
    bounds={"mu": (0.0, 5.0)},
    variance="iterated",
)
```

`FitResult` exposes parameter values and uncertainties by name, the full
covariance and correlation matrices, pulls, post-fit expectations, chi-square,
iteration count, and active bounds. Singular free-parameter systems and
non-positive fitted expectations raise typed exceptions instead of being
silently regularized or clipped.

## Asymptotic inference

```python
q0 = model.hypotest(0.0, test_stat="q0")
qmu = model.hypotest(1.0, test_stat="q")
qtilde = model.hypotest(1.0, test_stat="qtilde")
limit = model.upper_limit(level=0.05, test_stat="qtilde")
```

The calculations use profiled quadratic statistics and Asimov data following
the conventions of [Cowan et al.](https://doi.org/10.1140/epjc/s10052-011-1554-0).
`qtilde` requires a parameter-of-interest lower bound of zero.

## Pseudo-experiments

```python
toys = model.toys(
    10_000,
    parameters={"mu": 0.0},
    sampling="poisson",
    rng=17,
)

toy_test = model.hypotest(
    1.0,
    calculator="toys",
    ntoys=10_000,
    rng=17,
)

discovery = model.hypotest(
    0.0,
    calculator="toys",
    test_stat="q0",
    ntoys=100_000,
    batch_size=10_000,
    rng=17,
)
print(discovery.p0, discovery.significance)
```

Poisson sampling is the default; Gaussian main-measurement sampling is also
available and can produce negative pseudo-observations, which the quadratic
fitter accepts without clipping. Constrained parameters receive independently
fluctuated Gaussian auxiliary measurements. Supplying an integer seed makes the
generated samples and toy-based result reproducible.

Toy inference uses the nominal quadratic variance, which is the library-wide
default, so its observed statistic matches the asymptotic one. It solves once
for an affine parameter-of-interest estimator and then evaluates toy
statistics with vectorized array operations. It supports the discovery
statistic `q0`, returning empirical `p0` and significance, as well as the
exclusion statistics `q` and `qtilde`. Pass `variance="iterated"` when
per-toy iterative reweighting is required; that mode profiles toys
individually and is substantially slower. Large ensembles are automatically
batched to limit temporary memory use; pass `batch_size=...` to control the
tradeoff explicitly. See [toy inference](docs/toy-inference.md) for the exact
statistics, conditional ensembles, and finite Monte Carlo resolution.

## Workspace compatibility

The compiler supports the standard modifiers used by pyhf workspace schema
version `1.0.0`:

- `normfactor`
- `normsys`
- `histosys`
- `staterror`
- `shapesys`
- `lumi`

`model.approximation` reports the modifier and parameter coverage of the
compiled model. The nominal predictions and local response derivatives are
tested against pyhf 0.7.6; the likelihood and resulting inference are
deliberately approximate.

SGHF uses an object API:

| pyhf-style operation | SGHF |
| --- | --- |
| `pyhf.Workspace(spec)` | `sghf.load(spec)` |
| `pyhf.infer.mle.fit(...)` | `model.fit(...)` |
| `pyhf.infer.hypotest(...)` | `model.hypotest(...)` |
| `pyhf.infer.intervals.upper_limits.upper_limit(...)` | `model.upper_limit(...)` |

Accidental pyhf-style calls through `sghf` raise an error naming the
corresponding object method.

## Development

```console
python -m pip install -e ".[test]"
python -m pytest
python -m build
```

The benchmark script compares repeated SGHF and pyhf fit latency on a supplied
workspace without claiming a universal speedup. For small models, fix the BLAS
thread count so thread-pool overhead does not dominate the comparison:

```console
OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_fit.py workspace.json --repeats 20
```

Use the equivalent setting for a different BLAS implementation and report the
thread count with benchmark results.

## Credits and license

This package is a clean standalone rewrite of SGHF. Sasha Glazov and Cyrille
Praz are credited for the original SGHF implementation and concepts. See
[AUTHORS.md](AUTHORS.md) for details.

SGHF is distributed under the [MIT License](LICENSE).
