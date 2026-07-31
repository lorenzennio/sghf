# SGHF worked recipes

Complete, runnable recipes. Substitute your own workspace path.

## Compile and inspect before trusting anything

```python
import sghf

model = sghf.load("workspace.json")

print(model.poi_name, len(model.parameter_names), "parameters")
print(model.channel_names)
print(model.approximation)      # modifier types, parameter names, warnings
```

`approximation.warnings` is non-empty when the loader accepted something
nonstandard — read it. An `UnsupportedModifier` here means SGHF cannot compile
this workspace at all and you need pyhf.

For a workspace with several measurements, name the one you want, otherwise the
first is taken and a `WorkspaceWarning` is emitted:

```python
model = sghf.load("workspace.json", measurement="Measurement")
```

## Fit and report a signal strength

```python
fit = model.fit()

print(f"{model.poi_name} = {fit.value(model.poi_name):.4f}"
      f" +/- {fit.error(model.poi_name):.4f}")
print("converged:", fit.converged, "iterations:", fit.iterations)
print("at bound:", fit.at_bound)
```

**Check `fit.at_bound` before quoting the uncertainty.** A parameter listed
there has zero covariance and its error reads `0.0`, which is a flag rather
than a measurement.

The reported error is the Wald error from the covariance diagonal, not a
profile-likelihood interval.

Most correlated nuisance parameters:

```python
import numpy as np

poi = model.parameter_index(model.poi_name)
correlation = fit.correlation[poi].copy()
correlation[poi] = 0.0
for index in np.argsort(-np.abs(correlation))[:5]:
    print(f"{model.parameter_names[index]:30s} {correlation[index]:+.3f}")
```

Fitting supplied data, or with a parameter held fixed:

```python
# Substitute your own counts; the shape must match the compiled channels.
counts = {
    name: model.observations[model.channel_slices[name]]
    for name in model.channel_names
}

fit = model.fit(
    data=counts,
    fixed={model.poi_name: 1.0},
    bounds={model.poi_name: (0.0, 5.0)},
)
```

A `data` mapping must carry **every** channel in `model.channel_names` and
nothing else, or you get `ValueError: Channel data mismatch` naming what is
missing or extra. A flat sequence in `model.channel_slices` order works too.
Names in `fixed` and `bounds` must be in `model.parameter_names`; unknown names
raise `KeyError`.

## Discovery significance

```python
result = model.hypotest(0.0, test_stat="q0")
print(f"q0 = {result.q:.3f}   Z = {result.significance:.3f} sigma")
print(f"p0 = {result.clsb:.4g}")     # asymptotic q0 has no .p0 field
```

`q0` requires `poi_test=0`; anything else raises `ValueError`.

The asymptotic result exposes the discovery p-value as **`clsb`**. Its `cls`
field is also populated but is an exclusion quantity and is wrong by orders of
magnitude if read as a p-value — see [pitfalls.md](pitfalls.md).

To calibrate that against pseudo-experiments rather than the asymptotic formula:

```python
toy = model.hypotest(0.0, test_stat="q0", calculator="toys",
                     ntoys=200_000, rng=1234)
print(toy.p0, toy.significance)
print("resolution floor:", 1.0 / (toy.ntoys + 1))
```

**Size `ntoys` from the p-value you need.** If `toy.p0` equals
`1/(ntoys+1)` the ensemble never reached the observed statistic and the p-value
is a lower bound, not a measurement — a significance quoted from it is wrong.
Resolving 3 sigma needs of order 1e4 toys; 5 sigma is out of reach by direct
counting.

The toy calculator generates its own ensembles at the conditional maximum
likelihood fit internally. You do not need to build them yourself.

## Exclusion at a fixed signal strength

```python
result = model.hypotest(1.0, test_stat="qtilde")
print(f"CLs={result.cls:.5f}  CLs+b={result.clsb:.5f}  CLb={result.clb:.5f}")
```

`qtilde` requires the POI lower bound to be exactly zero; use `q` if it is not.

`CLs` of exactly `1.0` for any mu below the best fit is correct, not a failure
— see [pitfalls.md](pitfalls.md).

## Upper limit with the expected band

```python
limit = model.upper_limit(level=0.05)

labels = ("-2 sigma", "-1 sigma", "median", "+1 sigma", "+2 sigma")
print(f"observed  {limit.observed:.4f}")
for label, value in zip(labels, limit.expected):
    print(f"{label:9s} {value:.4f}")
```

The band runs `-2 sigma` first. Label the entries explicitly; the toy discovery
`expected_significance` tuple runs the opposite way.

Toy-based limit, when the asymptotic assumptions are in doubt:

```python
limit = model.upper_limit(calculator="toys", ntoys=20_000, rng=1234)
print(limit.observed, limit.scan_values[0], limit.scan_values[-1])
```

This scans a grid and interpolates rather than root-finding, because the toy
CLs curve is stochastic. Its accuracy is bounded by `ntoys` and `points`,
neither of which is reported in the result.

If the automatic range misses the crossing you get `LimitNotFound`; pass an
explicit `bracket=(low, high)` or `scan=`.

## Generate pseudo-experiments directly

For calibration studies where you want the pseudo-data itself:

```python
background = model.fit(fixed={model.poi_name: 0.0})
toys = model.toys(1000, parameters=background, sampling="poisson", rng=17)

print(toys.observations.shape)   # (1000, nbins)
print(toys.auxiliary.shape)      # (1000, nparameters)
```

Passing an integer `rng` makes the ensemble reproducible. Constrained parameters
get independently fluctuated Gaussian auxiliary measurements; pass
`fluctuate_auxiliary=False` to hold them fixed. `sampling="gaussian"` can yield
negative counts, which the quadratic fitter accepts without clipping.

For an ensemble large enough to strain memory, `batch_size` bounds the temporary
arrays; it does not change the result.

## Plot a CLs curve

```python
import numpy as np
import matplotlib.pyplot as plt

limit = model.upper_limit(keep_results=True)

poi = np.array([test.poi_test for test in limit.results])
observed = np.array([test.cls for test in limit.results])
band = np.array([test.expected_cls for test in limit.results])

fig, ax = plt.subplots()
ax.fill_between(poi, band[:, 0], band[:, 4], color="gold", label=r"$\pm2\sigma$")
ax.fill_between(poi, band[:, 1], band[:, 3], color="limegreen", label=r"$\pm1\sigma$")
ax.plot(poi, band[:, 2], "k--", label="expected")
ax.plot(poi, observed, "k-", label="observed")
ax.axhline(limit.level, color="red", ls=":")
ax.set_yscale("log")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$\mathrm{CL}_s$")
ax.legend()
```

`keep_results=True` retains every fit — of order 200 MiB on a large model, which
is why it is off by default. For `calculator="asymptotic"` the points are
wherever the root finder looked: dense at the crossings, sparse between. Do not
read the gaps as structure. The toy calculator returns an even grid instead.
