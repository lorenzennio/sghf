# SGHF standalone package design

SGHF compiles a pyhf workspace JSON document into a local linear-Gaussian
model. The compiled model stores nominal bin expectations, their first-order
responses to every parameter, Gaussian constraints, bounds, fixed states,
channel slices, observations, and the selected parameter of interest.

The public API is object-oriented:

```python
model = sghf.load("workspace.json")
model.fit()
model.hypotest(1.0)
model.upper_limit()
model.toys(10_000, rng=17)
```

The runtime depends only on NumPy and SciPy. pyhf is used only in development
tests. Unknown top-level workspace metadata is accepted with a warning by
default and rejected in strict mode. All standard modifiers are supported:
`normfactor`, `normsys`, `histosys`, `staterror`, `shapesys`, and `lumi`.

Fits minimize a constrained quadratic objective. Poisson variances are
iterated from model expectations by default, with a frozen nominal option.
Asymptotic tests use profiled quadratic statistics and Asimov data. Toy
inference fluctuates Poisson main measurements and Gaussian auxiliary
measurements. Invalid domains and non-identifiable models fail explicitly.

The implementation is a clean rewrite under the MIT license. Sasha Glazov and
Cyrille Praz are credited for the original SGHF implementation and concepts.
