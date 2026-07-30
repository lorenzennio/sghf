[Documentation index](index.md) | Previous: [Simplified likelihood](simplified-likelihood.md) | Next: [Fitting](fitting.md)

# Workspace compilation

`sghf.load` translates a pyhf workspace JSON document into the arrays of the
[simplified likelihood](simplified-likelihood.md). Compilation is deterministic
and happens once. Subsequent fits and tests operate on the resulting `Model`;
they do not revisit the JSON modifier objects.

The external model and schema references are collected under
[HistFactory and pyhf](references.md#histfactory-and-pyhf). The formulas on
this page are the behavior of SGHF's compiler, not generic pyhf formulas.

## Loading a workspace

```python
import sghf

model = sghf.load("workspace.json")
```

The source can be:

- a mapping containing the decoded workspace;
- a path-like object or string;
- a text stream with a `.read()` method.

The public signature is

```python
sghf.load(source, *, measurement=None, strict=False)
```

The document must contain `channels`, `observations`, `measurements`,
`version`, and version must be `"1.0.0"`.

### Measurement selection

`measurement="name"` selects that measurement. Without an explicit name, SGHF
uses the first measurement. If several measurements are present, it emits
`WorkspaceWarning` and records the message in `Model.warnings`.

The selected measurement supplies:

- the parameter-of-interest name through `config.poi`;
- optional group-wide `inits`;
- optional per-scalar `bounds`;
- optional group-wide `fixed`;
- optional Gaussian `auxdata`;
- optional Gaussian `sigmas`, converted to precision $1/\sigma^2$.

Settings must match the scalar dimension of the parameter group. An initial or
fixed value outside the effective bounds is rejected.

### Unknown and unsupported content

Unknown top-level fields are ignored with `WorkspaceWarning`. With
`strict=True`, the same condition raises `InvalidWorkspace`. Strict mode does
not add a general schema validator for nested content.

An unrecognized modifier type raises `UnsupportedModifier`. The implemented
set is:

```text
normfactor, normsys, histosys, staterror, shapesys, lumi
```

## Deterministic layout

SGHF sorts channels alphabetically by channel name. Within each channel,
sample contributions are summed in their workspace order. Main observations,
nominal expectations, and response rows are flattened by sorted channel and
then by bin.

`Model.channel_names` gives this sorted order. `Model.channel_slices[name]`
maps a channel back to its contiguous section of a flattened array.

Parameter groups are sorted by modifier type and then name using this order:

1. `histosys`
2. `lumi`
3. `normfactor`
4. `normsys`
5. `shapesys`
6. `staterror`

A vector `staterror` or `shapesys` group contributes one scalar parameter per
channel bin, named `name[0]`, `name[1]`, and so on. Scalar groups retain their
modifier name.

`normsys` and `histosys` are allowed to share a parameter name because both
use a standard-normal coordinate. If the same name first appears under one of
these types and later under the other, its ordering category is the type of
the first occurrence after channel sorting. Other cross-type reuse is
rejected.

## General response construction

Consider one sample with raw nominal template $\mathbf s$. Let
$\mathcal H$ be its `histosys` modifiers and $\mathcal S$ its multiplicative
modifiers. At the configured initial values, SGHF constructs

$$
\mathbf t_0
=
\mathbf s
+
\sum_{h\in\mathcal H}
\boldsymbol{\delta}_h\theta_{0,h},
$$

$$
\mathbf f_0
=
\prod_{m\in\mathcal S}
\mathbf f_m(\boldsymbol{\theta}_{0,m}),
$$

and adds

$$
\boldsymbol{\nu}_{0,\mathrm{sample}}
=
\mathbf t_0\odot\mathbf f_0
$$

to the nominal expectation. Here $\odot$ is elementwise multiplication;
scalar factors are broadcast across sample bins.

For one scale parameter $\theta_m$, the response at the expansion point is

$$
\frac{\partial\boldsymbol{\nu}_{\mathrm{sample}}}
{\partial\theta_m}
=
\mathbf t_0
\odot
\frac{\partial\mathbf f_m}{\partial\theta_m}
\odot
\prod_{\substack{r\in\mathcal S\\r\ne m}}
\mathbf f_r(\boldsymbol{\theta}_{0,r}).
$$

For one `histosys` parameter,

$$
\frac{\partial\boldsymbol{\nu}_{\mathrm{sample}}}
{\partial\theta_h}
=
\boldsymbol{\delta}_h\odot\mathbf f_0.
$$

Contributions from all samples and channels sharing a scalar parameter are
added into the same response column.

These products define the derivative at the expansion point. Once compiled,
`Model.expected` is additive and does not recalculate products, so
multiplicative cross terms away from that point are absent.

## Modifier mappings

### `normfactor`

`normfactor` is an unconstrained scalar multiplicative normalization.
Its defaults are

$$
\theta_0=1,\qquad 0\le\theta\le10,\qquad P=0.
$$

For the sample contribution
$\mathbf t_0\theta\prod_{r\ne m}\mathbf f_r$, the nominal factor is
$\theta_0$ and

$$
\frac{\partial\boldsymbol{\nu}}{\partial\theta}
=
\mathbf t_0
\odot
\prod_{r\ne m}\mathbf f_r(\boldsymbol{\theta}_{0,r}).
$$

Measurement configuration can change its initial value, bounds, fixed state,
auxiliary mean, and precision. In ordinary use it remains unconstrained
unless a precision is explicitly configured.

The compiled response cannot reproduce products between a shifted
`normfactor` and another simultaneously shifted scale modifier.

### `lumi`

`lumi` uses the same local multiplicative response as `normfactor`. Its
defaults are

$$
\theta_0=1,\qquad 10^{-10}\le\theta\le10,\qquad P=0.
$$

A measurement `sigma` can supply a Gaussian luminosity constraint
$P=1/\sigma^2$. The compiler does not infer a luminosity uncertainty from the
modifier object itself.

Thus the local response is

$$
\frac{\partial\boldsymbol{\nu}}{\partial\theta_\mathrm{lumi}}
=
\mathbf t_0
\odot
\prod_{r\ne\mathrm{lumi}}
\mathbf f_r(\boldsymbol{\theta}_{0,r}).
$$

The exact nonlinear multiplicative dependence is retained only to first order
about the configured initial point.

### `histosys`

For low and high templates $\mathbf s_\mathrm{lo}$ and
$\mathbf s_\mathrm{hi}$, SGHF defines

$$
\boldsymbol{\delta}
=
\frac{\mathbf s_\mathrm{hi}-\mathbf s_\mathrm{lo}}{2}.
$$

The sample template at the expansion point is

$$
\mathbf t_0
=
\mathbf s+\boldsymbol{\delta}\theta_0
$$

including the analogous additive terms from any other `histosys` modifiers.
The response column receives

$$
\frac{\partial\boldsymbol{\nu}}{\partial\theta}
=
\boldsymbol{\delta}\odot\mathbf f_0.
$$

Defaults are

$$
\theta_0=0,\qquad -5\le\theta\le5,\qquad
a=0,\qquad P=1.
$$

This is a symmetric linear interpolation. It records neither the separate
low/high curvature nor the piecewise interpolation used by pyhf away from
zero. A strongly asymmetric pair of templates is therefore a direct warning
that the local response may not describe relevant parameter shifts.

### `normsys`

Let $l>0$ and $h>0$ be the low and high normalization factors. SGHF uses

$$
f(\theta)=1+d\theta
$$

where $d$ is the derivative at zero of pyhf's default code-4 interpolator.
The implementation computes

$$
\begin{aligned}
b_0&=h-1, & b_1&=l-1,\\
b_2&=h\log h, & b_3&=-l\log l,\\
b_4&=h(\log h)^2, & b_5&=l(\log l)^2,
\end{aligned}
$$

then

$$
d
=
\frac{15}{16}b_0
-\frac{15}{16}b_1
-\frac{7}{16}b_2
-\frac{7}{16}b_3
+\frac{1}{16}b_4
-\frac{1}{16}b_5.
$$

The nominal factor is $1+d\theta_0$; the response is

$$
\frac{\partial\boldsymbol{\nu}}{\partial\theta}
=
d\,\mathbf t_0
\odot
\prod_{r\ne m}\mathbf f_r(\boldsymbol{\theta}_{0,r}).
$$

Defaults are the same standard-normal coordinate as `histosys`:

$$
\theta_0=0,\qquad -5\le\theta\le5,\qquad
a=0,\qquad P=1.
$$

The common shortcut $(h-l)/2$ is not the SGHF formula. Even with the correct
central derivative, exponential tails and code-4 polynomial curvature are
not present after compilation.

### `staterror`

`staterror` contributes one multiplicative parameter per bin. Suppose all
occurrences sharing the name provide absolute uncertainties
$\sigma_{ji}$ and raw nominal affected yields $s_{ji}$ in bin $i$. SGHF
combines them as

$$
\sigma_i^\mathrm{combined}
=
\left(\sum_j\sigma_{ji}^2\right)^{1/2},
\qquad
s_i^\mathrm{affected}
=
\sum_j s_{ji}.
$$

For positive combined uncertainty, it sets the effective Gaussian precision

$$
P_{ii}
=
\left(
\frac{s_i^\mathrm{affected}}
{\sigma_i^\mathrm{combined}}
\right)^2.
$$

The parameter defaults to one with positive bounds. Its response is nonzero
only in its matching bin and equals the affected sample template times all
other initial scale factors. Multiple samples using the same name contribute
to the same binwise parameter, and shared names across channels require equal
bin counts.

If the combined uncertainty is zero, the corresponding parameter is fixed.
If the affected nominal sum is zero while a positive uncertainty is supplied,
the derived precision is zero and identifiability must come from another
response or the fit raises `NonIdentifiableModel`.

HistFactory's Poisson/gamma statistical constraint is represented here only
by its local Gaussian precision.

### `shapesys`

`shapesys` also contributes one positive multiplicative parameter per bin and
uses the same effective precision construction:

$$
P_{ii}
=
\left(
\frac{s_i}{\sigma_i}
\right)^2
$$

for positive absolute uncertainty $\sigma_i$. Its matching-bin response is
the affected template times the other initial scale factors. Zero-uncertainty
elements are fixed.

Unlike `staterror`, a `shapesys` name cannot be shared by another occurrence.
This preserves one independent binwise group for the sample that owns it.
The nonlinear gamma-type constraint and multiplicative response are replaced
by their local Gaussian/quadratic representation.

## Parameter defaults and overrides

Before measurement overrides, compiler defaults are:

| Group | Initial and auxiliary mean | Bounds | Precision |
|---|---:|---:|---:|
| `normfactor` | $1$ | $[0,10]$ | $0$ |
| `normsys`, `histosys` | $0$ | $[-5,5]$ | $1$ |
| `lumi`, `staterror`, `shapesys` | $1$ | $[10^{-10},10]$ | $0$ before modifier-specific or measurement configuration |

Measurement settings are applied by parameter-group name. For vector
statistical modifiers, positive modifier uncertainties determine the final
effective precision during compilation; zero-uncertainty elements are fixed.

`Model.initial` is both the default starting vector and the expansion point of
the response. Changing a parameter through a later fit call does not
recompile $A$.

## Observations and model contents

Every sorted channel must have exactly one observation entry with the same
number of bins as its samples. Observations must be finite. Sample nominal
yields must be finite and nonnegative.

The returned `Model` exposes:

| Field | Meaning |
|---|---|
| `channel_names`, `channel_slices` | deterministic flattened channel layout |
| `parameter_names`, `poi_name` | scalar parameter layout and POI |
| `nominal_expected`, `response` | $\boldsymbol{\nu}_0$ and $A$ |
| `observations` | flattened $\mathbf n$ |
| `initial` | $\boldsymbol{\theta}_0$ |
| `lower_bounds`, `upper_bounds`, `fixed` | parameter domain |
| `constraint_mean`, `constraint_precision` | $\mathbf a$ and diagonal of $P$ |
| `modifier_types` | sorted set of modifier types encountered |
| `warnings` | accepted compilation warnings |

`model.approximation` returns an `ApproximationReport` with
`modifier_types`, `parameter_names`, `variance_default="iterated"`, and
`warnings`. It is a description of the compiled approximation, not a
numerical accuracy certificate.

## Failure modes

Compilation raises:

- `InvalidWorkspace` for missing structure, inconsistent bin counts or
  parameter dimensions, invalid values, invalid bounds, or incompatible
  modifier-name reuse;
- `UnsupportedModifier` for a modifier outside the implemented set;
- `WorkspaceWarning` for an accepted extra top-level field or implicit choice
  among several measurements.

Loading can succeed even when a later fit is non-identifiable or a trial point
leaves the positive-yield domain. Those properties depend on the assembled
response, constraints, fixed parameters, and requested fit. They are checked
by the fitting and evaluation paths.

---

[Documentation index](index.md) | Previous: [Simplified likelihood](simplified-likelihood.md) | Next: [Fitting](fitting.md)
