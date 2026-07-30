[Documentation index](index.md) | Previous: [API reference](api-reference.md) | Next: [References](references.md)

# Validation and applicability

SGHF is useful when the local linear response and quadratic statistic preserve
the inference features that matter for an analysis. That condition must be
checked; successful loading and fitting do not establish it.

This page gives a layered decision procedure. The checks are deliberately tied
to the actual SGHF arrays and result objects. No universal numerical threshold
is prescribed because acceptable error depends on the analysis estimand,
tail probability, and required precision.

## Minimum decision checklist

Before using an SGHF result:

1. Verify nominal expectations and local response columns against pyhf.
2. Scan the response beyond an infinitesimal neighborhood in every parameter
   that materially affects the result.
3. Inspect expected counts, fitted-yield positivity, nuisance pulls, and active
   bounds.
4. Check identifiability and covariance conditioning.
5. Compare nominal and iterated variance results.
6. Use sufficient toys when empirical tail calibration is required.
7. Compare the final relevant test or limit with pyhf when diagnostics are
   unfavorable or the conclusion is tail-, boundary-, or nonlinearity-sensitive.

The rest of this page defines what each check can establish.

## 1. Nominal compilation parity

At the expansion point, check

$$
\boldsymbol{\nu}_\mathrm{SGHF}(\boldsymbol{\theta}_0)
=
\boldsymbol{\nu}_0
$$

against the pyhf expected main data at the same parameter values.

Useful summaries include a binwise relative difference

$$
r_i
=
\frac{
\nu_{0,i}^\mathrm{SGHF}-\nu_{0,i}^\mathrm{pyhf}
}{
\max(|\nu_{0,i}^\mathrm{pyhf}|,\epsilon)
},
$$

where $\epsilon$ is an analysis-chosen numerical floor, and an absolute
difference in bins near zero.

Agreement validates sample summation, ordering, configured initial factors,
and the nominal part of the compiler. It does not validate response curvature
or constraint distributions.

## 2. Local derivative parity

For scalar parameter $j$, compare the SGHF response column $A_{\cdot j}$ with
central finite differences of the pyhf expectation:

$$
D_{\cdot j}(h)
=
\frac{
\boldsymbol{\nu}_\mathrm{pyhf}
(\boldsymbol{\theta}_0+h\mathbf e_j)
-
\boldsymbol{\nu}_\mathrm{pyhf}
(\boldsymbol{\theta}_0-h\mathbf e_j)
}{
2h
}.
$$

Use several decreasing $h$ values that remain numerically resolvable and
inside allowed parameter ranges. Stable convergence
$D_{\cdot j}(h)\to A_{\cdot j}$ supports the compiler's local derivative.

This check is especially important for:

- asymmetric `normsys` and `histosys`;
- shared modifiers across samples or channels;
- vector `staterror` and `shapesys`;
- models with non-default initial values;
- samples with several simultaneous multiplicative modifiers.

Derivative parity validates first-order response only.

## 3. Response curvature over the fitted range

Identify parameter values reached by:

- the free fit;
- each tested conditional fit;
- conditional toy-generation truths;
- upper-limit scans.

At representative displacements $\delta$, compare

$$
\boldsymbol{\nu}_\mathrm{linear}
(\boldsymbol{\theta}_0+\boldsymbol{\delta})
=
\boldsymbol{\nu}_0+A\boldsymbol{\delta}
$$

with pyhf's nonlinear expectation. Evaluate binwise absolute differences,
relative differences where rates are not near zero, and changes relative to
the statistical scale relevant to the fit.

A useful one-parameter curvature diagnostic is

$$
C_{ij}(h)
=
\nu_i(\boldsymbol{\theta}_0+h\mathbf e_j)
+
\nu_i(\boldsymbol{\theta}_0-h\mathbf e_j)
-
2\nu_i(\boldsymbol{\theta}_0).
$$

SGHF sets this second difference to zero. A pyhf value that is material
relative to the analysis tolerance indicates discarded curvature.

Joint scans are required when multiple scale parameters can move together:
one-dimensional derivative agreement does not test the multiplicative cross
terms absent from SGHF.

## 4. Modifier asymmetry

For an up/down template modifier, inspect

$$
\boldsymbol{\Delta}_+
=
\boldsymbol{\nu}(+1)-\boldsymbol{\nu}(0),
\qquad
\boldsymbol{\Delta}_-
=
\boldsymbol{\nu}(0)-\boldsymbol{\nu}(-1).
$$

Large differences in magnitude or shape between
$\boldsymbol{\Delta}_+$ and $\boldsymbol{\Delta}_-$ cannot be represented by
one fixed central response column. The same issue arises for unequal
`normsys` factors even though SGHF uses the correct code-4 derivative at zero.

Escalate to pyhf when an influential parameter is strongly asymmetric and the
fit or tested point moves appreciably away from zero.

## 5. Expected counts and the quadratic objective

Inspect nominal and fitted expectations by bin. The Gaussian
main-measurement objective is most questionable when:

- expected counts are small;
- observations are zero or far from their expectation;
- a discovery or exclusion result depends on a sparse tail;
- one or two bins dominate the statistic;
- conditional fits move rates substantially.

There is no count threshold that guarantees validity for every test. Compare
the relevant objective differences with the exact pyhf likelihood and use
Poisson toys when the sampling distribution of the SGHF statistic itself must
be calibrated.

Poisson toy generation addresses discreteness around the compiled mean. It
does not restore the exact likelihood-ratio ordering or nonlinear response.

## 6. Pulls, bounds, and positivity

For every returned `FitResult`, inspect:

```python
fit.pulls
fit.at_bound
fit.expected
```

Large absolute pulls indicate movement away from the auxiliary mean. Also
inspect `fit.values - model.initial`, which is the displacement from the
response expansion point; the two displacements are identical only when the
configured auxiliary mean equals the initial value. An active bound changes
the local covariance convention and can invalidate regular asymptotic
arguments. A zero covariance row at a bound is a conditional convention, not
evidence of zero uncertainty.

All expected yields must remain strictly positive. A `ModelDomainError` is a
hard indication that the linear response has left its evaluable domain.
Values barely above zero are also fragile because small additional
displacements can cross the boundary.

Inspect free, conditional, Asimov, background, and discovery-alternative fits;
checking only the observed free fit is insufficient.

## 7. Identifiability and conditioning

`NonIdentifiableModel` detects numerical rank deficiency. A full-rank model
can still be poorly conditioned.

For nominal variance and free indices $F$, the information matrix is

$$
I_F
=
A_F^\mathsf T
\operatorname{diag}(\boldsymbol{\nu}_0)^{-1}
A_F
+
P_F.
$$

Inspect its eigenvalue range or condition number at the numerical scale used
by the analysis. Very strong correlations, unstable errors under small
perturbations, or sensitivity to floating-point precision indicate weak
identification.

At active bounds, the returned covariance is based on the inactive information
submatrix. It should not be compared mechanically with an unconstrained pyhf
Hessian without matching the boundary convention.

## 8. Nominal-versus-iterated stability

Compare at least:

```python
fit_nominal = model.fit(variance="nominal")
fit_iterated = model.fit(variance="iterated")
```

For the parameter of interest, nuisance pulls, fitted expectations, and
relevant statistics, assess the shift relative to the precision required for
the analysis. One possible scale-free diagnostic is

$$
S_\mu
=
\frac{
|\hat\mu_\mathrm{nominal}-\hat\mu_\mathrm{iterated}|
}{
\max(
\sigma_{\mu,\mathrm{nominal}},
\sigma_{\mu,\mathrm{iterated}},
\epsilon_\mu
)
},
$$

where $\epsilon_\mu$ is an analysis-chosen floor.

A material difference indicates sensitivity to the choice of diagonal
Gaussian weights. Agreement does not validate the original Poisson likelihood;
both paths use the same linear response and quadratic residual form.

For iterated hypothesis tests, remember that free and conditional fits can use
different final weights. Compare test statistics and limits directly, not only
best-fit parameters.

## 9. Toy resolution and Monte Carlo stability

The corrected empirical tail has minimum

$$
p_\mathrm{min}
=
\frac{1}{N_\mathrm{toys}+1}.
$$

Choose `ntoys` so this resolution is well below the tail probability that
would affect the conclusion. Also examine repeat-seed variation, especially
for:

- small $p_0$;
- ratios with small $CL_b$;
- expected bands built from the same finite ensemble;
- comparisons close to a decision threshold.

SGHF returns statistic arrays so users can inspect empirical distributions,
ties, extreme counts, and stability. It does not return Monte Carlo confidence
intervals.

## 10. End-to-end pyhf comparison

For final scientific use, compare pyhf and SGHF at the level that enters the
conclusion:

- free and conditional POI estimates;
- relevant objective or likelihood-ratio differences;
- observed and expected $CL_s$;
- discovery $p_0$ or significance;
- upper-limit crossings;
- toy distributions when empirical calibration is central.

Match the data, auxiliary observations, bounds, modifier configuration, and
test-statistic convention. A numerical difference without matched conventions
does not isolate the SGHF approximation.

## What local parity does not validate

Exact agreement of $\boldsymbol{\nu}_0$ and $A$ does **not** establish:

- nonlinear expected-yield agreement away from
  $\boldsymbol{\theta}_0$;
- equality of Poisson/gamma and Gaussian auxiliary constraints;
- equality of SGHF's quadratic objective and pyhf's likelihood ratio;
- regularity at parameter boundaries;
- asymptotic coverage;
- empirical coverage for the original workspace;
- upper-limit agreement over a scan far from the expansion point.

These are separate falsifiable checks.

## Escalation rules

| Observation | Minimum response |
|---|---|
| nominal or derivative mismatch | do not use the compiled result until the workspace translation is understood |
| material curvature or cross-term discrepancy over fitted/tested points | use pyhf for the affected result or restrict SGHF to a demonstrated local task |
| influential asymmetric modifier with large displacement | compare the complete test or limit with pyhf |
| sparse or far-tail result | use toy calibration and compare statistic ordering with pyhf |
| active POI or nuisance bound | inspect boundary conventions and prefer toy or pyhf validation |
| `ModelDomainError` or near-zero fitted yield | treat the linear model as outside its domain |
| rank failure or unstable covariance | reparameterize, fix the degeneracy, or use the original model |
| material nominal-versus-iterated shift | report the sensitivity and validate against pyhf |
| empirical result near the finite-toy resolution | increase `ntoys` and assess repeat-seed stability |

Final scientific results should be checked with pyhf whenever approximation
diagnostics are unfavorable or the result is sensitive to tails, boundaries,
or nonlinear modifier behavior.

---

[Documentation index](index.md) | Previous: [API reference](api-reference.md) | Next: [References](references.md)
