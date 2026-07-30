[Documentation index](index.md) | Previous: [Validation and applicability](validation-and-applicability.md) | Next: [Documentation index](index.md)

# References

The external sources below define HistFactory, pyhf, profile-likelihood
asymptotics, $CL_s$, and Monte Carlo tail calculations. They motivate the
statistical constructions, but they do not by themselves establish that SGHF
implements an exact likelihood or has exact coverage. Exact SGHF behavior is
defined by the local source files listed at the end of this page.

## HistFactory and pyhf

K. Cranmer, G. Lewis, L. Moneta, A. Shibata, and W. Verkerke,
*HistFactory: A tool for creating statistical models for use with RooFit and
RooStats*, CERN-OPEN-2012-016 (2012).
[CERN record and report](https://cds.cern.ch/record/1456844),
[DOI: 10.17181/CERN-OPEN-2012-016](https://doi.org/10.17181/CERN-OPEN-2012-016).

This is the primary model reference for channels, samples, modifiers,
parameters, main measurements, and auxiliary constraints. SGHF implements
only the local approximation documented in
[the simplified likelihood](simplified-likelihood.md).

L. Heinrich, M. Feickert, G. Stark, and K. Cranmer,
*pyhf: pure-Python implementation of HistFactory statistical models*,
Journal of Open Source Software **6** (58), 2823 (2021).
[JOSS article](https://joss.theoj.org/papers/10.21105/joss.02823),
[DOI: 10.21105/joss.02823](https://doi.org/10.21105/joss.02823).

The official pyhf documentation provides the
[workspace likelihood specification](https://scikit-hep.org/pyhf/likelihood.html),
[schema version 1.0.0](https://scikit-hep.org/pyhf/schemas/1.0.0/workspace.json),
and the
[code-4 interpolation definition](https://scikit-hep.org/pyhf/_generated/pyhf.interpolators.code4.html).
These are the comparison targets for SGHF workspace compilation. pyhf remains
the reference implementation used for final cross-checks.

## Profile likelihood and asymptotic tests

G. Cowan, K. Cranmer, E. Gross, and O. Vitells,
*Asymptotic formulae for likelihood-based tests of new physics*,
European Physical Journal C **71**, 1554 (2011).
[arXiv:1007.1727](https://arxiv.org/abs/1007.1727),
[DOI: 10.1140/epjc/s10052-011-1554-0](https://doi.org/10.1140/epjc/s10052-011-1554-0).

This paper derives one-sided profile-likelihood statistics, their Wald
approximations, boundary behavior, and the Asimov data set. SGHF applies
related transformations to its compiled quadratic objective. In iterated
variance mode, its free and conditional objective values can use different
weights, so the paper's likelihood-ratio derivation is not an identity for the
implemented statistic.

## Modified-frequentist `CLs`

A. L. Read, *Presentation of search results: the CLs technique*,
Journal of Physics G: Nuclear and Particle Physics **28**, 2693–2704 (2002).
[DOI: 10.1088/0954-3899/28/10/313](https://doi.org/10.1088/0954-3899/28/10/313).

This is the primary reference for the modified-frequentist ratio used for
exclusion. SGHF's exact definitions of `clsb`, `clb`, and `cls` are given in
[hypothesis testing](hypothesis-testing.md) and
[toy inference](toy-inference.md).

## Monte Carlo tail probabilities

B. Phipson and G. K. Smyth,
*Permutation P-values should never be zero: calculating exact P-values when
permutations are randomly drawn*, Statistical Applications in Genetics and
Molecular Biology **9**, Article 39 (2010).
[PubMed record](https://pubmed.ncbi.nlm.nih.gov/21044043/),
[DOI: 10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585).

The paper treats random permutations and also discusses the more general
problem of assigning $p$-values with Monte Carlo simulation. SGHF uses the
finite-ensemble upper-tail rule

$$
\hat p=\frac{k+1}{N+1},
$$

which prevents a reported zero and gives minimum resolution $1/(N+1)$. SGHF
uses independent pseudo-experiments from a fitted parametric model rather than
permutations, so the cited permutation-test exactness claims should not be
transferred without their assumptions.

## SGHF implementation sources

These local files are normative for implementation-specific equations and
defaults:

- [workspace compiler](../src/sghf/workspace.py): modifier slopes,
  parameter ordering, and effective constraints;
- [fitting engine](../src/sghf/fit.py): weighted solves, variance iteration,
  bounds, covariance, and pulls;
- [asymptotic inference](../src/sghf/asymptotic.py): statistics, conditional
  Asimov construction, probability transform, and limit roots;
- [toy inference](../src/sghf/toys.py): pseudo-data generation, affine
  estimator, empirical tails, discovery, exclusion, and batching;
- [public model API](../src/sghf/model.py) and
  [package exports](../src/sghf/__init__.py): signatures and dispatch
  defaults.

Repository tests provide direct behavioral evidence:

- [loading tests](../tests/test_loading.py);
- [pyhf parity tests](../tests/test_pyhf_parity.py);
- [fit tests](../tests/test_fit.py);
- [asymptotic tests](../tests/test_asymptotic.py);
- [toy tests](../tests/test_toys.py);
- [public API tests](../tests/test_public_api.py).

---

[Documentation index](index.md) | Previous: [Validation and applicability](validation-and-applicability.md) | Next: [Documentation index](index.md)
