# SGHF Theory-First Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a layered, theory-first documentation set that derives the
linear-Gaussian approximation implemented by SGHF and documents every public
fit, hypothesis-test, toy, and limit interface.

**Architecture:** Keep each statistical concern on a focused Markdown page and
use one notation consistently across pages. Derive equations from the current
compiler and inference code, then add a standard-library test that checks the
required page set and all relative Markdown links. Keep the README concise and
use it as the package-level entry point into the detailed pages.

**Tech Stack:** GitHub-flavored Markdown with display math, Python 3.10+
standard library for documentation checks, pytest, Hatchling

## Global Constraints

- Do not change package behavior or public APIs.
- Do not introduce a documentation framework, runtime dependency, or test
  dependency.
- Use $\mathbf{n}$, $\boldsymbol{\nu}_0$, $A$,
  $\boldsymbol{\theta}$, $\mathbf{a}$, $P$, $V$, $\mu$,
  $\hat{\boldsymbol{\theta}}$, and
  $\hat{\hat{\boldsymbol{\theta}}}(\mu)$ consistently.
- State implementation behavior and standard statistical conventions
  separately wherever they differ.
- Do not claim exact pyhf parity or exact frequentist coverage for the original
  nonlinear HistFactory likelihood.
- Every conceptual page must contain an operational summary, mathematical
  derivation, and implementation/validity notes.
- Every page must link to the index and to the logically previous and next
  pages.
- Use only claims supported by the repository implementation or the primary
  sources collected in `docs/references.md`.
- End each independently reviewable documentation unit with a logical commit.

## File Structure

- Create `docs/index.md`: landing page, five-minute workflow, fixed notation,
  and complete navigation.
- Create `docs/simplified-likelihood.md`: derivation of the compiled response
  model and quadratic objective, with approximation conditions.
- Create `docs/workspace-compilation.md`: exact workspace-to-array rules and
  supported-modifier response formulas.
- Create `docs/fitting.md`: weighted least squares, iterated variances,
  covariance, pulls, bounds, and failures.
- Create `docs/hypothesis-testing.md`: profile statistics, Asimov construction,
  asymptotic transformations, $p$-values, and expected bands.
- Create `docs/toy-inference.md`: pseudo-data generation, empirical tails,
  vectorized nominal statistics, discovery, exclusion, batching, and
  reproducibility.
- Create `docs/upper-limits.md`: asymptotic $CL_s$ crossings, bracketing,
  caching, expected limits, and failures.
- Create `docs/api-reference.md`: signatures, defaults, return fields,
  exceptions, and pyhf-style guidance for all public objects.
- Create `docs/validation-and-applicability.md`: quantitative diagnostics and
  escalation criteria for pyhf comparison.
- Create `docs/references.md`: primary HistFactory, pyhf, profile-likelihood,
  $CL_s$, and Monte Carlo calibration sources.
- Modify `README.md`: add prominent links into the layered documentation
  without replacing the existing quick start.
- Create `tests/test_documentation.py`: required-page and relative-link checks
  using only `pathlib` and `re`.

---

### Task 1: Simplified Likelihood and Workspace Compilation

**Files:**
- Create: `docs/simplified-likelihood.md`
- Create: `docs/workspace-compilation.md`

**Interfaces:**
- Consumes: `sghf.workspace.load(source, *, measurement=None, strict=False)`,
  `Model.nominal_expected`, `Model.response`, `Model.initial`,
  `Model.constraint_mean`, and `Model.constraint_precision`.
- Produces: the fixed notation and modifier formulas used by all later theory
  pages.

- [ ] **Step 1: Write the simplified-likelihood operational layer**

Define SGHF as a local approximation to a binned HistFactory model. State that
workspace compilation produces

$$
\boldsymbol{\nu}(\boldsymbol{\theta})
=
\boldsymbol{\nu}_0+
A(\boldsymbol{\theta}-\boldsymbol{\theta}_0)
$$

and that `Model.expected` rejects non-finite or non-positive components. Give a
short decision table for nominal versus iterated diagonal variance.

- [ ] **Step 2: Derive the quadratic objective and normal quantities**

Define every symbol and write

$$
\chi^2(\boldsymbol{\theta})
=
(\mathbf n-\boldsymbol{\nu}(\boldsymbol{\theta}))^\mathsf{T}
V^{-1}
(\mathbf n-\boldsymbol{\nu}(\boldsymbol{\theta}))
+
(\boldsymbol{\theta}-\mathbf a)^\mathsf{T}
P(\boldsymbol{\theta}-\mathbf a).
$$

State explicitly that $V=\operatorname{diag}(\boldsymbol{\nu}_0)$ in nominal
mode, while iterated mode replaces the diagonal with the most recent positive
fitted expectation. Explain that zero precision means no auxiliary penalty.

- [ ] **Step 3: Document each approximation and its validity condition**

Cover, separately: first-order parameter response, diagonal Gaussian
main-measurement objective, Gaussian auxiliary constraints, fixed response
matrix, local asymmetric-modifier slopes, bounded linear solutions,
positive-yield domain, and omitted nonlinear response away from the expansion
point. For each item give a failure indicator such as large pulls, active
bounds, strong modifier asymmetry, low expected counts, negative trial
expectations, or nominal/iterated instability.

- [ ] **Step 4: Document deterministic workspace layout**

In `docs/workspace-compilation.md`, specify alphabetically sorted channels,
flattening by channel then bin, and parameter-group ordering
`histosys`, `lumi`, `normfactor`, `normsys`, `shapesys`, `staterror`, with names
sorted inside each type. Document first-measurement selection, explicit
`measurement=`, strict unknown-top-level behavior, workspace version `1.0.0`,
and rejection of unsupported modifiers.

- [ ] **Step 5: Derive the exact compiler response for every modifier**

For a sample template $\mathbf s$ and all other scale factors evaluated at the
configured initial point, document:

- `normfactor` and `lumi`: initial value one by default and derivative equal to
  the sample template times the other initial scale factors;
- `histosys`: central shift
  $\boldsymbol{\delta}=(\mathbf s_\mathrm{hi}-\mathbf s_\mathrm{lo})/2$,
  nominal contribution
  $\mathbf s+\theta_0\boldsymbol{\delta}$, and response
  $\boldsymbol{\delta}$ times the initial scale;
- `normsys`: factor $1+d\theta$, where $d$ is the derivative at zero of pyhf's
  code-4 interpolator computed by `_normsys_code4_slope`, not the simpler
  $(h-l)/2$ formula;
- `staterror` and `shapesys`: one parameter per channel bin, multiplicative
  response in the matching bin, combined absolute uncertainty
  $\sigma_i=(\sum_j\sigma_{ji}^2)^{1/2}$, and effective Gaussian precision
  $P_{ii}=(\sum_j s_{ji}/\sigma_i)^2$ for positive uncertainty;
- zero-uncertainty vector parameters are fixed, `shapesys` names cannot be
  shared, while `staterror` contributions with a shared name combine in
  quadrature.

Distinguish configured `inits`, `bounds`, `fixed`, `auxdata`, and `sigmas` from
the modifier defaults. Explain the cross-factor derivative at the expansion
point and the loss of multiplicative curvature away from it.

- [ ] **Step 6: Review formulas directly against compiler code**

Run:

```bash
rg -n "_normsys_code4_slope|template \\+=|scale \\*=|response\\[|uncertainty_sq|constraint_precision" src/sghf/workspace.py docs/simplified-likelihood.md docs/workspace-compilation.md
```

Expected: each compiler construction has a corresponding documented equation
or implementation note, with no unsupported modifier listed.

- [ ] **Step 7: Commit the theory foundation**

```bash
git add docs/simplified-likelihood.md docs/workspace-compilation.md
git commit -m "docs: derive the compiled likelihood"
```

---

### Task 2: Fitting

**Files:**
- Create: `docs/fitting.md`

**Interfaces:**
- Consumes: the compiled model and notation from Task 1;
  `Model.fit(data=None, *, auxdata=None, fixed=None, bounds=None,
  variance="iterated", rtol=1e-6, max_iter=25) -> FitResult`.
- Produces: definitions of free and conditional fits, covariance, and
  diagnostics used by Tasks 3 and 4.

- [ ] **Step 1: Derive the augmented weighted solve**

For free parameters $F$, write

$$
B=
\begin{bmatrix}
V^{-1/2}A_F\\
P_F^{1/2}
\end{bmatrix},
\qquad
\mathbf y=
\begin{bmatrix}
V^{-1/2}\mathbf r\\
P_F^{1/2}(\mathbf a_F-\boldsymbol{\theta}_{0,F})
\end{bmatrix},
$$

where $\mathbf r$ subtracts nominal expectation and any fixed-parameter
response. Then derive
$\Delta\hat{\boldsymbol{\theta}}_F=(B^\mathsf TB)^{-1}B^\mathsf T\mathbf y$.
State that rank deficiency of $B^\mathsf TB$ raises
`NonIdentifiableModel`.

- [ ] **Step 2: Document bounds and covariance semantics**

Explain that the unconstrained normal solve is used when it lies inside bounds
and SciPy constrained least squares is used otherwise. Define the reported
inactive-parameter covariance as the inverse inactive block of the information
matrix; fixed and active-bound rows/columns are zero. Derive errors,
correlations, constrained pulls, and explain `NaN` pulls for unconstrained
parameters.

- [ ] **Step 3: Document nominal and iterated variance paths**

State that nominal mode performs one solve. For iterated mode, reproduce the
implemented relative-change criterion

$$
\max_i\frac{|\nu_i^{(k)}-V_{ii}^{(k)}|}
{\max(|V_{ii}^{(k)}|,1)}\leq \texttt{rtol},
$$

the defaults `rtol=1e-6`, `max_iter=25`, and the 50:50 damping activated when
an update is closer to the previous variance vector than to the current one.
Document `FitConvergenceError` and `ModelDomainError`.

- [ ] **Step 4: Document every `FitResult` field and helper**

Cover `parameter_names`, `values`, `errors`, `covariance`, `correlation`,
`pulls`, `expected`, `chi2`, `converged`, `iterations`, `at_bound`,
`parameter_index`, `value`, `error`, and `__getitem__`. State that `chi2` uses
the final variance vector.

- [ ] **Step 5: Audit and commit**

Run:

```bash
rg -n "rtol|max_iter|damp_updates|active_local|pulls|chi2" src/sghf/fit.py docs/fitting.md
git add docs/fitting.md
git commit -m "docs: explain fitting and covariance"
```

Expected: the page states the exact tolerance, iteration limit, damping rule,
active-bound covariance convention, and result fields.

---

### Task 3: Asymptotic Hypothesis Tests and Upper Limits

**Files:**
- Create: `docs/hypothesis-testing.md`
- Create: `docs/upper-limits.md`

**Interfaces:**
- Consumes: `Model.hypotest`, `_profile_statistic`, `HypothesisTestResult`,
  `Model.upper_limit`, and `LimitResult`.
- Produces: test-statistic and $CL_s$ definitions reused by toy inference and
  validation guidance.

- [ ] **Step 1: Define profile differences and clipping rules**

Define
$\Delta\chi^2_\mu=\chi^2(\hat{\hat{\boldsymbol{\theta}}}(\mu))
-\chi^2(\hat{\boldsymbol{\theta}})$, clipped below at zero. Document:

$$
q_0=\begin{cases}\Delta\chi^2_0,&\hat\mu\ge0\\0,&\hat\mu<0,\end{cases}
\qquad
q_\mu=\begin{cases}\Delta\chi^2_\mu,&\hat\mu\le\mu\\0,&\hat\mu>\mu,\end{cases}
$$

and state that `qtilde` uses the same profiled difference/clipping in the
general fit path but requires a POI lower bound exactly equal to zero. Explain
the unbounded-below free fit used for `q` and the model-bound free fit used for
`q0` and `qtilde`.

- [ ] **Step 2: Derive SGHF's Asimov transformation**

Document background Asimov data for exclusion and the conditional $\mu=1$
Asimov data for discovery. Define $q_A$, then reproduce the implemented
transformation

$$
t=
\begin{cases}
\sqrt q-\sqrt{q_A},&
\tilde q_\mu\ \text{and}\ \sqrt q\le\sqrt{q_A},\\
(q-q_A)/(2\sqrt{q_A}),&
\tilde q_\mu\ \text{and}\ \sqrt q>\sqrt{q_A},\\
\sqrt q-\sqrt{q_A},&q_\mu\text{ or }q_0.
\end{cases}
$$

and the implemented probabilities
$CL_{s+b}=\Phi[-(t+\sqrt{q_A})]$,
$CL_b=\Phi(-t)$, and $CL_s=CL_{s+b}/CL_b$. State that discovery
`significance=sqrt(q0)` in the asymptotic result.

- [ ] **Step 3: Document expected asymptotic bands and assumptions**

Explain the returned ordering `(-2 sigma, -1 sigma, median, +1 sigma,
+2 sigma)` and distinguish Wald/Asimov assumptions from the earlier
linearization and Gaussian-objective approximations. Document every
`HypothesisTestResult` field, including the four diagnostic fit objects.

- [ ] **Step 4: Derive upper-limit root finding**

In `docs/upper-limits.md`, define observed and expected roots
$CL_s(\mu)=\alpha$, default $\alpha=0.05$, supported statistics `q` and
`qtilde`, cache behavior, Brent tolerance `1e-10`, and the five expected-root
ordering. State that the method is asymptotic only.

- [ ] **Step 5: Document bracketing and failures**

Explain the automatic lower endpoint `max(0, POI lower bound)`, initial upper
endpoint from the model bound, doubling of bracket width for at most 20
expansions until all six curves are at or below the level, and no expansion
for explicit brackets. Cover invalid brackets, non-finite $CL_s$, absent
crossings, and `LimitNotFound`.

- [ ] **Step 6: Audit and commit**

Run:

```bash
rg -n "_profile_statistic|q_asimov|transformed|expected_cls|brentq|20" src/sghf/asymptotic.py docs/hypothesis-testing.md docs/upper-limits.md
git add docs/hypothesis-testing.md docs/upper-limits.md
git commit -m "docs: derive asymptotic inference"
```

Expected: all implemented statistic branches, returned probabilities, expected
bands, and limit failure modes are represented.

---

### Task 4: Toy Inference

**Files:**
- Create: `docs/toy-inference.md`

**Interfaces:**
- Consumes: `Model.toys`, `Model.hypotest(..., calculator="toys")`,
  `ToyResult`, `ToyHypothesisTestResult`, and `ToyDiscoveryTestResult`.
- Produces: complete generation, calibration, discovery, exclusion, and
  performance semantics for the public toy calculator.

- [ ] **Step 1: Document pseudo-observation generation**

Define Poisson draws $n_i^*\sim\operatorname{Pois}(\nu_i)$, Gaussian draws
$n_i^*\sim N(\nu_i,\nu_i)$, and constrained auxiliary draws
$a_j^*\sim N(\theta_j,1/P_{jj})$. State that unconstrained auxiliary values
remain fixed, `fluctuate_auxiliary=False` fixes all auxiliary values, and an
integer seed or a supplied `numpy.random.Generator` controls reproducibility.

- [ ] **Step 2: Derive the vectorized nominal POI estimator**

For free-parameter design $A_F$, define

$$
I=A_F^\mathsf TV^{-1}A_F+P_F,\qquad
\mathbf c=I^{-1}\mathbf e_\mu,\qquad
\sigma_\mu^2=c_\mu,
$$

$$
\mathbf w_n=V^{-1}A_F\mathbf c,\qquad
\mathbf w_a=P_F\mathbf c,
$$

and

$$
\hat\mu=\mu_0+
\mathbf w_n^\mathsf T(\mathbf n-\boldsymbol\nu_0)+
\mathbf w_a^\mathsf T(\mathbf a_F-\boldsymbol\theta_{0,F}).
$$

State that this is exact for SGHF's nominal unconstrained quadratic estimator,
not the nonlinear HistFactory likelihood, and that rank or POI-variance
failures raise `NonIdentifiableModel`.

- [ ] **Step 3: Derive all vectorized nominal statistics**

Write the implemented piecewise formulas:

$$
q_0=\begin{cases}\hat\mu^2/\sigma_\mu^2,&\hat\mu\ge0\\0,&\hat\mu<0,\end{cases}
$$

$$
q_\mu=\begin{cases}(\mu-\hat\mu)^2/\sigma_\mu^2,&\hat\mu\le\mu\\
0,&\hat\mu>\mu,\end{cases}
$$

$$
\tilde q_\mu=\begin{cases}
0,&\hat\mu>\mu,\\
(\mu-\hat\mu)^2/\sigma_\mu^2,&0\le\hat\mu\le\mu,\\
(\mu^2-2\mu\hat\mu)/\sigma_\mu^2,&\hat\mu<0.
\end{cases}
$$

Explain that `variance="iterated"` instead profiles every toy with the fit
engine.

- [ ] **Step 4: Document exclusion ensembles and empirical probabilities**

Describe conditional signal-plus-background toys at tested $\mu$ and
background-only toys at $\mu=0$. Define inclusive upper-tail counts and

$$
\hat p(t)=\frac{\#\{T_i\ge t\}+1}{N+1},
\qquad
CL_s=\widehat{CL}_{s+b}/\widehat{CL}_b.
$$

Document observed values, per-background-toy expected $CL_s$, percentile
levels `2.27501319`, `15.86552539`, `50`, `84.13447461`, `97.72498681`, and
the returned order.

- [ ] **Step 5: Document toy-based `q0` discovery**

State that `q0` requires `poi_test=0`; null toys use the conditional
background-only fit and expected-discovery toys use the conditional $\mu=1$
fit. Define $p_0$ from the null tail, expected $p_0$ from alternative-toy
statistics evaluated against the same null distribution, and
$Z=\max[\Phi^{-1}(1-p_0),0]$. Explicitly state the minimum attainable
tail probability $1/(N+1)$.

- [ ] **Step 6: Document batching, memory, results, and defaults**

State that temporary toy matrices are batched, an explicit positive
`batch_size` is capped at `ntoys`, and the automatic batch targets at most
128 MiB using the estimate
$8N_\mathrm{batch}(N_\mathrm{bins}+N_\mathrm{parameters})$. Clarify that both
length-`ntoys` statistic arrays are retained in the result. Document nominal
variance as the toy-calculator default and iterated variance as the
asymptotic-calculator default. Cover every field of all three toy result
dataclasses.

- [ ] **Step 7: Audit and commit**

Run:

```bash
rg -n "_nominal_poi_estimates|_nominal_test_statistics|_tail_probability|_toy_discovery|_MAX_TOY_BATCH_BYTES" src/sghf/toys.py docs/toy-inference.md
git add docs/toy-inference.md
git commit -m "docs: derive toy-based inference"
```

Expected: generation, three statistics, exclusion, discovery, finite-sample
calibration, both variance paths, and memory behavior are documented.

---

### Task 5: API Reference and Applicability

**Files:**
- Create: `docs/api-reference.md`
- Create: `docs/validation-and-applicability.md`

**Interfaces:**
- Consumes: all public exports in `src/sghf/__init__.py`, signatures in
  `model.py`, and result fields in `fit.py`, `asymptotic.py`, and `toys.py`.
- Produces: a complete lookup layer and scientific-use checklist.

- [ ] **Step 1: Inventory and document the public API**

Use this command as the authoritative inventory:

```bash
rg -n "^class |^def |    def |__all__" src/sghf/__init__.py src/sghf/model.py src/sghf/fit.py src/sghf/asymptotic.py src/sghf/toys.py src/sghf/exceptions.py
```

Document `sghf.load`, every `Model` property/method, `FitResult` helpers, all
public dataclass fields, and all exported exceptions. Copy signatures and
defaults exactly, including calculator-dependent variance defaults.

- [ ] **Step 2: Add minimal public-object examples**

Include runnable examples for loading a mapping or JSON path, inspecting
parameters/responses, evaluating expected yields, fitting, asymptotic
hypothesis testing, toy exclusion, toy discovery, toy generation, and
asymptotic upper limits. Use only `sghf.load` and `Model` methods. Document that
pyhf-style `Workspace` and `infer.*` calls raise `SGHFUsageError` with object
API guidance.

- [ ] **Step 3: Write the applicability checklist**

In `docs/validation-and-applicability.md`, make each item actionable:
compare modifier slopes near the initial point; inspect high/low asymmetry;
check expected counts; inspect pulls and `at_bound`; ensure all fitted
expectations remain positive; assess covariance rank/conditioning; compare
nominal and iterated results; and compare relevant observed/expected results
against pyhf for final analyses.

- [ ] **Step 4: State what validation can establish**

Explain that nominal expectation and local derivative parity test compilation
at the expansion point but do not establish tail calibration, nonlinear
modifier agreement, boundary behavior, or coverage. Require pyhf cross-checks
when results depend on large shifts, sparse tails, active bounds, strong
asymmetry, or nominal-versus-iterated differences.

- [ ] **Step 5: Audit and commit**

Run:

```bash
python - <<'PY'
import sghf
print("\\n".join(sghf.__all__))
PY
rg -n "^## |`Model\\.|`FitResult|`HypothesisTestResult|`LimitResult|`Toy" docs/api-reference.md
git add docs/api-reference.md docs/validation-and-applicability.md
git commit -m "docs: add API and applicability guidance"
```

Expected: every name printed from `sghf.__all__` is documented or included in
the exceptions table, and the checklist distinguishes local parity from
end-to-end statistical validation.

---

### Task 6: Navigation, References, Automated Checks, and Final Verification

**Files:**
- Create: `docs/index.md`
- Create: `docs/references.md`
- Create: `tests/test_documentation.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all nine focused pages from Tasks 1-5.
- Produces: complete navigation and a falsifiable documentation integrity test.

- [ ] **Step 1: Write the landing page**

Add a five-minute flow `load -> approximation -> fit -> hypotest/toys ->
upper_limit`, the fixed notation table, a calculator comparison table, and a
complete ordered table of contents linking all ten pages. State at the top that
SGHF is a fast approximation and that final scientific applicability depends
on the diagnostics page.

- [ ] **Step 2: Add primary references**

Add linked entries for:

- Cranmer et al., *HistFactory: A tool for creating statistical models for use
  with RooFit and RooStats*, CERN-OPEN-2012-016,
  DOI `10.17181/CERN-OPEN-2012-016`;
- Heinrich et al., *pyhf: pure-Python implementation of HistFactory
  statistical models*, JOSS 2021, DOI `10.21105/joss.02823`;
- Cowan et al., *Asymptotic formulae for likelihood-based tests of new
  physics*, EPJC 2011, DOI `10.1140/epjc/s10052-011-1554-0`;
- Read, *Presentation of search results: the CLs technique*, J. Phys. G 2002,
  DOI `10.1088/0954-3899/28/10/313`;
- Phipson and Smyth, *Permutation P-values should never be zero*, Statistical
  Applications in Genetics and Molecular Biology 2010,
  DOI `10.2202/1544-6115.1585`.

Link each theory page to the relevant reference section without using the
sources to imply a stronger guarantee than SGHF implements.

- [ ] **Step 3: Add page-level navigation and README links**

Give every focused page a first-line index link and previous/next links. Add a
README documentation section linking directly to simplified likelihood,
hypothesis testing, toy inference, applicability, and API reference while
retaining the existing installation and quick-start content.

- [ ] **Step 4: Write the documentation integrity test**

Create `tests/test_documentation.py` with:

```python
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REQUIRED_PAGES = {
    "index.md",
    "simplified-likelihood.md",
    "workspace-compilation.md",
    "fitting.md",
    "hypothesis-testing.md",
    "toy-inference.md",
    "upper-limits.md",
    "api-reference.md",
    "validation-and-applicability.md",
    "references.md",
}
MARKDOWN_LINK = re.compile(r"(?<!!)\\[[^]]*\\]\\(([^)]+)\\)")


def test_required_documentation_pages_exist_and_are_linked():
    assert REQUIRED_PAGES <= {path.name for path in DOCS.glob("*.md")}
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for page in REQUIRED_PAGES - {"index.md"}:
        assert f"]({page})" in index


def test_relative_markdown_links_resolve():
    files = [ROOT / "README.md", *(DOCS / name for name in REQUIRED_PAGES)]
    for source in files:
        text = source.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (source.parent / target).resolve()
            assert resolved.exists(), f"{source}: missing link target {raw_target}"
```

- [ ] **Step 5: Run focused documentation checks**

Run:

```bash
.venv/bin/python -m pytest tests/test_documentation.py -v
rg -n "FIXME|draft text|exact pyhf|exact coverage" README.md docs/*.md
```

Expected: both documentation tests pass; the scan returns only explicit
warnings against exact-parity or exact-coverage claims, and no draft markers.

- [ ] **Step 6: Execute runnable documentation examples**

Extract or manually reproduce every Python code block marked as runnable using
the public example workspace in `tests/conftest.py` or an equivalent complete
inline workspace. Run them with:

```bash
.venv/bin/python /tmp/sghf-doc-examples.py
```

Expected: exit code 0; loading, fitting, asymptotic tests, toy exclusion,
toy discovery, toy generation, and upper-limit examples all complete.

- [ ] **Step 7: Run full verification**

Run:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m build
git diff --check
git status --short
```

Expected: the full test suite passes, wheel and sdist are built, the diff has
no whitespace errors, and only intended documentation/build artifacts are
present.

- [ ] **Step 8: Commit and push the completed documentation**

```bash
git add README.md docs/index.md docs/references.md tests/test_documentation.py
git commit -m "docs: add layered navigation and references"
git push origin main
```

Expected: `main` is synchronized with `origin/main` and the GitHub Actions run
for the pushed commit succeeds before reporting completion.
