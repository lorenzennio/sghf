from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .exceptions import LimitNotFound
from .fit import FitResult, _data_array

if TYPE_CHECKING:
    from .model import Model


TestStatistic = Literal["q0", "q", "qtilde"]
Variance = Literal["iterated", "nominal"]


@dataclass(frozen=True)
class HypothesisTestResult:
    """Asymptotic profile-test result."""

    poi_test: float
    test_stat: TestStatistic
    q: float
    q_asimov: float
    transformed: float
    clsb: float
    clb: float
    cls: float
    expected_cls: tuple[float, float, float, float, float]
    significance: float
    free_fit: FitResult
    fixed_fit: FitResult
    asimov_free_fit: FitResult
    asimov_fixed_fit: FitResult


@dataclass(frozen=True)
class LimitResult:
    """Observed and expected asymptotic upper limits."""

    observed: float
    expected: tuple[float, float, float, float, float]
    level: float
    test_stat: Literal["q", "qtilde"]
    bracket: tuple[float, float]
    results: tuple[HypothesisTestResult, ...] = ()
    """Every test evaluated while locating the crossings, sorted by POI value.

    These are the points the bracket expansion and root finder happened to
    visit, not a regular scan: they cluster tightly around each crossing and
    leave wide gaps between them. Useful for plotting and diagnostics, but do
    not read the spacing as structure.
    """


def _profile_statistic(
    model: Model,
    poi_test: float,
    data: Sequence[float] | Mapping[str, Sequence[float]],
    *,
    auxdata: Sequence[float] | None,
    test_stat: TestStatistic,
    variance: Variance,
) -> tuple[float, FitResult, FitResult]:
    poi_index = model.parameter_index(model.poi_name)
    free_bounds = None
    if test_stat == "q":
        free_bounds = {
            model.poi_name: (-np.inf, float(model.upper_bounds[poi_index]))
        }
    free_fit = model.fit(
        data,
        auxdata=auxdata,
        bounds=free_bounds,
        variance=variance,
    )
    fixed_bounds = {
        model.poi_name: (
            min(float(model.lower_bounds[poi_index]), poi_test),
            max(float(model.upper_bounds[poi_index]), poi_test),
        )
    }
    fixed_fit = model.fit(
        data,
        auxdata=auxdata,
        fixed={model.poi_name: poi_test},
        bounds=fixed_bounds,
        variance=variance,
    )
    mu_hat = free_fit.value(model.poi_name)
    delta = max(0.0, fixed_fit.chi2 - free_fit.chi2)
    if test_stat in {"q", "qtilde"} and mu_hat > poi_test:
        delta = 0.0
    if test_stat == "q0" and mu_hat < 0.0:
        delta = 0.0
    return delta, free_fit, fixed_fit


def _pvalues(
    transformed: float,
    sqrt_q_asimov: float,
    *,
    clipped: bool,
) -> tuple[float, float, float]:
    cutoff = -sqrt_q_asimov if clipped else -np.inf
    if transformed < cutoff:
        return np.nan, np.nan, np.nan
    clsb = float(norm.cdf(-(transformed + sqrt_q_asimov)))
    clb = float(norm.cdf(-transformed))
    cls = clsb / clb if clb > 0.0 else np.nan
    return clsb, clb, cls


def _expected_cls(
    sqrt_q_asimov: float, *, clipped: bool
) -> tuple[float, float, float, float, float]:
    cutoff = -sqrt_q_asimov if clipped else -np.inf
    values: list[float] = []
    for n_sigma in (2.0, 1.0, 0.0, -1.0, -2.0):
        transformed = max(n_sigma, cutoff)
        values.append(
            _pvalues(
                transformed,
                sqrt_q_asimov,
                clipped=clipped,
            )[2]
        )
    return tuple(values)  # type: ignore[return-value]


def hypotest(
    model: Model,
    poi_test: float,
    data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    *,
    test_stat: TestStatistic = "qtilde",
    variance: Variance = "nominal",
) -> HypothesisTestResult:
    """Run an asymptotic profile test using the compiled quadratic model."""

    if test_stat not in {"q0", "q", "qtilde"}:
        raise ValueError("test_stat must be 'q0', 'q', or 'qtilde'")
    poi_test = float(poi_test)
    if not np.isfinite(poi_test):
        raise ValueError("poi_test must be finite")
    if test_stat == "q0" and poi_test != 0.0:
        raise ValueError("q0 requires poi_test=0")
    poi_index = model.parameter_index(model.poi_name)
    if test_stat == "qtilde" and model.lower_bounds[poi_index] != 0.0:
        raise ValueError("qtilde requires the POI lower bound to be zero")

    observed = _data_array(model, data)
    q, free_fit, fixed_fit = _profile_statistic(
        model,
        poi_test,
        observed,
        auxdata=None,
        test_stat=test_stat,
        variance=variance,
    )

    asimov_mu = 1.0 if test_stat == "q0" else 0.0
    asimov_truth = model.fit(
        observed,
        fixed={model.poi_name: asimov_mu},
        variance=variance,
    )
    asimov_data = model.expected(asimov_truth)
    asimov_auxdata = asimov_truth.values
    q_asimov, asimov_free_fit, asimov_fixed_fit = _profile_statistic(
        model,
        poi_test,
        asimov_data,
        auxdata=asimov_auxdata,
        test_stat=test_stat,
        variance=variance,
    )

    sqrt_q = float(np.sqrt(q))
    sqrt_q_asimov = float(np.sqrt(q_asimov))
    clipped = test_stat == "qtilde"
    if (
        clipped
        and sqrt_q_asimov > 0.0
        and sqrt_q > sqrt_q_asimov
    ):
        transformed = (q - q_asimov) / (2.0 * sqrt_q_asimov)
    else:
        transformed = sqrt_q - sqrt_q_asimov
    clsb, clb, cls = _pvalues(
        transformed, sqrt_q_asimov, clipped=clipped
    )
    return HypothesisTestResult(
        poi_test=poi_test,
        test_stat=test_stat,
        q=q,
        q_asimov=q_asimov,
        transformed=float(transformed),
        clsb=clsb,
        clb=clb,
        cls=cls,
        expected_cls=_expected_cls(sqrt_q_asimov, clipped=clipped),
        significance=sqrt_q if test_stat == "q0" else np.nan,
        free_fit=free_fit,
        fixed_fit=fixed_fit,
        asimov_free_fit=asimov_free_fit,
        asimov_fixed_fit=asimov_fixed_fit,
    )


def upper_limit(
    model: Model,
    data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    *,
    level: float = 0.05,
    test_stat: Literal["q", "qtilde"] = "qtilde",
    variance: Variance = "nominal",
    bracket: tuple[float, float] | None = None,
) -> LimitResult:
    """Find CLs upper-limit crossings with scalar root finding."""

    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    poi_index = model.parameter_index(model.poi_name)
    automatic_bracket = bracket is None
    if automatic_bracket:
        low = max(0.0, float(model.lower_bounds[poi_index]))
        high = float(model.upper_bounds[poi_index])
    else:
        low, high = map(float, bracket)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("bracket must contain two finite increasing values")

    cache: dict[float, HypothesisTestResult] = {}

    def result_at(value: float) -> HypothesisTestResult:
        if value not in cache:
            cache[value] = hypotest(
                model,
                value,
                data=data,
                test_stat=test_stat,
                variance=variance,
            )
        return cache[value]

    def crossing(index: int | None) -> float:
        def objective(value: float) -> float:
            result = result_at(value)
            cls = result.cls if index is None else result.expected_cls[index]
            return cls - level

        at_low, at_high = objective(low), objective(high)
        if not np.isfinite(at_low) or not np.isfinite(at_high):
            raise LimitNotFound("CLs is non-finite within the requested bracket")
        if at_low == 0.0:
            return low
        if at_high == 0.0:
            return high
        if at_low * at_high > 0.0:
            raise LimitNotFound(
                f"No CLs={level:g} crossing in bracket ({low:g}, {high:g})"
            )
        return float(brentq(objective, low, high, xtol=1.0e-10))

    if automatic_bracket:
        for _ in range(20):
            result = result_at(high)
            values = (result.cls, *result.expected_cls)
            if all(np.isfinite(value) and value <= level for value in values):
                break
            high = low + 2.0 * (high - low)
        else:
            raise LimitNotFound(
                f"Could not bracket all CLs={level:g} crossings after "
                "20 expansions"
            )

    observed = crossing(None)
    expected = tuple(crossing(index) for index in range(5))
    return LimitResult(
        observed=observed,
        expected=expected,  # type: ignore[arg-type]
        level=level,
        test_stat=test_stat,
        bracket=(low, high),
        results=tuple(cache[key] for key in sorted(cache)),
    )
