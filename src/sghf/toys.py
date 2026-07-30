from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from .asymptotic import TestStatistic, Variance, _profile_statistic
from .fit import FitResult, _data_array

if TYPE_CHECKING:
    from .model import Model


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ToyResult:
    """Pseudo-observations generated from a compiled SGHF model."""

    observations: NDArray[np.integer] | FloatArray
    auxiliary: FloatArray
    parameters: FloatArray
    expected: FloatArray
    sampling: Literal["poisson", "gaussian"]


@dataclass(frozen=True)
class ToyHypothesisTestResult:
    """Empirical profile-test result from pseudo-experiments."""

    poi_test: float
    test_stat: TestStatistic
    q: float
    clsb: float
    clb: float
    cls: float
    expected_cls: tuple[float, float, float, float, float]
    ntoys: int
    sb_statistics: FloatArray
    b_statistics: FloatArray
    free_fit: FitResult
    fixed_fit: FitResult


def _rng(value: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(value, np.random.Generator):
        return value
    return np.random.default_rng(value)


def _parameter_values(
    model: Model,
    parameters: Sequence[float] | Mapping[str, float] | FitResult | None,
) -> FloatArray:
    if parameters is None:
        return model.initial.copy()
    if isinstance(parameters, FitResult):
        values = np.asarray(parameters.values, dtype=float)
    elif isinstance(parameters, Mapping):
        values = model.initial.copy()
        for name, value in parameters.items():
            values[model.parameter_index(name)] = float(value)
    else:
        values = np.asarray(parameters, dtype=float)
    if values.shape != model.initial.shape:
        raise ValueError(
            f"Parameters must contain {model.initial.size} value(s)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Parameter values must be finite")
    return values


def generate_toys(
    model: Model,
    size: int,
    *,
    parameters: Sequence[float] | Mapping[str, float] | FitResult | None = None,
    sampling: Literal["poisson", "gaussian"] = "poisson",
    fluctuate_auxiliary: bool = True,
    rng: int | np.random.Generator | None = None,
) -> ToyResult:
    """Generate main and Gaussian auxiliary pseudo-observations."""

    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be positive")
    if sampling not in {"poisson", "gaussian"}:
        raise ValueError("sampling must be 'poisson' or 'gaussian'")
    values = _parameter_values(model, parameters)
    expected = model.expected(values)
    generator = _rng(rng)
    if sampling == "poisson":
        observations = generator.poisson(expected, size=(size, expected.size))
    else:
        observations = generator.normal(
            expected,
            np.sqrt(expected),
            size=(size, expected.size),
        )

    auxiliary = np.broadcast_to(values, (size, values.size)).copy()
    constrained = model.constraint_precision > 0
    if fluctuate_auxiliary and np.any(constrained):
        sigma = 1.0 / np.sqrt(model.constraint_precision[constrained])
        auxiliary[:, constrained] = generator.normal(
            values[constrained],
            sigma,
            size=(size, int(np.count_nonzero(constrained))),
        )
    return ToyResult(
        observations=observations,
        auxiliary=auxiliary,
        parameters=values,
        expected=expected,
        sampling=sampling,
    )


def _tail_probability(sorted_statistics: FloatArray, value: float) -> float:
    count = sorted_statistics.size - np.searchsorted(
        sorted_statistics, value, side="left"
    )
    return float((count + 1) / (sorted_statistics.size + 1))


def toy_hypotest(
    model: Model,
    poi_test: float,
    data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    *,
    test_stat: TestStatistic = "qtilde",
    variance: Variance = "iterated",
    ntoys: int = 10_000,
    sampling: Literal["poisson", "gaussian"] = "poisson",
    rng: int | np.random.Generator | None = None,
) -> ToyHypothesisTestResult:
    """Run an exclusion test with generated pseudo-experiments."""

    if test_stat not in {"q0", "q", "qtilde"}:
        raise ValueError("test_stat must be 'q0', 'q', or 'qtilde'")
    if test_stat == "q0":
        raise NotImplementedError(
            "Toy-based q0 discovery tests are not implemented; "
            "use calculator='asymptotic'"
        )
    if ntoys <= 0:
        raise ValueError("ntoys must be positive")
    observed = _data_array(model, data)
    q, free_fit, fixed_fit = _profile_statistic(
        model,
        float(poi_test),
        observed,
        auxdata=None,
        test_stat=test_stat,
        variance=variance,
    )
    background_fit = model.fit(
        observed,
        fixed={model.poi_name: 0.0},
        variance=variance,
    )
    generator = _rng(rng)
    sb_toys = generate_toys(
        model,
        ntoys,
        parameters=fixed_fit,
        sampling=sampling,
        rng=generator,
    )
    b_toys = generate_toys(
        model,
        ntoys,
        parameters=background_fit,
        sampling=sampling,
        rng=generator,
    )

    sb_statistics = np.empty(ntoys)
    b_statistics = np.empty(ntoys)
    for index in range(ntoys):
        sb_statistics[index] = _profile_statistic(
            model,
            float(poi_test),
            sb_toys.observations[index],
            auxdata=sb_toys.auxiliary[index],
            test_stat=test_stat,
            variance=variance,
        )[0]
        b_statistics[index] = _profile_statistic(
            model,
            float(poi_test),
            b_toys.observations[index],
            auxdata=b_toys.auxiliary[index],
            test_stat=test_stat,
            variance=variance,
        )[0]

    sorted_sb = np.sort(sb_statistics)
    sorted_b = np.sort(b_statistics)
    clsb = _tail_probability(sorted_sb, q)
    clb = _tail_probability(sorted_b, q)
    toy_cls = np.empty(ntoys)
    for index, statistic in enumerate(b_statistics):
        toy_clsb = _tail_probability(sorted_sb, statistic)
        toy_clb = _tail_probability(sorted_b, statistic)
        toy_cls[index] = toy_clsb / toy_clb
    expected_cls = tuple(
        float(value)
        for value in np.percentile(
            toy_cls,
            [2.27501319, 15.86552539, 50.0, 84.13447461, 97.72498681],
        )
    )
    return ToyHypothesisTestResult(
        poi_test=float(poi_test),
        test_stat=test_stat,
        q=q,
        clsb=clsb,
        clb=clb,
        cls=clsb / clb,
        expected_cls=expected_cls,  # type: ignore[arg-type]
        ntoys=ntoys,
        sb_statistics=sb_statistics,
        b_statistics=b_statistics,
        free_fit=free_fit,
        fixed_fit=fixed_fit,
    )
