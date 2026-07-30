from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .asymptotic import TestStatistic, Variance, _profile_statistic
from .exceptions import NonIdentifiableModel
from .fit import FitResult, _data_array

if TYPE_CHECKING:
    from .model import Model


FloatArray = NDArray[np.float64]
_MAX_TOY_BATCH_BYTES = 128 * 1024**2


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
    batch_size: int
    variance: Variance
    sb_statistics: FloatArray
    b_statistics: FloatArray
    free_fit: FitResult
    fixed_fit: FitResult


@dataclass(frozen=True)
class ToyDiscoveryTestResult:
    """Empirical ``q0`` discovery result from pseudo-experiments."""

    poi_test: float
    test_stat: Literal["q0"]
    q: float
    p0: float
    significance: float
    expected_p0: tuple[float, float, float, float, float]
    expected_significance: tuple[float, float, float, float, float]
    ntoys: int
    batch_size: int
    variance: Variance
    null_statistics: FloatArray
    alternative_statistics: FloatArray
    free_fit: FitResult
    fixed_fit: FitResult
    alternative_fit: FitResult


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


def _tail_probabilities(
    sorted_statistics: FloatArray,
    values: FloatArray,
) -> FloatArray:
    counts = sorted_statistics.size - np.searchsorted(
        sorted_statistics,
        values,
        side="left",
    )
    return (counts + 1) / (sorted_statistics.size + 1)


def _nominal_poi_estimates(
    model: Model,
    observations: NDArray[np.floating] | NDArray[np.integer],
    auxiliary: NDArray[np.floating],
) -> tuple[FloatArray, float]:
    """Evaluate the nominal linear-Gaussian POI estimator in one batch."""

    observed = np.asarray(observations, dtype=float)
    auxdata = np.asarray(auxiliary, dtype=float)
    if observed.ndim == 1:
        observed = observed[np.newaxis, :]
    if auxdata.ndim == 1:
        auxdata = auxdata[np.newaxis, :]
    if observed.ndim != 2 or observed.shape[1] != model.nominal_expected.size:
        raise ValueError("Toy observations have the wrong shape")
    if auxdata.shape != (observed.shape[0], model.initial.size):
        raise ValueError("Toy auxiliary data have the wrong shape")

    free = np.flatnonzero(~model.fixed)
    poi_index = model.parameter_index(model.poi_name)
    poi_locations = np.flatnonzero(free == poi_index)
    if poi_locations.size != 1:
        raise ValueError("The parameter of interest must be free for a toy test")
    poi_local = int(poi_locations[0])

    variances = model.nominal_expected
    design = model.response[:, free]
    precision = model.constraint_precision[free]
    information = design.T @ (design / variances[:, np.newaxis])
    information += np.diag(precision)
    rank = np.linalg.matrix_rank(information)
    if rank < free.size:
        raise NonIdentifiableModel(
            f"Free model parameters are not identifiable "
            f"(information rank {rank} < {free.size})"
        )

    poi_unit = np.zeros(free.size)
    poi_unit[poi_local] = 1.0
    poi_covariance_column = np.linalg.solve(information, poi_unit)
    poi_variance = float(poi_covariance_column[poi_local])
    if not np.isfinite(poi_variance) or poi_variance <= 0.0:
        raise NonIdentifiableModel(
            "The parameter-of-interest variance is not finite and positive"
        )

    data_weights = (
        design @ poi_covariance_column
    ) / variances
    auxiliary_weights = precision * poi_covariance_column
    estimates = (
        model.initial[poi_index]
        + (observed - model.nominal_expected) @ data_weights
        + (
            auxdata[:, free] - model.initial[free]
        ) @ auxiliary_weights
    )
    return np.asarray(estimates, dtype=float), poi_variance


def _nominal_test_statistics(
    model: Model,
    poi_test: float,
    observations: NDArray[np.floating] | NDArray[np.integer],
    auxiliary: NDArray[np.floating],
    *,
    test_stat: TestStatistic,
) -> FloatArray:
    """Vectorized Wald-equivalent statistics for the nominal quadratic model."""

    estimates, poi_variance = _nominal_poi_estimates(
        model,
        observations,
        auxiliary,
    )
    if test_stat == "q0":
        if poi_test != 0.0:
            raise ValueError("q0 requires poi_test=0")
        statistics = np.where(
            estimates >= 0.0,
            np.square(estimates) / poi_variance,
            0.0,
        )
    elif test_stat == "q":
        statistics = np.where(
            estimates <= poi_test,
            np.square(poi_test - estimates) / poi_variance,
            0.0,
        )
    elif test_stat == "qtilde":
        poi_index = model.parameter_index(model.poi_name)
        if model.lower_bounds[poi_index] != 0.0:
            raise ValueError("qtilde requires the POI lower bound to be zero")
        statistics = np.where(
            estimates > poi_test,
            0.0,
            np.where(
                estimates >= 0.0,
                np.square(poi_test - estimates) / poi_variance,
                (poi_test**2 - 2.0 * poi_test * estimates) / poi_variance,
            ),
        )
    else:
        raise ValueError("test_stat must be 'q0', 'q', or 'qtilde'")
    return np.maximum(np.asarray(statistics, dtype=float), 0.0)


def _toy_statistics(
    model: Model,
    poi_test: float,
    toys: ToyResult,
    *,
    test_stat: TestStatistic,
    variance: Variance,
) -> FloatArray:
    if variance == "nominal":
        return _nominal_test_statistics(
            model,
            poi_test,
            toys.observations,
            toys.auxiliary,
            test_stat=test_stat,
        )
    statistics = np.empty(toys.observations.shape[0])
    for index in range(statistics.size):
        statistics[index] = _profile_statistic(
            model,
            poi_test,
            toys.observations[index],
            auxdata=toys.auxiliary[index],
            test_stat=test_stat,
            variance=variance,
        )[0]
    return statistics


def _resolve_batch_size(
    model: Model,
    ntoys: int,
    batch_size: int | None,
) -> int:
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(
            batch_size,
            (int, np.integer),
        ):
            raise TypeError("batch_size must be an integer or None")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return min(int(batch_size), ntoys)
    bytes_per_toy = 8 * (
        model.nominal_expected.size + model.initial.size
    )
    memory_limited = max(1, _MAX_TOY_BATCH_BYTES // bytes_per_toy)
    return min(ntoys, memory_limited)


def _generate_test_statistics(
    model: Model,
    poi_test: float,
    *,
    parameters: FitResult,
    test_stat: TestStatistic,
    variance: Variance,
    sampling: Literal["poisson", "gaussian"],
    ntoys: int,
    batch_size: int,
    generator: np.random.Generator,
) -> FloatArray:
    statistics = np.empty(ntoys)
    for start in range(0, ntoys, batch_size):
        stop = min(start + batch_size, ntoys)
        toys = generate_toys(
            model,
            stop - start,
            parameters=parameters,
            sampling=sampling,
            rng=generator,
        )
        statistics[start:stop] = _toy_statistics(
            model,
            poi_test,
            toys,
            test_stat=test_stat,
            variance=variance,
        )
    return statistics


def _conditional_fit(
    model: Model,
    observed: FloatArray,
    poi_value: float,
    *,
    variance: Variance,
) -> FitResult:
    poi_index = model.parameter_index(model.poi_name)
    return model.fit(
        observed,
        fixed={model.poi_name: poi_value},
        bounds={
            model.poi_name: (
                min(float(model.lower_bounds[poi_index]), poi_value),
                max(float(model.upper_bounds[poi_index]), poi_value),
            )
        },
        variance=variance,
    )


def _toy_discovery(
    model: Model,
    observed: FloatArray,
    *,
    variance: Variance,
    ntoys: int,
    batch_size: int,
    sampling: Literal["poisson", "gaussian"],
    generator: np.random.Generator,
) -> ToyDiscoveryTestResult:
    _, free_fit, fixed_fit = _profile_statistic(
        model,
        0.0,
        observed,
        auxdata=None,
        test_stat="q0",
        variance=variance,
    )
    alternative_fit = _conditional_fit(
        model,
        observed,
        1.0,
        variance=variance,
    )
    null_statistics = _generate_test_statistics(
        model,
        0.0,
        parameters=fixed_fit,
        test_stat="q0",
        variance=variance,
        sampling=sampling,
        ntoys=ntoys,
        batch_size=batch_size,
        generator=generator,
    )
    alternative_statistics = _generate_test_statistics(
        model,
        0.0,
        parameters=alternative_fit,
        test_stat="q0",
        variance=variance,
        sampling=sampling,
        ntoys=ntoys,
        batch_size=batch_size,
        generator=generator,
    )
    if variance == "nominal":
        observed_q = float(
            _nominal_test_statistics(
                model,
                0.0,
                observed,
                model.constraint_mean,
                test_stat="q0",
            )[0]
        )
    else:
        observed_q = max(0.0, fixed_fit.chi2 - free_fit.chi2)

    sorted_null = np.sort(null_statistics)
    p0 = _tail_probability(sorted_null, observed_q)
    alternative_p0 = _tail_probabilities(
        sorted_null,
        alternative_statistics,
    )
    expected_p0 = tuple(
        float(value)
        for value in np.percentile(
            alternative_p0,
            [2.27501319, 15.86552539, 50.0, 84.13447461, 97.72498681],
        )
    )
    expected_significance = tuple(
        float(value)
        for value in np.maximum(norm.isf(expected_p0), 0.0)
    )
    return ToyDiscoveryTestResult(
        poi_test=0.0,
        test_stat="q0",
        q=observed_q,
        p0=p0,
        significance=float(max(norm.isf(p0), 0.0)),
        expected_p0=expected_p0,  # type: ignore[arg-type]
        expected_significance=expected_significance,  # type: ignore[arg-type]
        ntoys=ntoys,
        batch_size=batch_size,
        variance=variance,
        null_statistics=null_statistics,
        alternative_statistics=alternative_statistics,
        free_fit=free_fit,
        fixed_fit=fixed_fit,
        alternative_fit=alternative_fit,
    )


def toy_hypotest(
    model: Model,
    poi_test: float,
    data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    *,
    test_stat: TestStatistic = "qtilde",
    variance: Variance = "nominal",
    ntoys: int = 10_000,
    batch_size: int | None = None,
    sampling: Literal["poisson", "gaussian"] = "poisson",
    rng: int | np.random.Generator | None = None,
) -> ToyHypothesisTestResult | ToyDiscoveryTestResult:
    """Run a vectorized discovery or exclusion pseudo-experiment test."""

    if test_stat not in {"q0", "q", "qtilde"}:
        raise ValueError("test_stat must be 'q0', 'q', or 'qtilde'")
    poi_test = float(poi_test)
    if not np.isfinite(poi_test):
        raise ValueError("poi_test must be finite")
    if test_stat == "q0" and poi_test != 0.0:
        raise ValueError("q0 requires poi_test=0")
    if variance not in {"iterated", "nominal"}:
        raise ValueError("variance must be 'iterated' or 'nominal'")
    if isinstance(ntoys, bool) or not isinstance(ntoys, (int, np.integer)):
        raise TypeError("ntoys must be an integer")
    if ntoys <= 0:
        raise ValueError("ntoys must be positive")
    ntoys = int(ntoys)
    batch_size = _resolve_batch_size(model, ntoys, batch_size)
    observed = _data_array(model, data)
    generator = _rng(rng)
    if test_stat == "q0":
        return _toy_discovery(
            model,
            observed,
            variance=variance,
            ntoys=ntoys,
            batch_size=batch_size,
            sampling=sampling,
            generator=generator,
        )

    q, free_fit, fixed_fit = _profile_statistic(
        model,
        poi_test,
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
    sb_statistics = _generate_test_statistics(
        model,
        poi_test,
        parameters=fixed_fit,
        test_stat=test_stat,
        variance=variance,
        sampling=sampling,
        ntoys=ntoys,
        batch_size=batch_size,
        generator=generator,
    )
    b_statistics = _generate_test_statistics(
        model,
        poi_test,
        parameters=background_fit,
        test_stat=test_stat,
        variance=variance,
        sampling=sampling,
        ntoys=ntoys,
        batch_size=batch_size,
        generator=generator,
    )
    if variance == "nominal":
        q = float(
            _nominal_test_statistics(
                model,
                poi_test,
                observed,
                model.constraint_mean,
                test_stat=test_stat,
            )[0]
        )

    sorted_sb = np.sort(sb_statistics)
    sorted_b = np.sort(b_statistics)
    clsb = _tail_probability(sorted_sb, q)
    clb = _tail_probability(sorted_b, q)
    toy_cls = _tail_probabilities(
        sorted_sb, b_statistics
    ) / _tail_probabilities(sorted_b, b_statistics)
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
        batch_size=batch_size,
        variance=variance,
        sb_statistics=sb_statistics,
        b_statistics=b_statistics,
        free_fit=free_fit,
        fixed_fit=fixed_fit,
    )
