from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from .exceptions import (
    FitConvergenceError,
    ModelDomainError,
    NonIdentifiableModel,
)

if TYPE_CHECKING:
    from .model import Model


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FitResult:
    """Result of a constrained SGHF fit."""

    parameter_names: tuple[str, ...]
    values: FloatArray
    errors: FloatArray
    covariance: FloatArray
    correlation: FloatArray
    pulls: FloatArray
    expected: FloatArray
    chi2: float
    converged: bool
    iterations: int
    at_bound: tuple[str, ...]

    def parameter_index(self, name: str) -> int:
        try:
            return self.parameter_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown parameter {name!r}") from error

    def value(self, name: str) -> float:
        return float(self.values[self.parameter_index(name)])

    def error(self, name: str) -> float:
        return float(self.errors[self.parameter_index(name)])

    def __getitem__(self, name: str) -> float:
        return self.value(name)


def _data_array(
    model: Model,
    data: Sequence[float] | Mapping[str, Sequence[float]] | None,
) -> FloatArray:
    if data is None:
        result = model.observations.copy()
    elif isinstance(data, Mapping):
        missing = set(model.channel_names).difference(data)
        extra = set(data).difference(model.channel_names)
        if missing or extra:
            raise ValueError(
                f"Channel data mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        result = np.concatenate(
            [np.asarray(data[name], dtype=float) for name in model.channel_names]
        )
    else:
        result = np.asarray(data, dtype=float)
    if result.shape != model.nominal_expected.shape:
        raise ValueError(
            f"Data must contain {model.nominal_expected.size} bin value(s)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Data values must be finite")
    return result


def _parameter_options(
    model: Model,
    fixed: Mapping[str, float] | None,
    bounds: Mapping[str, tuple[float, float]] | None,
) -> tuple[NDArray[np.bool_], FloatArray, FloatArray, FloatArray]:
    is_fixed = model.fixed.copy()
    fixed_values = model.initial.copy()
    lower = model.lower_bounds.copy()
    upper = model.upper_bounds.copy()
    for name, value in (fixed or {}).items():
        index = model.parameter_index(name)
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Fixed value for {name!r} must be finite")
        is_fixed[index] = True
        fixed_values[index] = value
    for name, pair in (bounds or {}).items():
        index = model.parameter_index(name)
        if len(pair) != 2:
            raise ValueError(f"Bounds for {name!r} must be a (low, high) pair")
        low, high = map(float, pair)
        if np.isnan(low) or np.isnan(high) or low >= high:
            raise ValueError(f"Bounds for {name!r} are invalid")
        lower[index], upper[index] = low, high
    if np.any(fixed_values[is_fixed] < lower[is_fixed]) or np.any(
        fixed_values[is_fixed] > upper[is_fixed]
    ):
        raise ValueError("A fixed parameter value lies outside its bounds")
    return is_fixed, fixed_values, lower, upper


def _constraint_means(
    model: Model,
    auxdata: Sequence[float] | Mapping[str, float] | None,
) -> FloatArray:
    if auxdata is None:
        result = model.constraint_mean.copy()
    elif isinstance(auxdata, Mapping):
        result = model.constraint_mean.copy()
        for name, value in auxdata.items():
            index = model.parameter_index(name)
            if model.constraint_precision[index] == 0:
                raise ValueError(f"Parameter {name!r} has no auxiliary constraint")
            result[index] = float(value)
    else:
        result = np.asarray(auxdata, dtype=float)
    if result.shape != model.initial.shape:
        raise ValueError(
            f"Auxiliary data must contain {model.initial.size} parameter value(s)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Auxiliary data values must be finite")
    return result


def _solve_once(
    model: Model,
    data: FloatArray,
    variances: FloatArray,
    is_fixed: NDArray[np.bool_],
    fixed_values: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    constraint_mean: FloatArray,
) -> tuple[FloatArray, FloatArray, tuple[int, ...]]:
    npars = len(model.parameter_names)
    values = model.initial.copy()
    values[is_fixed] = fixed_values[is_fixed]
    covariance = np.zeros((npars, npars))
    free = np.flatnonzero(~is_fixed)
    if free.size == 0:
        return values, covariance, ()

    fixed_indices = np.flatnonzero(is_fixed)
    residual_target = data - model.nominal_expected
    if fixed_indices.size:
        residual_target -= model.response[:, fixed_indices] @ (
            values[fixed_indices] - model.initial[fixed_indices]
        )

    design = model.response[:, free]
    sqrt_weight = 1.0 / np.sqrt(variances)
    weighted_design = design * sqrt_weight[:, np.newaxis]
    weighted_target = residual_target * sqrt_weight
    precision = model.constraint_precision[free]
    prior_design = np.diag(np.sqrt(precision))
    prior_target = np.sqrt(precision) * (
        constraint_mean[free] - model.initial[free]
    )
    augmented_design = np.vstack([weighted_design, prior_design])
    augmented_target = np.concatenate([weighted_target, prior_target])
    information = augmented_design.T @ augmented_design

    rank = np.linalg.matrix_rank(information)
    if rank < free.size:
        raise NonIdentifiableModel(
            f"Free model parameters are not identifiable "
            f"(information rank {rank} < {free.size})"
        )

    rhs = augmented_design.T @ augmented_target
    delta = np.linalg.solve(information, rhs)
    delta_lower = lower[free] - model.initial[free]
    delta_upper = upper[free] - model.initial[free]
    if np.any(delta < delta_lower) or np.any(delta > delta_upper):
        bounded = lsq_linear(
            augmented_design,
            augmented_target,
            bounds=(delta_lower, delta_upper),
            method="trf",
            lsmr_tol="auto",
        )
        if not bounded.success:
            raise FitConvergenceError(
                f"Bounded fit failed: {bounded.message}"
            )
        delta = bounded.x

    values[free] = model.initial[free] + delta
    active_local = np.flatnonzero(
        np.isclose(values[free], lower[free], rtol=0.0, atol=1.0e-8)
        | np.isclose(values[free], upper[free], rtol=0.0, atol=1.0e-8)
    )
    inactive_local = np.setdiff1d(
        np.arange(free.size), active_local, assume_unique=True
    )
    if inactive_local.size:
        inactive_information = information[
            np.ix_(inactive_local, inactive_local)
        ]
        inactive_covariance = np.linalg.inv(inactive_information)
        inactive_global = free[inactive_local]
        covariance[np.ix_(inactive_global, inactive_global)] = inactive_covariance
    return values, covariance, tuple(free[active_local].tolist())


def fit_model(
    model: Model,
    data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    *,
    auxdata: Sequence[float] | Mapping[str, float] | None = None,
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    variance: Literal["iterated", "nominal"] = "iterated",
    rtol: float = 1.0e-6,
    max_iter: int = 25,
) -> FitResult:
    """Fit ``model`` using constrained iteratively reweighted least squares."""

    if variance not in {"iterated", "nominal"}:
        raise ValueError("variance must be 'iterated' or 'nominal'")
    if rtol <= 0 or max_iter < 1:
        raise ValueError("rtol must be positive and max_iter must be at least one")
    observed = _data_array(model, data)
    is_fixed, fixed_values, lower, upper = _parameter_options(
        model, fixed, bounds
    )
    constraint_mean = _constraint_means(model, auxdata)
    variances = model.nominal_expected.copy()
    if np.any(variances <= 0):
        raise ModelDomainError("Nominal expected counts must be positive")

    converged = False
    values = model.initial.copy()
    covariance = np.zeros((values.size, values.size))
    active: tuple[int, ...] = ()
    previous_variances: FloatArray | None = None
    damp_updates = False
    for iterations in range(1, max_iter + 1):
        values, covariance, active = _solve_once(
            model,
            observed,
            variances,
            is_fixed,
            fixed_values,
            lower,
            upper,
            constraint_mean,
        )
        expected = model.nominal_expected + model.response @ (
            values - model.initial
        )
        if np.any(expected <= 0) or not np.all(np.isfinite(expected)):
            raise ModelDomainError(
                "Fitted linearized expectations must be finite and positive"
            )
        if variance == "nominal":
            converged = True
            break
        relative_change = np.max(
            np.abs(expected - variances)
            / np.maximum(np.abs(variances), 1.0)
        )
        if relative_change <= rtol:
            converged = True
            variances = expected
            break
        if previous_variances is not None:
            distance_to_previous = np.max(
                np.abs(expected - previous_variances)
                / np.maximum(np.abs(previous_variances), 1.0)
            )
            if distance_to_previous < relative_change:
                damp_updates = True
        previous_variances = variances
        variances = (
            0.5 * (variances + expected) if damp_updates else expected
        )
    if not converged:
        raise FitConvergenceError(
            f"Expected-variance iteration did not converge after {max_iter} steps"
        )

    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(errors, errors)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0,
    )
    pulls = np.full(values.size, np.nan)
    constrained = model.constraint_precision > 0
    pulls[constrained] = (
        values[constrained] - constraint_mean[constrained]
    ) * np.sqrt(model.constraint_precision[constrained])
    chi2 = np.sum(np.square(observed - expected) / variances)
    chi2 += np.sum(
        model.constraint_precision
        * np.square(values - constraint_mean)
    )
    return FitResult(
        parameter_names=model.parameter_names,
        values=values,
        errors=errors,
        covariance=covariance,
        correlation=correlation,
        pulls=pulls,
        expected=expected,
        chi2=float(chi2),
        converged=True,
        iterations=iterations,
        at_bound=tuple(model.parameter_names[index] for index in active),
    )
