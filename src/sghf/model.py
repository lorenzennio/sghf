from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

if TYPE_CHECKING:
    from .fit import FitResult


@dataclass(frozen=True)
class ApproximationReport:
    """Description of the approximation compiled for a workspace."""

    modifier_types: tuple[str, ...]
    parameter_names: tuple[str, ...]
    variance_default: Literal["nominal"] = "nominal"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Model:
    """A pyhf workspace compiled into a linear-Gaussian model."""

    channel_names: tuple[str, ...]
    channel_slices: dict[str, slice]
    parameter_names: tuple[str, ...]
    poi_name: str
    nominal_expected: FloatArray
    response: FloatArray
    observations: FloatArray
    initial: FloatArray
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    fixed: NDArray[np.bool_]
    constraint_mean: FloatArray
    constraint_precision: FloatArray
    modifier_types: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def approximation(self) -> ApproximationReport:
        """Return a concise description of the compiled approximation."""

        return ApproximationReport(
            modifier_types=self.modifier_types,
            parameter_names=self.parameter_names,
            warnings=self.warnings,
        )

    def parameter_index(self, name: str) -> int:
        """Return the scalar parameter index for ``name``."""

        try:
            return self.parameter_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown parameter {name!r}") from error

    def response_for(self, name: str) -> FloatArray:
        """Return the expected-yield response for a scalar parameter."""

        return self.response[:, self.parameter_index(name)].copy()

    def expected(
        self,
        parameters: Sequence[float] | Mapping[str, float] | FitResult,
    ) -> FloatArray:
        """Evaluate the compiled linear model at ``parameters``."""

        if hasattr(parameters, "values") and not isinstance(parameters, Mapping):
            values = np.asarray(parameters.values, dtype=float)
        elif isinstance(parameters, Mapping):
            values = self.initial.copy()
            for name, value in parameters.items():
                values[self.parameter_index(name)] = float(value)
        else:
            values = np.asarray(parameters, dtype=float)
        if values.shape != self.initial.shape:
            raise ValueError(
                f"Parameters must contain {self.initial.size} value(s)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Parameter values must be finite")
        result = self.nominal_expected + self.response @ (
            values - self.initial
        )
        if not np.all(np.isfinite(result)) or np.any(result <= 0):
            from .exceptions import ModelDomainError

            raise ModelDomainError(
                "Linearized expectations must be finite and positive"
            )
        return result

    def fit(
        self,
        data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        *,
        auxdata: Sequence[float] | Mapping[str, float] | None = None,
        fixed: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        variance: Literal["iterated", "nominal"] = "nominal",
        rtol: float = 1.0e-6,
        max_iter: int = 25,
    ):
        """Fit the linear-Gaussian model to observations."""

        from .fit import fit_model

        return fit_model(
            self,
            data=data,
            auxdata=auxdata,
            fixed=fixed,
            bounds=bounds,
            variance=variance,
            rtol=rtol,
            max_iter=max_iter,
        )

    def hypotest(
        self,
        poi_test: float,
        data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        *,
        test_stat: Literal["q0", "q", "qtilde"] = "qtilde",
        calculator: Literal["asymptotic", "toys"] = "asymptotic",
        variance: Literal["iterated", "nominal"] = "nominal",
        ntoys: int = 10_000,
        batch_size: int | None = None,
        sampling: Literal["poisson", "gaussian"] = "poisson",
        rng: int | np.random.Generator | None = None,
    ):
        """Run an asymptotic or toy-based profile test for ``poi_test``."""

        if calculator == "asymptotic":
            from .asymptotic import hypotest

            return hypotest(
                self,
                poi_test,
                data=data,
                test_stat=test_stat,
                variance=variance,
            )
        if calculator == "toys":
            from .toys import toy_hypotest

            return toy_hypotest(
                self,
                poi_test,
                data=data,
                test_stat=test_stat,
                variance=variance,
                ntoys=ntoys,
                batch_size=batch_size,
                sampling=sampling,
                rng=rng,
            )
        raise ValueError("calculator must be 'asymptotic' or 'toys'")

    def toys(
        self,
        size: int,
        *,
        parameters: Sequence[float] | Mapping[str, float] | FitResult | None = None,
        sampling: Literal["poisson", "gaussian"] = "poisson",
        fluctuate_auxiliary: bool = True,
        rng: int | np.random.Generator | None = None,
    ):
        """Generate main and auxiliary pseudo-observations."""

        from .toys import generate_toys

        return generate_toys(
            self,
            size,
            parameters=parameters,
            sampling=sampling,
            fluctuate_auxiliary=fluctuate_auxiliary,
            rng=rng,
        )

    def upper_limit(
        self,
        data: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        *,
        level: float = 0.05,
        test_stat: Literal["q", "qtilde"] = "qtilde",
        variance: Literal["iterated", "nominal"] = "nominal",
        bracket: tuple[float, float] | None = None,
    ):
        """Calculate observed and expected asymptotic upper limits."""

        from .asymptotic import upper_limit

        return upper_limit(
            self,
            data=data,
            level=level,
            test_stat=test_stat,
            variance=variance,
            bracket=bracket,
        )
