"""Fast linear-Gaussian inference for pyhf workspace models."""

from __future__ import annotations

from .exceptions import (
    FitConvergenceError,
    FitError,
    InvalidWorkspace,
    InferenceError,
    LimitNotFound,
    ModelDomainError,
    NonIdentifiableModel,
    SGHFUsageError,
    UnsupportedModifier,
    WorkspaceWarning,
)
from .asymptotic import HypothesisTestResult, LimitResult
from .fit import FitResult
from .model import ApproximationReport, Model
from .toys import (
    ToyDiscoveryTestResult,
    ToyHypothesisTestResult,
    ToyResult,
)
from .workspace import load

__version__ = "0.1.0"


class _PyhfAPIGuidance:
    def __init__(self, path: str):
        self.path = path

    def __getattr__(self, name: str) -> _PyhfAPIGuidance:
        return _PyhfAPIGuidance(f"{self.path}.{name}")

    def __call__(self, *args, **kwargs):
        if self.path == "sghf.Workspace":
            replacement = "model = sghf.load(workspace)"
        elif self.path.endswith(".mle.fit"):
            replacement = "model.fit(...)"
        elif self.path.endswith(".hypotest"):
            replacement = "model.hypotest(...)"
        elif self.path.endswith(".upper_limit"):
            replacement = "model.upper_limit(...)"
        else:
            replacement = "sghf.load(...) and the resulting model methods"
        raise SGHFUsageError(
            f"{self.path}(...) is a pyhf-style call; use {replacement}."
        )


infer = _PyhfAPIGuidance("sghf.infer")


def __getattr__(name: str):
    if name == "Workspace":
        return _PyhfAPIGuidance("sghf.Workspace")
    raise AttributeError(f"module 'sghf' has no attribute {name!r}")


__all__ = [
    "ApproximationReport",
    "InvalidWorkspace",
    "HypothesisTestResult",
    "InferenceError",
    "LimitNotFound",
    "LimitResult",
    "FitConvergenceError",
    "FitError",
    "FitResult",
    "Model",
    "ModelDomainError",
    "NonIdentifiableModel",
    "SGHFUsageError",
    "UnsupportedModifier",
    "ToyDiscoveryTestResult",
    "ToyHypothesisTestResult",
    "ToyResult",
    "WorkspaceWarning",
    "load",
]
