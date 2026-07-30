from __future__ import annotations


class SGHFError(Exception):
    """Base class for SGHF failures."""


class SGHFUsageError(SGHFError, AttributeError):
    """Raised when a pyhf-style call should use the SGHF object API."""


class InvalidWorkspace(SGHFError, ValueError):
    """Raised when a workspace cannot be compiled safely."""


class UnsupportedModifier(InvalidWorkspace):
    """Raised for modifier types outside the supported pyhf set."""


class WorkspaceWarning(UserWarning):
    """Warning emitted for accepted but nonstandard workspace content."""


class FitError(SGHFError, RuntimeError):
    """Base class for fit failures."""


class NonIdentifiableModel(FitError):
    """Raised when free parameters are not identifiable."""


class ModelDomainError(FitError):
    """Raised when the linearized model leaves its valid count domain."""


class FitConvergenceError(FitError):
    """Raised when iterated expected variances do not converge."""


class InferenceError(SGHFError, RuntimeError):
    """Base class for hypothesis-test and interval failures."""


class LimitNotFound(InferenceError):
    """Raised when the requested confidence-level crossing is not bracketed."""
