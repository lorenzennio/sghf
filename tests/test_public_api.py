from __future__ import annotations

import pytest

import sghf

from test_loading import one_bin_workspace


def test_package_exposes_version_and_approximation_report() -> None:
    model = sghf.load(one_bin_workspace())

    assert sghf.__version__ == "0.1.0"
    assert sghf.ToyDiscoveryTestResult.__name__ == "ToyDiscoveryTestResult"
    assert model.approximation.modifier_types == ("normfactor",)
    assert model.approximation.variance_default == "iterated"
    assert model.approximation.parameter_names == ("mu",)


def test_pyhf_workspace_style_call_points_to_load() -> None:
    with pytest.raises(sghf.SGHFUsageError, match=r"sghf\.load"):
        sghf.Workspace(one_bin_workspace())


def test_pyhf_fit_style_call_points_to_model_fit() -> None:
    with pytest.raises(sghf.SGHFUsageError, match=r"model\.fit"):
        sghf.infer.mle.fit()


def test_pyhf_hypotest_style_call_points_to_model_hypotest() -> None:
    with pytest.raises(sghf.SGHFUsageError, match=r"model\.hypotest"):
        sghf.infer.hypotest()
