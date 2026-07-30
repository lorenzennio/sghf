from __future__ import annotations

import numpy as np
import pytest

import sghf


def counting_workspace(
    *,
    observation: float = 60.0,
    signal_modifiers: list[dict] | None = None,
) -> dict:
    return {
        "channels": [
            {
                "name": "single",
                "samples": [
                    {
                        "name": "signal",
                        "data": [10.0],
                        "modifiers": signal_modifiers
                        or [{"name": "mu", "type": "normfactor", "data": None}],
                    },
                    {"name": "background", "data": [40.0], "modifiers": []},
                ],
            }
        ],
        "observations": [{"name": "single", "data": [observation]}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {"poi": "mu", "parameters": []},
            }
        ],
        "version": "1.0.0",
    }


def test_fit_recovers_one_bin_signal_strength_and_iterated_error() -> None:
    fit = sghf.load(counting_workspace()).fit(variance="iterated")

    assert fit.value("mu") == pytest.approx(2.0)
    assert fit.error("mu") == pytest.approx(np.sqrt(60.0) / 10.0)
    np.testing.assert_allclose(fit.expected, [60.0])
    assert fit.chi2 == pytest.approx(0.0)
    assert fit.converged
    assert fit.iterations >= 2


def test_nominal_variance_uses_one_solve() -> None:
    fit = sghf.load(counting_workspace()).fit(variance="nominal")

    assert fit.error("mu") == pytest.approx(np.sqrt(50.0) / 10.0)
    assert fit.iterations == 1


def test_fit_accepts_channel_mapping_data() -> None:
    fit = sghf.load(counting_workspace()).fit(data={"single": [70.0]})

    assert fit["mu"] == pytest.approx(3.0)


def test_fit_honors_call_time_fixed_parameter() -> None:
    fit = sghf.load(counting_workspace()).fit(
        data=[60.0], fixed={"mu": 1.0}, variance="nominal"
    )

    assert fit.value("mu") == 1.0
    assert fit.error("mu") == 0.0
    np.testing.assert_allclose(fit.expected, [50.0])
    assert fit.chi2 == pytest.approx(2.0)


def test_fit_honors_parameter_bound() -> None:
    fit = sghf.load(counting_workspace(observation=100.0)).fit(
        bounds={"mu": (0.0, 3.0)}
    )

    assert fit.value("mu") == pytest.approx(3.0)
    assert fit.error("mu") == 0.0
    assert fit.at_bound == ("mu",)


def test_gaussian_constraint_regularizes_nuisance() -> None:
    workspace = counting_workspace(
        signal_modifiers=[
            {"name": "mu", "type": "normfactor", "data": None},
            {
                "name": "eff",
                "type": "histosys",
                "data": {"lo_data": [0.0], "hi_data": [20.0]},
            },
        ]
    )
    workspace["measurements"][0]["config"]["parameters"] = [
        {"name": "mu", "fixed": True}
    ]
    fit = sghf.load(workspace).fit(variance="nominal")
    expected_shift = 2.0 / 3.0
    expected_error = np.sqrt(1.0 / 3.0)

    assert fit.value("eff") == pytest.approx(expected_shift)
    assert fit.error("eff") == pytest.approx(expected_error)


def test_fit_rejects_nonidentifiable_unconstrained_parameters() -> None:
    workspace = counting_workspace(
        signal_modifiers=[
            {"name": "mu", "type": "normfactor", "data": None},
            {"name": "scale", "type": "normfactor", "data": None},
        ]
    )

    with pytest.raises(sghf.NonIdentifiableModel, match="identifiable"):
        sghf.load(workspace).fit()


def test_iterated_fit_damps_oscillating_variance_updates() -> None:
    model = sghf.Model(
        channel_names=("channel",),
        channel_slices={"channel": slice(0, 2)},
        parameter_names=("mu",),
        poi_name="mu",
        nominal_expected=np.array([36.44146467, 55.10089041]),
        response=np.array([[34.85448381], [-15.85935079]]),
        observations=np.array([112.21122542, 89.25408215]),
        initial=np.array([1.0]),
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([10.0]),
        fixed=np.array([False]),
        constraint_mean=np.array([1.0]),
        constraint_precision=np.array([0.0]),
        modifier_types=("normfactor",),
    )

    fit = model.fit(variance="iterated")

    assert fit.value("mu") == pytest.approx(1.994928, rel=1.0e-5)
    assert fit.iterations <= 25
