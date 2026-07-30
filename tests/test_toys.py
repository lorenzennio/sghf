from __future__ import annotations

import numpy as np
import pytest

import sghf


def counting_workspace(*, constrained: bool = False) -> dict:
    signal_modifiers = [
        {"name": "mu", "type": "normfactor", "data": None}
    ]
    parameters = []
    if constrained:
        signal_modifiers.append(
            {
                "name": "eff",
                "type": "histosys",
                "data": {"lo_data": [0.0], "hi_data": [20.0]},
            }
        )
        parameters.append({"name": "mu", "fixed": True})
    return {
        "channels": [
            {
                "name": "single",
                "samples": [
                    {
                        "name": "signal",
                        "data": [10.0],
                        "modifiers": signal_modifiers,
                    },
                    {"name": "background", "data": [40.0], "modifiers": []},
                ],
            }
        ],
        "observations": [{"name": "single", "data": [40.0]}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {"poi": "mu", "parameters": parameters},
            }
        ],
        "version": "1.0.0",
    }


def test_poisson_toys_are_reproducible() -> None:
    model = sghf.load(counting_workspace())

    first = model.toys(8, parameters={"mu": 0.0}, rng=1234)
    second = model.toys(8, parameters={"mu": 0.0}, rng=1234)

    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.auxiliary, second.auxiliary)
    assert first.observations.shape == (8, 1)
    assert first.auxiliary.shape == (8, 1)
    np.testing.assert_allclose(first.expected, [40.0])


def test_toys_fluctuate_only_constrained_auxiliary_parameters() -> None:
    model = sghf.load(counting_workspace(constrained=True))
    toys = model.toys(20, rng=7)
    mu = model.parameter_index("mu")
    eff = model.parameter_index("eff")

    np.testing.assert_allclose(toys.auxiliary[:, mu], model.initial[mu])
    assert np.std(toys.auxiliary[:, eff]) > 0.0


def test_fit_accepts_toy_auxiliary_data() -> None:
    model = sghf.load(counting_workspace(constrained=True))
    auxiliary = model.constraint_mean.copy()
    auxiliary[model.parameter_index("eff")] = 1.0

    fit = model.fit(data=[50.0], auxdata=auxiliary, variance="nominal")

    assert fit.value("eff") == pytest.approx(1.0 / 3.0)


def test_fit_accepts_negative_gaussian_pseudo_observations() -> None:
    workspace = counting_workspace()
    workspace["channels"][0]["samples"][1]["data"] = [0.01]
    workspace["observations"][0]["data"] = [0.01]
    model = sghf.load(workspace)
    toys = model.toys(
        8,
        parameters={"mu": 0.0},
        sampling="gaussian",
        rng=1,
    )
    negative = toys.observations[toys.observations < 0]

    assert negative.size > 0
    fit = model.fit(data=[negative[0]], variance="nominal")
    assert fit.converged


def test_toy_hypotest_is_reproducible_and_uses_smoothed_pvalues() -> None:
    model = sghf.load(counting_workspace())
    first = model.hypotest(
        1.0,
        calculator="toys",
        ntoys=30,
        variance="nominal",
        rng=19,
    )
    second = model.hypotest(
        1.0,
        calculator="toys",
        ntoys=30,
        variance="nominal",
        rng=19,
    )

    np.testing.assert_array_equal(first.sb_statistics, second.sb_statistics)
    np.testing.assert_array_equal(first.b_statistics, second.b_statistics)
    assert first.ntoys == 30
    assert 0.0 < first.clsb <= 1.0
    assert 0.0 < first.clb <= 1.0
    assert first.cls == pytest.approx(first.clsb / first.clb)
    assert len(first.expected_cls) == 5


def test_toys_require_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        sghf.load(counting_workspace()).toys(0)
