from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import sghf


def counting_workspace(*, observation: float = 40.0) -> dict:
    return {
        "channels": [
            {
                "name": "single",
                "samples": [
                    {
                        "name": "signal",
                        "data": [10.0],
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
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


def test_expected_accepts_parameter_mapping_and_fit_result() -> None:
    model = sghf.load(counting_workspace())
    fit = model.fit(variance="nominal")

    np.testing.assert_allclose(model.expected({"mu": 2.0}), [60.0])
    np.testing.assert_allclose(model.expected(fit), fit.expected)


@pytest.mark.parametrize("test_stat", ["q", "qtilde"])
def test_upper_limit_statistics_for_background_asimov_data(
    test_stat: str,
) -> None:
    result = sghf.load(counting_workspace()).hypotest(
        1.0, test_stat=test_stat, variance="nominal"
    )

    assert result.q == pytest.approx(2.0)
    assert result.q_asimov == pytest.approx(2.0)
    assert result.clsb == pytest.approx(norm.cdf(-np.sqrt(2.0)))
    assert result.clb == pytest.approx(0.5)
    assert result.cls == pytest.approx(2.0 * norm.cdf(-np.sqrt(2.0)))
    assert len(result.expected_cls) == 5


def test_q0_discovery_statistic() -> None:
    result = sghf.load(counting_workspace(observation=60.0)).hypotest(
        0.0, test_stat="q0", variance="nominal"
    )

    assert result.q == pytest.approx(8.0)
    assert result.significance == pytest.approx(np.sqrt(8.0))


def test_qtilde_uses_bounded_denominator_for_background_deficit() -> None:
    result = sghf.load(counting_workspace(observation=30.0)).hypotest(
        1.0, test_stat="qtilde", variance="nominal"
    )

    assert result.free_fit.value("mu") == pytest.approx(0.0)
    assert result.free_fit.at_bound == ("mu",)
    assert result.q == pytest.approx(6.0)


def test_asymptotic_upper_limit_matches_one_bin_gaussian_result() -> None:
    result = sghf.load(counting_workspace()).upper_limit(
        variance="nominal"
    )
    expected_median = norm.isf(0.025) * np.sqrt(50.0) / 10.0

    assert result.observed == pytest.approx(expected_median, rel=1.0e-7)
    assert result.expected[2] == pytest.approx(expected_median, rel=1.0e-7)
    assert result.level == 0.05


def test_q0_requires_zero_test_value() -> None:
    with pytest.raises(ValueError, match="q0"):
        sghf.load(counting_workspace()).hypotest(1.0, test_stat="q0")


def test_upper_limit_expands_past_suggested_poi_bound() -> None:
    workspace = counting_workspace()
    workspace["measurements"][0]["config"]["parameters"] = [
        {"name": "mu", "inits": [0.5], "bounds": [[0.0, 0.75]]}
    ]

    result = sghf.load(workspace).upper_limit(variance="nominal")

    assert result.observed > 0.75
    assert result.bracket[1] > 0.75
