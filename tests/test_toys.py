from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, poisson

import sghf
import sghf.toys as toy_module


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
    assert first.variance == "nominal"
    assert first.q == pytest.approx(
        model.hypotest(1.0, test_stat="qtilde", variance="nominal").q
    )


def test_toy_q0_returns_empirical_discovery_result() -> None:
    workspace = counting_workspace()
    workspace["observations"][0]["data"] = [45.0]
    model = sghf.load(workspace)

    result = model.hypotest(
        0.0,
        calculator="toys",
        test_stat="q0",
        ntoys=20_000,
        rng=23,
    )

    assert isinstance(result, sghf.ToyDiscoveryTestResult)
    assert result.test_stat == "q0"
    assert result.variance == "nominal"
    assert result.q == pytest.approx(
        model.hypotest(0.0, test_stat="q0", variance="nominal").q
    )
    expected_p0 = (
        np.count_nonzero(result.null_statistics >= result.q) + 1
    ) / (result.ntoys + 1)
    assert result.p0 == pytest.approx(expected_p0)
    assert result.p0 == pytest.approx(poisson.sf(44, 40), abs=0.015)
    assert result.significance == pytest.approx(norm.isf(result.p0))
    assert len(result.expected_p0) == 5
    assert len(result.expected_significance) == 5
    assert result.null_statistics.shape == (20_000,)
    assert result.alternative_statistics.shape == (20_000,)


def test_toy_q0_vectorizes_profile_statistics(monkeypatch) -> None:
    model = sghf.load(counting_workspace())
    calls = 0
    original = toy_module._profile_statistic

    def counted_profile_statistic(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        toy_module,
        "_profile_statistic",
        counted_profile_statistic,
    )

    model.hypotest(
        0.0,
        calculator="toys",
        test_stat="q0",
        ntoys=1_000,
        rng=5,
    )

    assert calls <= 1


def test_toy_q0_generates_statistics_in_batches(monkeypatch) -> None:
    model = sghf.load(counting_workspace())
    generated_sizes = []
    original = toy_module.generate_toys

    def counted_generate_toys(model, size, **kwargs):
        generated_sizes.append(size)
        return original(model, size, **kwargs)

    monkeypatch.setattr(
        toy_module,
        "generate_toys",
        counted_generate_toys,
    )

    result = model.hypotest(
        0.0,
        calculator="toys",
        test_stat="q0",
        ntoys=53,
        batch_size=11,
        rng=5,
    )

    assert result.batch_size == 11
    assert generated_sizes == [11, 11, 11, 11, 9] * 2


@pytest.mark.parametrize(
    ("test_stat", "poi_test"),
    [("q0", 0.0), ("q", 1.0), ("qtilde", 1.0)],
)
def test_vectorized_nominal_statistics_match_profile_fits(
    test_stat: str,
    poi_test: float,
) -> None:
    model = sghf.load(counting_workspace())
    observations = np.array([[30.0], [40.0], [45.0], [60.0]])
    auxiliary = np.broadcast_to(
        model.constraint_mean,
        (observations.shape[0], model.initial.size),
    )

    vectorized = toy_module._nominal_test_statistics(
        model,
        poi_test,
        observations,
        auxiliary,
        test_stat=test_stat,
    )
    profiled = np.array(
        [
            toy_module._profile_statistic(
                model,
                poi_test,
                observation,
                auxdata=auxdata,
                test_stat=test_stat,
                variance="nominal",
            )[0]
            for observation, auxdata in zip(observations, auxiliary)
        ]
    )

    np.testing.assert_allclose(vectorized, profiled, atol=1.0e-12)


def test_toys_require_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        sghf.load(counting_workspace()).toys(0)


def test_both_calculators_share_the_default_variance() -> None:
    workspace = counting_workspace()
    workspace["observations"][0]["data"] = [70.0]
    model = sghf.load(workspace)

    asymptotic = model.hypotest(4.0)
    toys = model.hypotest(4.0, calculator="toys", ntoys=500, rng=17)

    assert asymptotic.q == pytest.approx(toys.q)
    assert toys.variance == "nominal"
