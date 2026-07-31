from __future__ import annotations

import numpy as np
import pyhf
import pytest

import sghf


def parity_workspace() -> dict:
    return {
        "channels": [
            {
                "name": "channel",
                "samples": [
                    {
                        "name": "signal",
                        "data": [10.0, 20.0],
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None},
                            {
                                "name": "eff",
                                "type": "normsys",
                                "data": {"lo": 0.9, "hi": 1.1},
                            },
                            {
                                "name": "shape",
                                "type": "histosys",
                                "data": {
                                    "lo_data": [8.0, 22.0],
                                    "hi_data": [12.0, 18.0],
                                },
                            },
                        ],
                    },
                    {
                        "name": "background",
                        "data": [40.0, 60.0],
                        "modifiers": [
                            {
                                "name": "stat",
                                "type": "staterror",
                                "data": [4.0, 6.0],
                            },
                            {"name": "lumi", "type": "lumi", "data": None},
                        ],
                    },
                    {
                        "name": "minor",
                        "data": [7.0, 9.0],
                        "modifiers": [
                            {
                                "name": "minor_shape",
                                "type": "shapesys",
                                "data": [2.0, 3.0],
                            },
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "channel", "data": [50.0, 80.0]}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {
                            "name": "lumi",
                            "inits": [1.0],
                            "bounds": [[0.0, 10.0]],
                            "auxdata": [1.0],
                            "sigmas": [0.02],
                        }
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def test_compiler_matches_pyhf_nominal_and_local_slopes() -> None:
    workspace = parity_workspace()
    linear = sghf.load(workspace)
    exact_workspace = pyhf.Workspace(workspace)
    exact = exact_workspace.model()
    initial = np.asarray(exact.config.suggested_init(), dtype=float)

    assert linear.channel_names == tuple(exact.config.channels)
    # Only the set is asserted, not the order. Both group scalars into the same
    # per-modifier-type blocks, but SGHF sorts each block by name while pyhf
    # does not, so the two orders coincide only when a workspace happens to
    # declare its modifiers alphabetically -- as this one does. On a
    # 232-parameter production workspace the histosys blocks hold the same 39
    # names in different orders. Nothing depends on the order: SGHF addresses
    # every parameter by name through Model.parameter_index.
    assert set(linear.parameter_names) == set(exact.config.par_names)
    exact_nominal = np.asarray(
        exact.expected_data(initial, include_auxdata=False), dtype=float
    )
    np.testing.assert_allclose(linear.nominal_expected, exact_nominal)

    epsilon = 1.0e-5
    for parameter_name in linear.parameter_names:
        exact_index = exact.config.par_names.index(parameter_name)
        high = initial.copy()
        low = initial.copy()
        high[exact_index] += epsilon
        low[exact_index] -= epsilon
        exact_slope = (
            np.asarray(exact.expected_data(high, include_auxdata=False))
            - np.asarray(exact.expected_data(low, include_auxdata=False))
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            linear.response_for(parameter_name),
            exact_slope,
            rtol=2.0e-5,
            atol=1.0e-7,
            err_msg=parameter_name,
        )


def test_compiler_uses_pyhf_channel_order() -> None:
    workspace = parity_workspace()
    original = workspace["channels"][0]
    # A shapesys parameter is per-channel and cannot be shared, so only the
    # samples whose modifiers may legitimately appear in both are duplicated.
    shared = [
        sample
        for sample in original["samples"]
        if all(
            modifier["type"] != "shapesys" for modifier in sample["modifiers"]
        )
    ]
    workspace["channels"] = [
        {"name": "z_channel", "samples": shared},
        {"name": "a_channel", "samples": shared},
    ]
    workspace["observations"] = [
        {"name": "z_channel", "data": [51.0, 81.0]},
        {"name": "a_channel", "data": [49.0, 79.0]},
    ]
    linear = sghf.load(workspace)
    exact_workspace = pyhf.Workspace(workspace)
    exact = exact_workspace.model()

    assert linear.channel_names == ("a_channel", "z_channel")
    np.testing.assert_allclose(
        linear.nominal_expected,
        exact.expected_data(
            exact.config.suggested_init(), include_auxdata=False
        ),
    )
    np.testing.assert_allclose(
        linear.observations,
        exact_workspace.data(exact)[: exact.config.nmaindata],
    )


def _pyhf_parameter_configuration(exact) -> dict[str, dict[str, float]]:
    """Read pyhf's per-scalar parameter configuration, keyed by SGHF name.

    SGHF stores the Gaussian constraint in parameter space. pyhf stores
    ``auxdata`` in the space of the constraint term, which coincides with
    parameter space for its normal constraints but not for the Poisson
    constraint it uses for ``shapesys``: there the auxiliary datum is the
    expected count ``factor``, and the parameter it multiplies is centred on
    one. Dividing by ``factor`` (equivalently, multiplying by the squared
    width) maps it back.
    """

    config = exact.config
    initial = np.asarray(config.suggested_init(), dtype=float)
    bounds = np.asarray(config.suggested_bounds(), dtype=float)
    fixed = np.asarray(config.suggested_fixed(), dtype=bool)

    expected: dict[str, dict[str, float]] = {}
    for name in config.par_order:
        parameter = config.param_set(name)
        indices = range(*config.par_slice(name).indices(initial.size))
        constrained = getattr(parameter, "constrained", False)
        if constrained:
            width = np.asarray(parameter.width(), dtype=float)
            precision = 1.0 / np.square(width)
            auxdata = np.asarray(parameter.auxdata, dtype=float)
            mean = (
                auxdata
                if parameter.pdf_type == "normal"
                else auxdata * np.square(width)
            )
        for offset, index in enumerate(indices):
            scalar = name if parameter.n_parameters == 1 else f"{name}[{offset}]"
            expected[scalar] = {
                "initial": float(initial[index]),
                "lower": float(bounds[index, 0]),
                "upper": float(bounds[index, 1]),
                "fixed": bool(fixed[index]),
                "mean": float(mean[offset]) if constrained else None,
                "precision": float(precision[offset]) if constrained else 0.0,
            }
    return expected


def test_compiler_matches_pyhf_parameter_configuration() -> None:
    """Guard the compiled parameter block against drift from pyhf.

    The nominal yields and response slopes are covered elsewhere; this pins
    the rest of the compiled model -- initial values, bounds, fixed flags and
    the Gaussian constraint -- against what pyhf reports for the same
    workspace, across every modifier type SGHF supports.
    """

    workspace = parity_workspace()
    linear = sghf.load(workspace)
    exact = pyhf.Workspace(workspace).model()
    expected = _pyhf_parameter_configuration(exact)

    assert set(linear.modifier_types) == {
        "histosys",
        "lumi",
        "normfactor",
        "normsys",
        "shapesys",
        "staterror",
    }
    assert set(linear.parameter_names) == set(expected)

    for scalar, reference in expected.items():
        index = linear.parameter_index(scalar)
        assert linear.initial[index] == pytest.approx(
            reference["initial"]
        ), scalar
        assert linear.lower_bounds[index] == pytest.approx(
            reference["lower"]
        ), scalar
        assert linear.upper_bounds[index] == pytest.approx(
            reference["upper"]
        ), scalar
        assert bool(linear.fixed[index]) == reference["fixed"], scalar
        assert linear.constraint_precision[index] == pytest.approx(
            reference["precision"]
        ), scalar
        if reference["mean"] is not None:
            assert linear.constraint_mean[index] == pytest.approx(
                reference["mean"]
            ), scalar
