from __future__ import annotations

import numpy as np
import pyhf

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
    assert linear.parameter_names == tuple(exact.config.par_names)
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
    workspace["channels"] = [
        {
            "name": "z_channel",
            "samples": original["samples"],
        },
        {
            "name": "a_channel",
            "samples": original["samples"],
        },
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
