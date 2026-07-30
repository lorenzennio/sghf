from __future__ import annotations

import json

import numpy as np
import pytest

import sghf


def one_bin_workspace() -> dict:
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
                    {
                        "name": "background",
                        "data": [40.0],
                        "modifiers": [],
                    },
                ],
            }
        ],
        "observations": [{"name": "single", "data": [50.0]}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {"poi": "mu", "parameters": []},
            }
        ],
        "version": "1.0.0",
    }


def test_load_compiles_one_bin_normfactor() -> None:
    model = sghf.load(one_bin_workspace())

    assert model.channel_names == ("single",)
    assert model.parameter_names == ("mu",)
    assert model.poi_name == "mu"
    np.testing.assert_allclose(model.nominal_expected, [50.0])
    np.testing.assert_allclose(model.response, [[10.0]])
    np.testing.assert_allclose(model.observations, [50.0])


def modifier_workspace() -> dict:
    workspace = {
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
                        ],
                    },
                    {
                        "name": "shape",
                        "data": [30.0, 50.0],
                        "modifiers": [
                            {
                                "name": "shape",
                                "type": "histosys",
                                "data": {
                                    "lo_data": [24.0, 55.0],
                                    "hi_data": [36.0, 45.0],
                                },
                            }
                        ],
                    },
                    {
                        "name": "stat",
                        "data": [40.0, 60.0],
                        "modifiers": [
                            {
                                "name": "stat",
                                "type": "staterror",
                                "data": [4.0, 12.0],
                            }
                        ],
                    },
                    {
                        "name": "shape_unc",
                        "data": [20.0, 40.0],
                        "modifiers": [
                            {
                                "name": "shape_unc",
                                "type": "shapesys",
                                "data": [2.0, 4.0],
                            }
                        ],
                    },
                    {
                        "name": "lumi_sample",
                        "data": [5.0, 5.0],
                        "modifiers": [
                            {"name": "lumi", "type": "lumi", "data": None}
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "channel", "data": [115.0, 175.0]}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {
                            "name": "mu",
                            "inits": [2.0],
                            "bounds": [[0.0, 5.0]],
                        },
                        {
                            "name": "lumi",
                            "auxdata": [1.0],
                            "sigmas": [0.02],
                        },
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }
    return workspace


def test_load_compiles_all_standard_modifiers() -> None:
    model = sghf.load(modifier_workspace())

    np.testing.assert_allclose(model.nominal_expected, [115.0, 195.0])
    np.testing.assert_allclose(model.response_for("mu"), [10.0, 20.0])
    np.testing.assert_allclose(
        model.response_for("eff"), [2.0029275573005556, 4.005855114601111]
    )
    np.testing.assert_allclose(model.response_for("shape"), [6.0, -5.0])
    np.testing.assert_allclose(model.response_for("stat[0]"), [40.0, 0.0])
    np.testing.assert_allclose(model.response_for("stat[1]"), [0.0, 60.0])
    np.testing.assert_allclose(
        model.response_for("shape_unc[0]"), [20.0, 0.0]
    )
    np.testing.assert_allclose(model.response_for("lumi"), [5.0, 5.0])
    assert model.initial[model.parameter_index("mu")] == 2.0
    assert model.lower_bounds[model.parameter_index("mu")] == 0.0
    assert model.upper_bounds[model.parameter_index("mu")] == 5.0
    assert model.constraint_precision[model.parameter_index("eff")] == 1.0
    assert model.constraint_precision[model.parameter_index("stat[0]")] == 100.0
    assert model.constraint_precision[model.parameter_index("stat[1]")] == 25.0
    assert model.constraint_precision[model.parameter_index("lumi")] == 2500.0


def test_load_reads_path_and_matches_observations_by_channel(tmp_path) -> None:
    workspace = one_bin_workspace()
    workspace["metadata"] = {"analysis": "example"}
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(workspace))

    with pytest.warns(sghf.WorkspaceWarning, match="metadata"):
        model = sghf.load(path)

    np.testing.assert_allclose(model.observations, [50.0])
    assert any("metadata" in warning for warning in model.warnings)


def test_strict_loading_rejects_unknown_top_level_metadata() -> None:
    workspace = one_bin_workspace()
    workspace["content"] = []

    with pytest.raises(sghf.InvalidWorkspace, match="content"):
        sghf.load(workspace, strict=True)


def test_load_rejects_unknown_modifier() -> None:
    workspace = one_bin_workspace()
    workspace["channels"][0]["samples"][0]["modifiers"][0]["type"] = "custom"

    with pytest.raises(sghf.UnsupportedModifier, match="custom"):
        sghf.load(workspace)


def test_load_requires_measurement_name_when_requested() -> None:
    with pytest.raises(sghf.InvalidWorkspace, match="Missing"):
        sghf.load(one_bin_workspace(), measurement="Missing")
