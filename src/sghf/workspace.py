from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import InvalidWorkspace, UnsupportedModifier, WorkspaceWarning
from .model import Model


WorkspaceSource = Mapping[str, Any] | str | PathLike[str] | IO[str]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_TOP_LEVEL_KEYS = {"channels", "observations", "measurements", "version"}
_NORMAL_TYPES = {"normsys", "histosys"}
_VECTOR_TYPES = {"staterror", "shapesys"}
# Compilation order of the supported modifiers; the keys are the whitelist.
_MODIFIER_ORDER = {
    "histosys": 0,
    "lumi": 1,
    "normfactor": 2,
    "normsys": 3,
    "shapesys": 4,
    "staterror": 5,
}
_SUPPORTED_MODIFIERS = _MODIFIER_ORDER.keys()


@dataclass
class _ParameterGroup:
    name: str
    modifier_type: str
    size: int
    initial: FloatArray
    lower: FloatArray
    upper: FloatArray
    fixed: BoolArray
    mean: FloatArray
    precision: FloatArray
    response: FloatArray | None = None
    uncertainty_sq: FloatArray | None = None
    affected_nominal: FloatArray | None = None

    @property
    def scalar_names(self) -> tuple[str, ...]:
        if self.size == 1:
            return (self.name,)
        return tuple(f"{self.name}[{index}]" for index in range(self.size))


def _read_source(source: WorkspaceSource) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    try:
        if hasattr(source, "read"):
            loaded = json.load(source)
        else:
            with open(source, encoding="utf-8") as stream:
                loaded = json.load(stream)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise InvalidWorkspace(f"Could not read workspace: {error}") from error
    if not isinstance(loaded, dict):
        raise InvalidWorkspace("Workspace JSON must contain an object")
    return loaded


def _as_float_array(
    value: Sequence[float] | None,
    *,
    size: int,
    default: float,
    field: str,
    parameter: str,
) -> FloatArray:
    if value is None:
        return np.full(size, default, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise InvalidWorkspace(
            f"Parameter {parameter!r} field {field!r} must contain {size} value(s)"
        )
    if not np.all(np.isfinite(array)):
        raise InvalidWorkspace(
            f"Parameter {parameter!r} field {field!r} must be finite"
        )
    return array


def _new_group(name: str, modifier_type: str, size: int) -> _ParameterGroup:
    if modifier_type == "normfactor":
        centre, lower, upper, precision = 1.0, 0.0, 10.0, 0.0
    elif modifier_type in _NORMAL_TYPES:
        centre, lower, upper, precision = 0.0, -5.0, 5.0, 1.0
    else:
        centre, lower, upper, precision = 1.0, 1.0e-10, 10.0, 0.0
    vector = np.full(size, centre)
    return _ParameterGroup(
        name=name,
        modifier_type=modifier_type,
        size=size,
        initial=vector.copy(),
        lower=np.full(size, lower),
        upper=np.full(size, upper),
        fixed=np.zeros(size, dtype=bool),
        mean=vector.copy(),
        precision=np.full(size, precision),
        uncertainty_sq=np.zeros(size) if modifier_type in _VECTOR_TYPES else None,
        affected_nominal=np.zeros(size) if modifier_type in _VECTOR_TYPES else None,
    )


def _compatible(existing: str, incoming: str) -> bool:
    return existing == incoming or (
        existing in _NORMAL_TYPES and incoming in _NORMAL_TYPES
    )


def _normsys_code4_slope(low: float, high: float) -> float:
    """Return the derivative at zero of pyhf's default code4 interpolator."""

    if low <= 0 or high <= 0:
        raise InvalidWorkspace("normsys lo and hi factors must be positive")
    b0 = high - 1.0
    b1 = low - 1.0
    b2 = np.log(high) * high
    b3 = -np.log(low) * low
    b4 = np.square(np.log(high)) * high
    b5 = np.square(np.log(low)) * low
    return float(
        (15.0 / 16.0) * b0
        - (15.0 / 16.0) * b1
        - (7.0 / 16.0) * b2
        - (7.0 / 16.0) * b3
        + (1.0 / 16.0) * b4
        - (1.0 / 16.0) * b5
    )


def _measurement(
    workspace: dict[str, Any],
    requested: str | None,
    warning_messages: list[str],
) -> dict[str, Any]:
    measurements = workspace.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise InvalidWorkspace("Workspace requires at least one measurement")
    if requested is not None:
        for candidate in measurements:
            if candidate.get("name") == requested:
                return candidate
        raise InvalidWorkspace(f"Measurement {requested!r} was not found")
    if len(measurements) > 1:
        message = (
            f"Workspace defines {len(measurements)} measurements; using "
            f"{measurements[0].get('name')!r}. Pass measurement=... to select one."
        )
        warning_messages.append(message)
        warnings.warn(message, WorkspaceWarning, stacklevel=3)
    return measurements[0]


def _discover_groups(
    channels: list[dict[str, Any]],
) -> tuple[list[_ParameterGroup], dict[str, _ParameterGroup]]:
    groups: list[_ParameterGroup] = []
    by_name: dict[str, _ParameterGroup] = {}
    for channel in channels:
        samples = channel.get("samples")
        if not isinstance(samples, list) or not samples:
            raise InvalidWorkspace(
                f"Channel {channel.get('name')!r} requires at least one sample"
            )
        nbins = len(samples[0].get("data", []))
        if nbins == 0:
            raise InvalidWorkspace(f"Channel {channel.get('name')!r} has no bins")
        for sample in samples:
            if len(sample.get("data", [])) != nbins:
                raise InvalidWorkspace(
                    f"Sample {sample.get('name')!r} in channel "
                    f"{channel.get('name')!r} has an inconsistent bin count"
                )
            for modifier in sample.get("modifiers", []):
                modifier_type = modifier.get("type")
                name = modifier.get("name")
                if modifier_type not in _SUPPORTED_MODIFIERS:
                    raise UnsupportedModifier(
                        f"Unsupported modifier type {modifier_type!r} for {name!r}"
                    )
                if not isinstance(name, str) or not name:
                    raise InvalidWorkspace("Every modifier requires a nonempty name")
                size = nbins if modifier_type in _VECTOR_TYPES else 1
                existing = by_name.get(name)
                if existing is None:
                    existing = _new_group(name, modifier_type, size)
                    by_name[name] = existing
                    groups.append(existing)
                elif not _compatible(existing.modifier_type, modifier_type):
                    raise InvalidWorkspace(
                        f"Modifier {name!r} reuses incompatible types "
                        f"{existing.modifier_type!r} and {modifier_type!r}"
                    )
                elif existing.size != size:
                    raise InvalidWorkspace(
                        f"Modifier {name!r} has inconsistent parameter dimensions"
                    )
                elif modifier_type == "shapesys":
                    raise InvalidWorkspace(
                        f"shapesys modifier {name!r} cannot be shared"
                    )
    groups.sort(key=lambda group: (_MODIFIER_ORDER[group.modifier_type], group.name))
    return groups, by_name


def _apply_measurement_settings(
    selected: dict[str, Any],
    groups: dict[str, _ParameterGroup],
) -> str:
    config = selected.get("config")
    if not isinstance(config, dict):
        raise InvalidWorkspace("Measurement requires a config object")
    poi_name = config.get("poi")
    if poi_name not in groups:
        raise InvalidWorkspace(f"Measurement POI {poi_name!r} is not a model parameter")
    for setting in config.get("parameters", []):
        name = setting.get("name")
        if name not in groups:
            raise InvalidWorkspace(
                f"Measurement config references unknown parameter {name!r}"
            )
        group = groups[name]
        group.initial = _as_float_array(
            setting.get("inits"),
            size=group.size,
            default=float(group.initial[0]),
            field="inits",
            parameter=name,
        )
        bounds = setting.get("bounds")
        if bounds is not None:
            bounds_array = np.asarray(bounds, dtype=float)
            if bounds_array.shape != (group.size, 2):
                raise InvalidWorkspace(
                    f"Parameter {name!r} bounds must contain {group.size} pair(s)"
                )
            group.lower = bounds_array[:, 0]
            group.upper = bounds_array[:, 1]
        if np.any(group.lower >= group.upper):
            raise InvalidWorkspace(f"Parameter {name!r} has invalid bounds")
        if np.any(group.initial < group.lower) or np.any(group.initial > group.upper):
            raise InvalidWorkspace(
                f"Parameter {name!r} initial value lies outside its bounds"
            )
        if "fixed" in setting:
            group.fixed[:] = bool(setting["fixed"])
        if "auxdata" in setting:
            group.mean = _as_float_array(
                setting["auxdata"],
                size=group.size,
                default=float(group.mean[0]),
                field="auxdata",
                parameter=name,
            )
        if "sigmas" in setting:
            sigmas = _as_float_array(
                setting["sigmas"],
                size=group.size,
                default=1.0,
                field="sigmas",
                parameter=name,
            )
            if np.any(sigmas <= 0):
                raise InvalidWorkspace(
                    f"Parameter {name!r} sigmas must be positive"
                )
            group.precision = 1.0 / np.square(sigmas)
    return str(poi_name)


def _channel_layout(
    channels: list[dict[str, Any]],
) -> tuple[tuple[str, ...], dict[str, slice], int]:
    names: list[str] = []
    slices: dict[str, slice] = {}
    offset = 0
    for channel in channels:
        name = channel.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidWorkspace("Every channel requires a nonempty name")
        if name in slices:
            raise InvalidWorkspace(f"Duplicate channel name {name!r}")
        nbins = len(channel["samples"][0]["data"])
        names.append(name)
        slices[name] = slice(offset, offset + nbins)
        offset += nbins
    return tuple(names), slices, offset


def _compile_sample(
    sample: dict[str, Any],
    channel_slice: slice,
    group_slices: dict[str, slice],
    groups: dict[str, _ParameterGroup],
    nominal_expected: FloatArray,
    response: FloatArray,
) -> None:
    raw_nominal = np.asarray(sample["data"], dtype=float)
    if np.any(raw_nominal < 0) or not np.all(np.isfinite(raw_nominal)):
        raise InvalidWorkspace(
            f"Sample {sample.get('name')!r} yields must be finite and nonnegative"
        )
    template = raw_nominal.copy()
    scale_modifiers: list[tuple[dict[str, Any], FloatArray, FloatArray]] = []
    histosys: list[tuple[dict[str, Any], FloatArray]] = []

    for modifier in sample.get("modifiers", []):
        modifier_type = modifier["type"]
        group = groups[modifier["name"]]
        if modifier_type == "histosys":
            data = modifier.get("data") or {}
            low = np.asarray(data.get("lo_data"), dtype=float)
            high = np.asarray(data.get("hi_data"), dtype=float)
            if low.shape != raw_nominal.shape or high.shape != raw_nominal.shape:
                raise InvalidWorkspace(
                    f"histosys modifier {group.name!r} must match sample bins"
                )
            delta = 0.5 * (high - low)
            template += delta * group.initial[0]
            histosys.append((modifier, delta))
        elif modifier_type == "normfactor":
            scale_modifiers.append(
                (modifier, group.initial.copy(), np.ones(group.size))
            )
        elif modifier_type == "normsys":
            data = modifier.get("data") or {}
            try:
                slope = _normsys_code4_slope(
                    float(data["lo"]), float(data["hi"])
                )
            except (KeyError, TypeError, ValueError) as error:
                raise InvalidWorkspace(
                    f"normsys modifier {group.name!r} requires numeric lo and hi"
                ) from error
            scale_modifiers.append(
                (
                    modifier,
                    1.0 + slope * group.initial,
                    np.full(group.size, slope),
                )
            )
        elif modifier_type == "lumi":
            scale_modifiers.append(
                (modifier, group.initial.copy(), np.ones(group.size))
            )
        elif modifier_type in _VECTOR_TYPES:
            uncertainty = np.asarray(modifier.get("data"), dtype=float)
            if uncertainty.shape != raw_nominal.shape or np.any(uncertainty < 0):
                raise InvalidWorkspace(
                    f"{modifier_type} modifier {group.name!r} must provide one "
                    "nonnegative absolute uncertainty per sample bin"
                )
            scale_modifiers.append(
                (modifier, group.initial.copy(), np.ones(group.size))
            )
            assert group.uncertainty_sq is not None
            assert group.affected_nominal is not None
            group.uncertainty_sq += np.square(uncertainty)
            group.affected_nominal += raw_nominal

    scale = np.ones_like(raw_nominal)
    for _, factor, _ in scale_modifiers:
        scale *= factor if factor.size > 1 else factor[0]
    nominal_expected[channel_slice] += template * scale

    for modifier, factor, derivative in scale_modifiers:
        group = groups[modifier["name"]]
        column_slice = group_slices[group.name]
        other_scale = np.ones_like(raw_nominal)
        for other_modifier, other_factor, _ in scale_modifiers:
            if other_modifier is modifier:
                continue
            other_scale *= (
                other_factor if other_factor.size > 1 else other_factor[0]
            )
        if group.size == 1:
            response[channel_slice, column_slice.start] += (
                template * other_scale * derivative[0]
            )
        else:
            for bin_index in range(group.size):
                response[
                    channel_slice.start + bin_index,
                    column_slice.start + bin_index,
                ] += template[bin_index] * other_scale[bin_index] * derivative[bin_index]

    for modifier, delta in histosys:
        group = groups[modifier["name"]]
        response[channel_slice, group_slices[group.name].start] += delta * scale


def _compile_observations(
    workspace: dict[str, Any],
    channel_names: tuple[str, ...],
    channel_slices: dict[str, slice],
) -> FloatArray:
    raw_observations = workspace.get("observations")
    if not isinstance(raw_observations, list):
        raise InvalidWorkspace("Workspace observations must be a list")
    by_name = {
        observation.get("name"): observation.get("data")
        for observation in raw_observations
    }
    chunks: list[FloatArray] = []
    for name in channel_names:
        if name not in by_name:
            raise InvalidWorkspace(f"Observation for channel {name!r} is missing")
        data = np.asarray(by_name[name], dtype=float)
        expected_size = channel_slices[name].stop - channel_slices[name].start
        if data.shape != (expected_size,) or not np.all(np.isfinite(data)):
            raise InvalidWorkspace(
                f"Observation for channel {name!r} must contain "
                f"{expected_size} finite value(s)"
            )
        chunks.append(data)
    return np.concatenate(chunks)


def load(
    source: WorkspaceSource,
    *,
    measurement: str | None = None,
    strict: bool = False,
) -> Model:
    """Load and compile a pyhf workspace JSON document."""

    workspace = _read_source(source)
    missing = _TOP_LEVEL_KEYS.difference(workspace)
    if missing:
        raise InvalidWorkspace(
            f"Workspace is missing required field(s): {', '.join(sorted(missing))}"
        )
    if workspace["version"] != "1.0.0":
        raise InvalidWorkspace(
            f"Unsupported workspace version {workspace['version']!r}; expected '1.0.0'"
        )

    warning_messages: list[str] = []
    extras = sorted(set(workspace).difference(_TOP_LEVEL_KEYS))
    if extras:
        message = f"Ignoring unknown top-level workspace field(s): {', '.join(extras)}"
        if strict:
            raise InvalidWorkspace(message)
        warning_messages.append(message)
        warnings.warn(message, WorkspaceWarning, stacklevel=2)

    channels = workspace["channels"]
    if not isinstance(channels, list) or not channels:
        raise InvalidWorkspace("Workspace requires at least one channel")
    if any(
        not isinstance(channel, dict)
        or not isinstance(channel.get("name"), str)
        or not channel["name"]
        for channel in channels
    ):
        raise InvalidWorkspace("Every channel requires a nonempty name")
    channels = sorted(channels, key=lambda channel: channel["name"])
    selected = _measurement(workspace, measurement, warning_messages)
    group_list, groups = _discover_groups(channels)
    poi_name = _apply_measurement_settings(selected, groups)
    channel_names, channel_slices, nbins = _channel_layout(channels)

    group_slices: dict[str, slice] = {}
    parameter_names: list[str] = []
    offset = 0
    for group in group_list:
        group_slices[group.name] = slice(offset, offset + group.size)
        parameter_names.extend(group.scalar_names)
        offset += group.size

    nominal_expected = np.zeros(nbins)
    response = np.zeros((nbins, offset))
    for channel in channels:
        for sample in channel["samples"]:
            _compile_sample(
                sample,
                channel_slices[channel["name"]],
                group_slices,
                groups,
                nominal_expected,
                response,
            )

    for group in group_list:
        if group.modifier_type in _VECTOR_TYPES:
            assert group.uncertainty_sq is not None
            assert group.affected_nominal is not None
            uncertainty = np.sqrt(group.uncertainty_sq)
            positive = uncertainty > 0
            group.precision[positive] = np.square(
                group.affected_nominal[positive] / uncertainty[positive]
            )
            group.fixed[~positive] = True

    return Model(
        channel_names=channel_names,
        channel_slices=channel_slices,
        parameter_names=tuple(parameter_names),
        poi_name=poi_name,
        nominal_expected=nominal_expected,
        response=response,
        observations=_compile_observations(
            workspace, channel_names, channel_slices
        ),
        initial=np.concatenate([group.initial for group in group_list]),
        lower_bounds=np.concatenate([group.lower for group in group_list]),
        upper_bounds=np.concatenate([group.upper for group in group_list]),
        fixed=np.concatenate([group.fixed for group in group_list]),
        constraint_mean=np.concatenate([group.mean for group in group_list]),
        constraint_precision=np.concatenate(
            [group.precision for group in group_list]
        ),
        modifier_types=tuple(
            sorted({group.modifier_type for group in group_list})
        ),
        warnings=tuple(warning_messages),
    )
