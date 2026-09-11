"""Fast standalone MiMeDO viewer helpers (v4 patched copy of mimedo_viewer_v3).

v4 keeps the v3 API and adds voxel-only deformed snapshot coloring. It can
color snapshots that do not propagate grain-level metadata by using voxel-level
fields such as stress, Mises stress/strain, strain, and plastic strain.

This module is intentionally independent from ``app.py``.  It establishes the
data-object inspection, control-state decisions, and Plotly figure builders that
can later be wired into Dash callbacks.

Main entry points:

    inspect_mimedo("c779a1c5.json")
    build_control_state("c779a1c5.json")
    build_microstructure_figure("c779a1c5.json")
    build_microstructure_snapshots_figure("c779a1c5.json")
    build_microstructure_comparison_figure("c779a1c5.json")
    build_texture_figure("c779a1c5.json")
    build_statistics_figure("c779a1c5.json")
    build_mechanical_figure("c779c1a4.json")
    mimedat_viewer("c779a1c5.json", "c779c1a4.json")

The implementation favors fast control reactions: the JSON file, snapshot
arrays, reusable voxel mesh geometry, and voxel graph topology are cached by
path, size, mtime, snapshot, grain filter, voxel size, and connectivity.
"""

from __future__ import annotations

import colorsys
import html
import importlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import plotly
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _plotly_supports_multi_legend() -> bool:
    """Multiple legends (trace ``legend=`` + layout ``legend2``/``legend3``)
    require plotly >= 5.15. Assume unsupported when the version is unclear."""
    version = getattr(plotly, "__version__", "")
    try:
        from packaging.version import Version

        return Version(version) >= Version("5.15")
    except Exception:
        pass
    try:
        parts = str(version).split(".")
        major = int("".join(ch for ch in parts[0] if ch.isdigit()) or -1)
        minor = int("".join(ch for ch in parts[1] if ch.isdigit()) or 0) if len(parts) > 1 else 0
        return (major, minor) >= (5, 15)
    except Exception:
        return False


_PLOTLY_MULTI_LEGEND = _plotly_supports_multi_legend()


CONNECTIVITY_OPTIONS = (6, 18, 26)
VIEW_TYPES = ("rve", "both", "graph")
SNAPSHOT_MODES = ("snapshots", "one", "multiple", "compare")
GRAPH_SCOPES = ("entire", "slice")
COMPARISON_MAX_PANELS = 12
MICROSTRUCTURE_MAX_SNAPSHOT_PANELS = 12
MICROSTRUCTURE_DETAILED_HOVER_DEFAULT = False
TEXTURE_VIEWS = ("pf", "pdf")
TEXTURE_REFERENCE_STATUSES = ("undeformed",)
TEXTURE_MAX_SNAPSHOTS = 3
TEXTURE_MAX_POINTS = 10000
TEXTURE_POLE_FAMILIES = {
    "<100>": (1, 0, 0),
    "<110>": (1, 1, 0),
    "<111>": (1, 1, 1),
}
STATISTICS_LABEL_INITIAL = "Initial"
STATISTICS_LABEL_REGRIDDED = "Cold rolled"
MULTI_COMPONENT_GROUPS = ("normal", "shear", "all")
STRAIN_SOURCES = ("total_strain", "plastic_strain")
MECHANICAL_COMPONENT_SUFFIXES = ("11", "22", "33", "12", "13", "23")
NORMAL_SUFFIXES = ("11", "22", "33")
SHEAR_SUFFIXES = ("12", "13", "23")
MECHANICAL_LINE_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
)

COLOR_FIELD_REQUIREMENTS = {
    "grain_id": "voxel.grain_id",
    "phase_id": "voxel.phase_id or grain.phase_id + voxel.grain_id",
    "IPFcolor_(1 0 0)": "voxel.IPFcolor_(1 0 0)",
    "mises_equivalent_stress": "voxel.mises_equivalent_stress",
    "stress": "voxel.stress",
    "strain": "voxel.strain",
    "mises_equivalent_strain": "voxel.mises_equivalent_strain",
    "plastic_strain": "voxel.plastic_strain",
    "mises_equivalent_plastic_strain": "voxel.mises_equivalent_plastic_strain",
    "first_piola_kirchhoff_stress": "voxel.first_piola_kirchhoff_stress",
    "deformation_gradient": "voxel.deformation_gradient",
    "plastic_deformation_gradient": "voxel.plastic_deformation_gradient",
    "elastic_deformation_gradient": "voxel.elastic_deformation_gradient",
    "plastic_velocity_gradient": "voxel.plastic_velocity_gradient",
    "resistance_against_plastic_slip": "voxel.resistance_against_plastic_slip",
}

COLOR_FIELD_LABELS = {
    "grain_id": "Grain ID",
    "phase_id": "Phase ID",
    "IPFcolor_(1 0 0)": "IPF color (100)",
    "mises_equivalent_stress": "Mises equivalent stress",
    "stress": "Stress tensor norm",
    "strain": "Strain tensor norm",
    "mises_equivalent_strain": "Mises equivalent strain",
    "plastic_strain": "Plastic strain tensor norm",
    "mises_equivalent_plastic_strain": "Mises equivalent plastic strain",
    "first_piola_kirchhoff_stress": "First Piola-Kirchhoff stress norm",
    "deformation_gradient": "Deformation gradient norm",
    "plastic_deformation_gradient": "Plastic deformation gradient norm",
    "elastic_deformation_gradient": "Elastic deformation gradient norm",
    "plastic_velocity_gradient": "Plastic velocity gradient norm",
    "resistance_against_plastic_slip": "Slip resistance norm",
}

COLOR_FIELD_DEFAULT_ORDER = (
    "grain_id",
    "IPFcolor_(1 0 0)",
    "mises_equivalent_stress",
    "mises_equivalent_strain",
    "mises_equivalent_plastic_strain",
    "stress",
    "strain",
    "plastic_strain",
    "first_piola_kirchhoff_stress",
    "resistance_against_plastic_slip",
    "phase_id",
    "deformation_gradient",
    "plastic_deformation_gradient",
    "elastic_deformation_gradient",
    "plastic_velocity_gradient",
)

# ---------------------------------------------------------------------------
# Abaqus-style scalar reduction of per-voxel tensor fields.
#
# A per-voxel field such as ``stress`` is stored as a full 3x3 tensor. To colour
# a contour plot the way Abaqus does, the tensor is reduced to a scalar: either a
# single component (S11 ... S23) or the von Mises equivalent. The reduction is
# encoded in ``color_by`` as ``"<field>:<reduction>"`` (e.g. ``"stress:11"`` or
# ``"strain:mises"``). A plain ``color_by`` with no colon keeps the legacy
# behaviour (categorical id, RGB, or tensor Frobenius norm).
# ---------------------------------------------------------------------------

# Tensor fields that receive component/Mises reductions in the colour menu.
TENSOR_REDUCTION_FIELDS = ("stress", "strain", "plastic_strain")

# Fields whose Mises uses the stress convention sqrt(3/2 s:s); everything else
# in TENSOR_REDUCTION_FIELDS uses the strain convention sqrt(2/3 e:e).
STRESS_LIKE_FIELDS = ("stress", "first_piola_kirchhoff_stress")

# Reduction suffix -> (row, col) into the symmetric 3x3 tensor (0-based).
TENSOR_COMPONENT_INDEX = {
    "11": (0, 0),
    "22": (1, 1),
    "33": (2, 2),
    "12": (0, 1),
    "13": (0, 2),
    "23": (1, 2),
}
TENSOR_REDUCTIONS = ("11", "22", "33", "12", "13", "23", "mises")

# Short symbol used in colourbar/menu labels for each reducible tensor field.
TENSOR_FIELD_SYMBOL = {
    "stress": "S",
    "strain": "E",
    "plastic_strain": "Ep",
}

# Mises menu/label text per field.
TENSOR_MISES_LABEL = {
    "stress": "Mises stress",
    "strain": "Mises strain",
    "plastic_strain": "Mises plastic strain",
}

# Menu-friendly display name for each reducible tensor field.
TENSOR_MENU_NAME = {
    "stress": "Stress",
    "strain": "Strain",
    "plastic_strain": "Plastic strain",
}

# Standalone per-voxel Mises scalar fields and the tensor they duplicate. When
# the tensor is present, the tensor's ":mises" reduction replaces the scalar in
# the colour menu to avoid two identical entries.
MISES_SCALAR_FOR_TENSOR = {
    "mises_equivalent_stress": "stress",
    "mises_equivalent_strain": "strain",
    "mises_equivalent_plastic_strain": "plastic_strain",
}

# Colourmaps offered to the user (label -> plotly colorscale name). The default
# is a blue->green->red rainbow, matching a default Abaqus contour plot.
COLORMAP_OPTIONS = (
    ("Rainbow (Abaqus)", "Jet"),
    ("Turbo", "Turbo"),
    ("Viridis", "Viridis"),
)
DEFAULT_COLORMAP = "Jet"

MAX_CUBE_AXIS_CELLS = 50
MAX_CUBE_VOXELS = MAX_CUBE_AXIS_CELLS**3

QUALITATIVE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
)


def _file_cache_key(json_path: str | Path) -> tuple[str, int, int]:
    path = Path(json_path).expanduser().resolve()
    stat = path.stat()
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=8)
def _load_json_cached(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    del mtime_ns, size
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("MiMeDO JSON root must be an object")
    return data


def load_mimedo(json_path: str | Path) -> dict[str, Any]:
    """Load and cache one MiMeDO JSON object."""
    return _load_json_cached(*_file_cache_key(json_path))


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return f"array len={len(value)}"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _is_snapshot_array(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _sample_keys(items: Any, limit: int = 64) -> set[str]:
    keys: set[str] = set()
    if not isinstance(items, list):
        return keys
    for item in items[:limit]:
        if isinstance(item, dict):
            keys.update(item.keys())
    return keys


def _snapshot(data: dict[str, Any], snapshot_index: int) -> dict[str, Any]:
    micro = data.get("microstructure")
    if not _is_snapshot_array(micro):
        raise ValueError("microstructure is not a valid snapshot array")
    if not (0 <= int(snapshot_index) < len(micro)):
        raise IndexError(
            f"snapshot_index={snapshot_index} out of range [0, {len(micro)})"
        )
    return micro[int(snapshot_index)]


def _available_color_fields(snapshot: dict[str, Any]) -> dict[str, bool]:
    voxels = snapshot.get("voxels")
    grains = snapshot.get("grains")
    voxel_keys = _sample_keys(voxels)
    grain_keys = _sample_keys(grains)
    available = {
        field: field in voxel_keys
        for field in COLOR_FIELD_REQUIREMENTS
        if field not in {"grain_id", "phase_id"}
    }
    available["grain_id"] = "grain_id" in voxel_keys
    available["phase_id"] = "phase_id" in voxel_keys or (
        "grain_id" in voxel_keys and "phase_id" in grain_keys
    )
    return {field: bool(available.get(field)) for field in COLOR_FIELD_REQUIREMENTS}


def _mechanical_lengths(module: Any) -> dict[str, int | None]:
    if not isinstance(module, dict):
        return {}
    return {
        key: len(value) if isinstance(value, list) else None
        for key, value in module.items()
    }


def _paired_strain_key(stress_key: str, strain_source: str) -> str | None:
    if stress_key == "equivalent_stress":
        if strain_source == "plastic_strain":
            return "equivalent_plastic_strain"
        return "equivalent_strain"
    suffix = _component_suffix(str(stress_key))
    if suffix in MECHANICAL_COMPONENT_SUFFIXES:
        return _strain_key(strain_source, suffix)
    return None


def _mechanical_pairs(
    stress_lengths: Mapping[str, int | None],
    strain_lengths: Mapping[str, int | None],
    strain_source: str,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for stress_key in _sort_component_keys(stress_lengths.keys()):
        strain_key = _paired_strain_key(str(stress_key), strain_source)
        if strain_key is None or strain_key not in strain_lengths:
            continue
        if not stress_lengths.get(stress_key) or not strain_lengths.get(strain_key):
            continue
        pairs.append((str(stress_key), strain_key))
    return pairs


def _sort_component_keys(keys: Any) -> list[str]:
    keys = list(keys)
    order = {
        suffix: index + 1
        for index, suffix in enumerate(MECHANICAL_COMPONENT_SUFFIXES)
    }
    order["equivalent_stress"] = 0
    order["equivalent_strain"] = 0
    order["equivalent_plastic_strain"] = 0
    return sorted(
        keys,
        key=lambda key: order.get(str(key), order.get(_component_suffix(str(key)), 999)),
    )


def _snapshot_grid_status(snapshot: dict[str, Any]) -> str:
    grid = snapshot.get("grid") if isinstance(snapshot, dict) else None
    if not isinstance(grid, dict):
        return ""
    return str(grid.get("status", "")).strip().lower()


def _is_texture_reference_snapshot(snapshot: dict[str, Any]) -> bool:
    return _snapshot_grid_status(snapshot) in TEXTURE_REFERENCE_STATUSES


def _has_voxel_orientations(snapshot: dict[str, Any]) -> bool:
    voxels = snapshot.get("voxels") if isinstance(snapshot, dict) else None
    return "orientation" in _sample_keys(voxels)


def _phase_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    phases = data.get("phases")
    if phases is None:
        phases = data.get("phase")
    if isinstance(phases, dict):
        phases = [phases]
    if not isinstance(phases, list):
        return []
    return [phase for phase in phases if isinstance(phase, dict)]


def _phase_symmetry_name(data: dict[str, Any]) -> str | None:
    phases = _phase_records(data)
    if not phases:
        return None
    orientation = phases[0].get("orientation")
    if not isinstance(orientation, dict):
        return None
    value = orientation.get("crystal_symmetry_group")
    return str(value).strip() if value else None


def _texture_report(data: dict[str, Any]) -> dict[str, Any]:
    micro = data.get("microstructure")
    symmetry_name = _phase_symmetry_name(data)
    undeformed: list[dict[str, Any]] = []
    if _is_snapshot_array(micro):
        for index, snapshot in enumerate(micro):
            status = _snapshot_grid_status(snapshot)
            if not _is_texture_reference_snapshot(snapshot):
                continue
            if not _has_voxel_orientations(snapshot):
                continue
            voxels = snapshot.get("voxels", [])
            undeformed.append(
                {
                    "index": index,
                    "time": snapshot.get("time_point"),
                    "status": status,
                    "voxel_count": len(voxels) if isinstance(voxels, list) else None,
                }
            )
    available = bool(undeformed and symmetry_name)
    reason = None
    if not available:
        if not _is_snapshot_array(micro):
            reason = "microstructure snapshots are not available"
        elif not symmetry_name:
            reason = "phase crystal_symmetry_group is not available"
        else:
            reason = (
                "no undeformed or initial snapshots with voxel orientations "
                "are available"
            )
    return {
        "available": available,
        "reason": reason,
        "undeformed_snapshots": undeformed,
        "snapshot_count": len(undeformed),
        "phase_symmetry": symmetry_name,
        "orientation_source": "voxels",
        "views": list(TEXTURE_VIEWS),
        "pole_families": list(TEXTURE_POLE_FAMILIES.keys()),
        "max_snapshots": TEXTURE_MAX_SNAPSHOTS,
        "default_snapshot_count": TEXTURE_MAX_SNAPSHOTS,
        "max_points": TEXTURE_MAX_POINTS,
    }


def _statistics_report(data: dict[str, Any]) -> dict[str, Any]:
    micro = data.get("microstructure")
    undeformed: list[dict[str, Any]] = []
    if _is_snapshot_array(micro):
        for index, snapshot in enumerate(micro):
            grid = snapshot.get("grid") if isinstance(snapshot, dict) else None
            voxels = snapshot.get("voxels") if isinstance(snapshot, dict) else None
            grains = snapshot.get("grains") if isinstance(snapshot, dict) else None
            if _snapshot_grid_status(snapshot) != "undeformed":
                continue
            if not isinstance(grid, dict):
                continue
            if not isinstance(voxels, list) or not voxels:
                continue
            if not isinstance(grains, list) or not grains:
                continue
            undeformed.append(
                {
                    "label": f"snapshot {index}"
                    + (
                        ""
                        if snapshot.get("time_point") is None
                        else f" | time={snapshot['time_point']}"
                    ),
                    "value": index,
                    "time": snapshot.get("time_point"),
                    "voxel_count": len(voxels),
                    "grain_count": len(grains),
                }
            )

    if not _is_snapshot_array(micro):
        reason = "microstructure snapshots are not available"
    elif not undeformed:
        reason = "no undeformed snapshots with voxel/grain/grid data are available"
    else:
        reason = None

    return {
        "available": reason is None,
        "reason": reason,
        "undeformed_snapshots": undeformed,
        "snapshot_count": len(undeformed),
        "default_initial": undeformed[0]["value"] if undeformed else None,
        "default_regridded": undeformed[-1]["value"] if undeformed else None,
        "default_snapshots": [item["value"] for item in undeformed[:2]],
    }


def inspect_mimedo(
    json_path: str | Path,
    *,
    snapshot_index: int = 0,
) -> dict[str, Any]:
    """Return detected modules and available plotting fields for one object."""
    data = load_mimedo(json_path)
    micro = data.get("microstructure")
    has_microstructure = _is_snapshot_array(micro) and len(micro) > 0
    has_stress = isinstance(data.get("stress"), dict)
    has_total_strain = isinstance(data.get("total_strain"), dict)
    has_plastic_strain = isinstance(data.get("plastic_strain"), dict)
    texture_report = _texture_report(data)
    statistics_report = _statistics_report(data)

    micro_report: dict[str, Any] = {
        "available": has_microstructure,
        "reason": None if has_microstructure else "microstructure module is not available",
    }
    if has_microstructure:
        sidx = min(max(0, int(snapshot_index)), len(micro) - 1)
        snap = micro[sidx]
        voxels = snap.get("voxels")
        grains = snap.get("grains")
        color_fields = _available_color_fields(snap)
        micro_report.update(
            {
                "snapshot_count": len(micro),
                "selected_snapshot": sidx,
                "snapshot_keys": sorted(snap.keys()),
                "voxel_count": len(voxels) if isinstance(voxels, list) else None,
                "grain_count": len(grains) if isinstance(grains, list) else None,
                "has_voxels": isinstance(voxels, list) and len(voxels) > 0,
                "has_grains": isinstance(grains, list) and len(grains) > 0,
                "has_grain_ids": bool(color_fields.get("grain_id")),
                "has_grid": isinstance(snap.get("grid"), dict),
                "time": snap.get("time_point"),
                "color_fields": color_fields,
            }
        )

    stress_lengths = _mechanical_lengths(data.get("stress"))
    strain_lengths = _mechanical_lengths(data.get("total_strain"))
    plastic_lengths = _mechanical_lengths(data.get("plastic_strain"))
    mechanical_pairs = _mechanical_pairs(
        stress_lengths,
        strain_lengths,
        "total_strain",
    )
    plastic_pairs = _mechanical_pairs(
        stress_lengths,
        plastic_lengths,
        "plastic_strain",
    )
    mechanical_available = has_stress and has_total_strain and bool(mechanical_pairs)
    mechanical_report = {
        "available": mechanical_available,
        "reason": None
        if mechanical_available
        else "no plottable stress/total_strain component pairs are available",
        "stress_components": _sort_component_keys(stress_lengths.keys()),
        "strain_components": _sort_component_keys(strain_lengths.keys()),
        "plastic_components": _sort_component_keys(plastic_lengths.keys()),
        "component_pairs": mechanical_pairs,
        "plastic_component_pairs": plastic_pairs,
        "stress_lengths": stress_lengths,
        "strain_lengths": strain_lengths,
        "plastic_lengths": plastic_lengths,
        "has_plastic_strain": has_plastic_strain,
    }

    return {
        "path": str(Path(json_path).expanduser().resolve()),
        "top_level": {key: _kind(value) for key, value in data.items()},
        "modules": {
            "microstructure": has_microstructure,
            "texture": texture_report["available"],
            "statistics": statistics_report["available"],
            "mechanical_response": mechanical_available,
            "plastic_strain": has_plastic_strain,
        },
        "microstructure": micro_report,
        "texture": texture_report,
        "statistics": statistics_report,
        "mechanical": mechanical_report,
    }


def _control(enabled: bool, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    state = {"enabled": bool(enabled), "frozen": not bool(enabled), "reason": reason}
    state.update(extra)
    return state


def build_control_state(
    json_path: str | Path,
    *,
    plot: str = "auto",
    snapshot_mode: str = "snapshots",
    snapshot_count: int = 1,
    view_type: str = "rve",
    graph_scope: str = "entire",
    snapshot_index: int = 0,
) -> dict[str, Any]:
    """Describe which visualization controls should be enabled or frozen."""
    report = inspect_mimedo(json_path, snapshot_index=snapshot_index)
    has_micro = report["modules"]["microstructure"]
    has_texture = report["modules"]["texture"]
    has_stats = report["modules"]["statistics"]
    has_mech = report["modules"]["mechanical_response"]
    snapshot_mode = snapshot_mode if snapshot_mode in SNAPSHOT_MODES else "snapshots"
    view_type = view_type if view_type in VIEW_TYPES else "rve"
    graph_scope = graph_scope if graph_scope in GRAPH_SCOPES else "entire"
    snapshots_active = has_micro and snapshot_mode in {"snapshots", "one", "multiple"}
    range_active = has_micro and snapshot_mode in {"snapshots", "multiple", "compare"}
    compare_active = has_micro and snapshot_mode == "compare"
    single_active = (
        has_micro
        and (
            snapshot_mode == "one"
            or (snapshot_mode == "snapshots" and int(snapshot_count) <= 1)
        )
    )
    graph_active = single_active and view_type in {"both", "graph"}
    slice_active = graph_active and graph_scope == "slice"

    if plot == "auto":
        active_plot = (
            "microstructure"
            if has_micro
            else "texture"
            if has_texture
            else "statistics"
            if has_stats
            else "mechanical"
            if has_mech
            else None
        )
    else:
        active_plot = plot

    micro_reason = None if has_micro else report["microstructure"]["reason"]
    texture_reason = None if has_texture else report["texture"]["reason"]
    stats_reason = None if has_stats else report["statistics"]["reason"]
    graph_reason = None if graph_active else "graph controls require one-snapshot graph view"
    mech_reason = None if has_mech else report["mechanical"]["reason"]

    controls = {
        "plot_family": _control(
            has_micro or has_texture or has_stats or has_mech,
            None
            if has_micro or has_texture or has_stats or has_mech
            else "no plottable modules are available",
            active=active_plot,
            options=[
                {"value": "microstructure", "enabled": has_micro},
                {"value": "texture", "enabled": has_texture},
                {"value": "statistics", "enabled": has_stats},
                {"value": "mechanical", "enabled": has_mech},
            ],
        ),
        "microstructure": {
            "snapshot_mode": _control(has_micro, micro_reason, value=snapshot_mode),
            "snapshot_selector": _control(
                single_active,
                None if has_micro else micro_reason,
                value=snapshot_index,
            ),
            "multiple_snapshot_range": _control(
                range_active or snapshots_active,
                None if has_micro else micro_reason,
            ),
            "comparison_color_rows": _control(
                compare_active,
                None if has_micro else micro_reason,
            ),
            "grain_selector": _control(
                has_micro and bool(report["microstructure"].get("has_grain_ids")),
                None if has_micro else micro_reason,
            ),
            "view_type": _control(has_micro, micro_reason, value=view_type),
            "field_coloring": _control(
                has_micro,
                micro_reason,
                fields=report["microstructure"].get("color_fields", {}),
            ),
            "grid_information": _control(
                has_micro and bool(report["microstructure"].get("has_grid")),
                None
                if has_micro and report["microstructure"].get("has_grid")
                else "grid is not available for this snapshot",
            ),
        },
        "graph": {
            "scope": _control(graph_active, graph_reason, value=graph_scope),
            "slice_axis": _control(slice_active, graph_reason),
            "slice_index": _control(slice_active, graph_reason),
            "edge_labels": _control(graph_active, graph_reason),
            "periodic_wrap_edges": _control(graph_active, graph_reason),
            "connectivity": _control(
                graph_active,
                graph_reason,
                options=list(CONNECTIVITY_OPTIONS),
                value=6,
            ),
            "node_size": _control(graph_active, graph_reason),
            "edge_width": _control(graph_active, graph_reason),
        },
        "texture": {
            "snapshot_selector": _control(
                has_texture,
                texture_reason,
                options=report["texture"]["undeformed_snapshots"],
            ),
            "view": _control(
                has_texture,
                texture_reason,
                options=list(TEXTURE_VIEWS),
                value="pf",
            ),
            "pole_families": _control(
                has_texture,
                texture_reason,
                options=list(TEXTURE_POLE_FAMILIES.keys()),
            ),
            "orientation_source": _control(
                False,
                "texture orientation source is fixed to voxel orientations",
                value="voxels",
            ),
        },
        "statistics": {
            "snapshot_selector": _control(
                has_stats,
                stats_reason,
                options=report["statistics"]["undeformed_snapshots"],
                value=report["statistics"].get("default_snapshots", []),
            ),
        },
        "mechanical": {
            "plot_mode": _control(has_mech, mech_reason),
            "stress_component": _control(
                has_mech,
                mech_reason,
                options=report["mechanical"]["stress_components"],
            ),
            "strain_component": _control(
                has_mech,
                mech_reason,
                options=report["mechanical"]["strain_components"],
            ),
            "multi_component_group": _control(
                has_mech,
                mech_reason,
                options=list(MULTI_COMPONENT_GROUPS),
            ),
            "plastic_strain": _control(
                has_mech and bool(report["mechanical"]["plastic_component_pairs"]),
                None
                if has_mech and report["mechanical"]["plastic_component_pairs"]
                else "plastic_strain is not available",
                options=report["mechanical"]["plastic_components"],
            ),
        },
    }
    return {"report": report, "controls": controls}


def _categorical_color_map(values: np.ndarray) -> dict[int, str]:
    unique = np.unique(values.astype(int))
    color_for_value = {}
    for index, value in enumerate(unique.tolist()):
        if index < len(QUALITATIVE_COLORS):
            color = QUALITATIVE_COLORS[index]
        else:
            hue = (index * 0.61803398875) % 1.0
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.88)
            color = f"rgb({int(red * 255)}, {int(green * 255)}, {int(blue * 255)})"
        color_for_value[int(value)] = color
    return color_for_value


def _as_numeric_array(values: list[Any]) -> np.ndarray:
    flattened = []
    for value in values:
        if isinstance(value, (int, float)):
            flattened.append(float(value))
        elif isinstance(value, list):
            arr = np.asarray(value, dtype=float)
            flattened.append(float(np.linalg.norm(arr.ravel())))
        else:
            flattened.append(np.nan)
    return np.asarray(flattened, dtype=float)


def _rgb_colors(values: list[Any]) -> list[str] | None:
    colors: list[str] = []
    for value in values:
        if not isinstance(value, list) or len(value) != 3:
            return None
        arr = np.asarray(value, dtype=float)
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(int)
        colors.append(f"rgb({arr[0]}, {arr[1]}, {arr[2]})")
    return colors


def _safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=32)
def _snapshot_arrays_cached(
    path: str,
    mtime_ns: int,
    size: int,
    snapshot_index: int,
) -> dict[str, Any]:
    data = _load_json_cached(path, mtime_ns, size)
    snap = _snapshot(data, snapshot_index)
    voxels = snap.get("voxels")
    if not isinstance(voxels, list) or not voxels:
        raise ValueError(f"Snapshot {snapshot_index} has no voxel data")

    voxel_index = np.asarray([v["voxel_index"] for v in voxels], dtype=np.int32)
    grain_id = np.asarray(
        [_safe_int(v.get("grain_id"), -1) for v in voxels],
        dtype=np.int32,
    )
    grains = snap.get("grains") if isinstance(snap.get("grains"), list) else []
    grain_to_phase = {
        int(grain["grain_id"]): int(grain["phase_id"])
        for grain in grains
        if isinstance(grain, dict)
        and "grain_id" in grain
        and "phase_id" in grain
    }
    if any(isinstance(voxel, dict) and "phase_id" in voxel for voxel in voxels):
        phase_id = np.asarray(
            [_safe_int(voxel.get("phase_id"), -1) for voxel in voxels],
            dtype=np.int32,
        )
    else:
        phase_id = np.asarray(
            [grain_to_phase.get(int(grain), -1) for grain in grain_id],
            dtype=np.int32,
        )
    field_values = {
        field: [voxel.get(field) for voxel in voxels]
        for field in COLOR_FIELD_REQUIREMENTS
        if field != "phase_id"
    }
    field_values["phase_id"] = phase_id.tolist()
    return {
        "voxel_index": voxel_index,
        "grain_id": grain_id,
        "phase_id": phase_id,
        "field_values": field_values,
        "time": snap.get("time_point"),
        "grid": snap.get("grid") if isinstance(snap.get("grid"), dict) else None,
        "grain_ids": sorted(
            int(value) for value in np.unique(grain_id).tolist() if int(value) >= 0
        ),
    }


def _snapshot_arrays(json_path: str | Path, snapshot_index: int) -> dict[str, Any]:
    return _snapshot_arrays_cached(*_file_cache_key(json_path), int(snapshot_index))


def _split_color_by(color_by: str) -> tuple[str, str | None]:
    """Split ``"stress:11"`` into ``("stress", "11")``; plain names get ``None``."""
    text = str(color_by)
    if ":" in text:
        base, reduction = text.split(":", 1)
        return base, reduction or None
    return text, None


def _is_reducible_tensor_field(base: str) -> bool:
    return base in TENSOR_REDUCTION_FIELDS


def _stack_tensors(raw_values: list[Any]) -> np.ndarray | None:
    """Stack a per-voxel list of 3x3 (or flat-9) tensors into an ``(n, 3, 3)`` array.

    Returns ``None`` when the values are not uniform tensors (e.g. scalars, RGB
    triples, or ragged/missing entries), so callers can fall back.
    """
    try:
        arr = np.asarray(raw_values, dtype=float)
    except (ValueError, TypeError):
        return None
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return arr
    if arr.ndim == 2 and arr.shape[1] == 9:
        return arr.reshape(-1, 3, 3)
    return None


def _reduce_stacked_tensors(
    tensors: np.ndarray,
    base: str,
    reduction: str,
) -> np.ndarray:
    """Reduce an ``(n, 3, 3)`` tensor stack to a scalar per voxel.

    Components read the symmetric tensor directly. ``mises`` uses the stress
    convention ``sqrt(3/2 s:s)`` for stress-like fields and the strain
    convention ``sqrt(2/3 e:e)`` otherwise, matching the homogenized definitions
    used by the exporter.
    """
    sym = 0.5 * (tensors + np.transpose(tensors, (0, 2, 1)))
    if reduction in TENSOR_COMPONENT_INDEX:
        row, col = TENSOR_COMPONENT_INDEX[reduction]
        return sym[:, row, col]
    if reduction == "mises":
        trace = np.trace(sym, axis1=1, axis2=2)
        dev = sym - (trace / 3.0)[:, None, None] * np.eye(3)[None, :, :]
        contraction = np.sum(dev * dev, axis=(1, 2))
        factor = 1.5 if base in STRESS_LIKE_FIELDS else (2.0 / 3.0)
        return np.sqrt(np.clip(factor * contraction, 0.0, None))
    return np.full(tensors.shape[0], np.nan, dtype=float)


def _field_scalar_array(arrays: dict[str, Any], color_by: str) -> np.ndarray:
    """Numeric per-voxel array for a (possibly reduction-qualified) colour field.

    ``"stress:11"``/``"strain:mises"`` reduce the stored tensor; a plain field
    name keeps the legacy behaviour (direct scalar, or tensor Frobenius norm).
    """
    base, reduction = _split_color_by(color_by)
    raw_values = arrays.get("field_values", {}).get(base, [])
    if reduction is not None and _is_reducible_tensor_field(base):
        stacked = _stack_tensors(raw_values)
        if stacked is not None:
            return _reduce_stacked_tensors(stacked, base, reduction)
    return _as_numeric_array(raw_values)


def _reduction_label(base: str, reduction: str | None) -> str:
    """Human label for a reduction, e.g. ``S11``, ``Mises stress``."""
    if reduction is None:
        return COLOR_FIELD_LABELS.get(base, base)
    if reduction == "mises":
        return TENSOR_MISES_LABEL.get(base, f"{base} Mises")
    symbol = TENSOR_FIELD_SYMBOL.get(base, base)
    return f"{symbol}{reduction}"


def _color_field_unit(data: dict[str, Any], base: str) -> str:
    """Unit string for a colour field, read from the JSON ``units`` block."""
    units = data.get("units") if isinstance(data.get("units"), dict) else {}
    if base in STRESS_LIKE_FIELDS or base == "resistance_against_plastic_slip":
        return _unit_text(units.get("Stress"))
    return "-"


def _colorbar_title(color_by: str, unit: str) -> str:
    base, reduction = _split_color_by(color_by)
    label = _reduction_label(base, reduction)
    if unit and unit != "-":
        return f"{label}<br>[{unit}]"
    return label


def _resolve_color_range(
    values: np.ndarray,
    cmin: float | None,
    cmax: float | None,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    lo = float(cmin) if cmin is not None else (float(np.min(finite)) if finite.size else 0.0)
    hi = float(cmax) if cmax is not None else (float(np.max(finite)) if finite.size else 1.0)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo + 1.0
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def _color_field_present_in_arrays(arrays: dict[str, Any], color_by: str) -> bool:
    base, _ = _split_color_by(color_by)
    if base == "grain_id":
        values = np.asarray(arrays.get("grain_id", []), dtype=np.int32)
        return bool(values.size and np.any(values >= 0))
    if base == "phase_id":
        values = np.asarray(arrays.get("phase_id", []), dtype=np.int32)
        return bool(values.size and np.any(values >= 0))
    raw_values = arrays.get("field_values", {}).get(base)
    return bool(raw_values and any(value is not None for value in raw_values))


def _default_color_field_from_arrays(arrays: dict[str, Any]) -> str:
    for field in COLOR_FIELD_DEFAULT_ORDER:
        if _color_field_present_in_arrays(arrays, field):
            return field
    return "grain_id"


def _normalize_color_field(arrays: dict[str, Any], color_by: str) -> str:
    base, _ = _split_color_by(color_by)
    if base in COLOR_FIELD_REQUIREMENTS and _color_field_present_in_arrays(arrays, color_by):
        return color_by
    return _default_color_field_from_arrays(arrays)


def _values_to_marker_colors(
    arrays: dict[str, Any],
    color_by: str,
) -> tuple[list[str] | np.ndarray, dict[str, Any]]:
    color_by = _normalize_color_field(arrays, color_by)
    if color_by == "grain_id":
        values = arrays["grain_id"]
        color_map = _categorical_color_map(values)
        return [color_map[int(value)] for value in values], {"showscale": False}
    if color_by == "phase_id":
        values = arrays["phase_id"]
        color_map = _categorical_color_map(values)
        return [color_map[int(value)] for value in values], {"showscale": False}

    raw_values = arrays["field_values"].get(color_by, [])
    rgb = _rgb_colors(raw_values)
    if rgb is not None:
        return rgb, {"showscale": False}
    numeric = _as_numeric_array(raw_values)
    return numeric, {"colorscale": "Viridis", "showscale": True}


def _values_to_marker_colors_with_scale(
    arrays: dict[str, Any],
    color_by: str,
    *,
    category_map: dict[int, str] | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    showscale: bool = False,
) -> tuple[list[str] | np.ndarray, dict[str, Any]]:
    color_by = _normalize_color_field(arrays, color_by)
    if color_by == "grain_id":
        values = arrays["grain_id"]
        color_map = category_map or _categorical_color_map(values)
        return [color_map[int(value)] for value in values], {"showscale": False}
    if color_by == "phase_id":
        values = arrays["phase_id"]
        color_map = category_map or _categorical_color_map(values)
        return [color_map[int(value)] for value in values], {"showscale": False}
    raw_values = arrays["field_values"].get(color_by, [])
    rgb = _rgb_colors(raw_values)
    if rgb is not None:
        return rgb, {"showscale": False}
    numeric = _as_numeric_array(raw_values)
    options: dict[str, Any] = {"colorscale": "Viridis", "showscale": showscale}
    if cmin is not None and cmax is not None and np.isfinite(cmin) and np.isfinite(cmax):
        options["cmin"] = float(cmin)
        options["cmax"] = float(cmax if not np.isclose(cmin, cmax) else cmin + 1.0)
    return numeric, options


def _numeric_colors(
    values: np.ndarray,
    *,
    cmin: float | None = None,
    cmax: float | None = None,
    colorscale: str = "Viridis",
) -> list[str]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ["rgb(160, 160, 160)"] * values.size
    lo = float(np.min(finite)) if cmin is None else float(cmin)
    hi = float(np.max(finite)) if cmax is None else float(cmax)
    if np.isclose(lo, hi):
        hi = lo + 1.0
    # As in v2, clip maps +/-inf onto the colorscale ends; only NaN stays
    # non-finite and falls back to grey.
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    sample_mask = np.isfinite(normalized)
    # One sample_colorscale call for all values; the per-value loop in v2
    # took ~30 s for 64k voxels because the colorscale is re-parsed per call.
    sampled = pc.sample_colorscale(colorscale, normalized[sample_mask].tolist())
    colors = ["rgb(160, 160, 160)"] * values.size
    for position, color in zip(np.flatnonzero(sample_mask), sampled):
        colors[position] = color
    return colors


def _values_to_color_strings(
    arrays: dict[str, Any],
    color_by: str,
    *,
    category_map: dict[int, str] | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    colorscale: str = DEFAULT_COLORMAP,
) -> list[str]:
    color_by = _normalize_color_field(arrays, color_by)
    base, reduction = _split_color_by(color_by)
    if base == "grain_id":
        values = arrays["grain_id"]
        color_map = category_map or _categorical_color_map(values)
        return [color_map[int(value)] for value in values]
    if base == "phase_id":
        values = arrays["phase_id"]
        color_map = category_map or _categorical_color_map(values)
        return [color_map[int(value)] for value in values]
    raw_values = arrays["field_values"].get(base, [])
    if reduction is None:
        rgb = _rgb_colors(raw_values)
        if rgb is not None:
            return rgb
    return _numeric_colors(
        _field_scalar_array(arrays, color_by),
        cmin=cmin,
        cmax=cmax,
        colorscale=colorscale,
    )


def _classify_color_field(arrays: dict[str, Any], color_by: str) -> str:
    """Return the render kind for a colour field: ``category``, ``rgb`` or ``numeric``."""
    resolved = _normalize_color_field(arrays, color_by)
    base, reduction = _split_color_by(resolved)
    if base in {"grain_id", "phase_id"}:
        return "category"
    if reduction is None:
        raw_values = arrays["field_values"].get(base, [])
        if _rgb_colors(raw_values) is not None:
            return "rgb"
    return "numeric"


def _add_voxel_mesh(
    fig: go.Figure,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    arrays: dict[str, Any],
    color_by: str,
    colorscale: str,
    category_map: dict[int, str] | None,
    cmin: float | None,
    cmax: float | None,
    hover_kwargs: dict[str, Any],
    name: str,
    row: int,
    col: int,
    show_colorbar: bool,
    colorbar: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add one voxel-cube Mesh3d, choosing facecolor (categorical/RGB) or a
    Plotly ``intensity`` scale that renders an Abaqus-style value colourbar.

    Returns the resolved numeric range ``{"cmin", "cmax"}`` for numeric fields so
    a shared colourbar can be reported, or ``{}`` otherwise.
    """
    resolved = _normalize_color_field(arrays, color_by)
    kind = _classify_color_field(arrays, resolved)
    base_kwargs = dict(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        flatshading=True,
        lighting={"ambient": 0.65, "diffuse": 0.75, "specular": 0.15},
        lightposition={"x": 100, "y": 200, "z": 100},
        name=name,
        showlegend=False,
    )

    if kind in {"category", "rgb"}:
        colors = _values_to_color_strings(
            arrays,
            resolved,
            category_map=category_map,
            cmin=cmin,
            cmax=cmax,
            colorscale=colorscale,
        )
        facecolors = [color for color in colors for _ in range(12)]
        fig.add_trace(
            go.Mesh3d(facecolor=facecolors, **base_kwargs, **hover_kwargs),
            row=row,
            col=col,
        )
        return {}

    values = _field_scalar_array(arrays, resolved)
    lo, hi = _resolve_color_range(values, cmin, cmax)
    # Per-face intensity (12 triangles per voxel). NaN voxels are pinned to the
    # low end so flatshading stays valid; fully-absent fields never reach here
    # because _normalize_color_field falls back to a present field first.
    intensity = np.repeat(np.nan_to_num(values, nan=lo), 12)
    mesh_kwargs: dict[str, Any] = dict(
        intensity=intensity,
        intensitymode="cell",
        colorscale=colorscale,
        cmin=lo,
        cmax=hi,
        showscale=bool(show_colorbar),
    )
    if show_colorbar and colorbar is not None:
        mesh_kwargs["colorbar"] = colorbar
    fig.add_trace(
        go.Mesh3d(**base_kwargs, **mesh_kwargs, **hover_kwargs),
        row=row,
        col=col,
    )
    return {"cmin": lo, "cmax": hi}


def _panel_colorbar(
    title: str,
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> dict[str, Any]:
    """A compact colourbar positioned at the right edge of one subplot panel."""
    x = min(1.0, col / cols + 0.005)
    y = 1.0 - (row - 0.5) / rows
    return {
        "title": {"text": title, "font": {"size": 10}},
        "x": x,
        "y": y,
        "len": max(0.12, 0.9 / rows),
        "thickness": 12,
        "tickfont": {"size": 8},
        "xanchor": "left",
    }


def _voxel_mesh_arrays(
    points: np.ndarray,
    colors: list[str],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vertices, faces = _voxel_mesh_geometry(points, voxel_size)
    facecolors = [color for color in colors for _ in range(12)]
    return vertices, faces, facecolors


def _voxel_mesh_geometry(
    points: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = float(voxel_size) / 2.0
    offsets = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    n_points = points.shape[0]
    vertices = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    base = (np.arange(n_points, dtype=np.int64) * 8)[:, None, None]
    faces = (triangles[None, :, :] + base).reshape(-1, 3)
    return vertices, faces


@lru_cache(maxsize=16)
def _voxel_mesh_geometry_cached(
    path: str,
    mtime_ns: int,
    size: int,
    snapshot_index: int,
    selected_grain_id: int | None,
    voxel_size_key: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _snapshot_arrays_cached(path, mtime_ns, size, snapshot_index)
    arrays_view = _filter_grain(arrays, selected_grain_id)
    points = arrays_view["voxel_index"]
    _check_voxel_cube_limit(points, f"snapshot {snapshot_index}")
    return _voxel_mesh_geometry(points.astype(np.float64), float(voxel_size_key))


def _mesh_geometry_for_snapshot(
    json_path: str | Path,
    snapshot_index: int,
    selected_grain_id: int | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _voxel_mesh_geometry_cached(
        *_file_cache_key(json_path),
        int(snapshot_index),
        selected_grain_id if selected_grain_id is None else int(selected_grain_id),
        round(float(voxel_size), 4),
    )


def _rve_continuity_value(
    data: dict[str, Any],
    rve_continuity: bool | None,
) -> bool:
    if "RVE_continuity" in data:
        value = data["RVE_continuity"]
        if not isinstance(value, bool):
            raise TypeError("RVE_continuity must be a JSON boolean true/false.")
        return value
    if rve_continuity is None:
        raise ValueError(
            "RVE_continuity is missing. Choose True or False before building "
            "the graph so periodic edges are handled explicitly."
        )
    return bool(rve_continuity)


def _check_voxel_cube_limit(points: np.ndarray, context: str) -> None:
    if points.size == 0:
        return
    extents = points.max(axis=0) - points.min(axis=0) + 1
    if points.shape[0] > MAX_CUBE_VOXELS or np.any(extents > MAX_CUBE_AXIS_CELLS):
        raise ValueError(
            f"{context} has {points.shape[0]} visible voxels with grid extents "
            f"{tuple(int(v) for v in extents)}. Cube rendering is limited to "
            f"{MAX_CUBE_AXIS_CELLS}x{MAX_CUBE_AXIS_CELLS}x{MAX_CUBE_AXIS_CELLS} "
            "to avoid freezing the browser. Select a grain/slice or use a "
            "smaller dataset."
        )


def _offsets_for_connectivity(connectivity: int) -> list[tuple[int, int, int]]:
    if connectivity not in CONNECTIVITY_OPTIONS:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    if connectivity == 6:
        return [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
    if connectivity == 18:
        return [
            (di, dj, dk)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            for dk in (-1, 0, 1)
            if 1 <= abs(di) + abs(dj) + abs(dk) <= 2
        ]
    return [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if not (di == 0 and dj == 0 and dk == 0)
    ]


# TODO(v3 audit, deferred): this per-voxel/per-offset Python loop measured
# 7.2 s (conn=6) and 29.4 s (conn=26) at 64k voxels. Planned fix: build a dense
# int lookup array grid[i, j, k] -> node index over the (<=50^3) extents, then
# compute all neighbors per offset in one vectorized step (modulo wrap when
# periodic) and deduplicate packed edge pairs with np.unique.
@lru_cache(maxsize=32)
def _fast_graph_cached(
    path: str,
    mtime_ns: int,
    size: int,
    snapshot_index: int,
    connectivity: int,
    rve_continuity: bool | None,
) -> dict[str, Any]:
    data = _load_json_cached(path, mtime_ns, size)
    arrays = _snapshot_arrays_cached(path, mtime_ns, size, snapshot_index)
    points = arrays["voxel_index"]
    periodic = _rve_continuity_value(data, rve_continuity)
    pos_to_idx = {tuple(map(int, pos)): idx for idx, pos in enumerate(points)}
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins + 1

    edge_to_periodic: dict[tuple[int, int], bool] = {}
    for pos_tuple, idx in pos_to_idx.items():
        i, j, k = pos_tuple
        for di, dj, dk in _offsets_for_connectivity(int(connectivity)):
            raw = np.asarray([i + di, j + dj, k + dk], dtype=np.int32)
            target = raw.copy()
            is_periodic = False
            if periodic:
                target = ((target - mins) % extents) + mins
                is_periodic = bool(np.any(target != raw))
            nb_idx = pos_to_idx.get(tuple(int(v) for v in target))
            if nb_idx is None or nb_idx == idx:
                continue
            a, b = (idx, nb_idx) if idx < nb_idx else (nb_idx, idx)
            key = (a, b)
            edge_to_periodic[key] = edge_to_periodic.get(key, False) or is_periodic

    if edge_to_periodic:
        edges = np.asarray(list(edge_to_periodic.keys()), dtype=np.int32)
        edge_is_periodic = np.asarray(list(edge_to_periodic.values()), dtype=bool)
    else:
        edges = np.zeros((0, 2), dtype=np.int32)
        edge_is_periodic = np.zeros((0,), dtype=bool)

    return {
        "voxel_index": points,
        "grain_id": arrays["grain_id"],
        "phase_id": arrays["phase_id"],
        "edges": edges,
        "edge_is_periodic": edge_is_periodic,
        "periodic": periodic,
        "connectivity": int(connectivity),
        "weights": None,
    }


def _fast_graph(
    json_path: str | Path,
    snapshot_index: int,
    connectivity: int,
    rve_continuity: bool | None,
) -> dict[str, Any]:
    return _fast_graph_cached(
        *_file_cache_key(json_path),
        int(snapshot_index),
        int(connectivity),
        rve_continuity,
    )


def _load_voxel_graph_backend(path: str):
    """
    Return the voxel-graph backend module.

    `voxel_graph` now lives next to this module in the shared `functions`
    package, so a normal package import is the primary route and works from
    any working directory. The old behaviour -- looking for a `voxel_graph.py`
    sitting beside `path` -- is kept only as a fallback for MiMeDO files that
    still ship their own copy.
    """
    try:
        from . import voxel_graph
        return voxel_graph
    except ImportError:
        pass

    graph_dir = str(Path(path).parent)
    if graph_dir not in sys.path:
        sys.path.insert(0, graph_dir)
    return importlib.import_module("voxel_graph")


@lru_cache(maxsize=16)
def _weighted_graph_cached(
    path: str,
    mtime_ns: int,
    size: int,
    snapshot_index: int,
    connectivity: int,
    rve_continuity: bool | None,
) -> dict[str, Any]:
    data = _load_json_cached(path, mtime_ns, size)
    _rve_continuity_value(data, rve_continuity)
    if "RVE_continuity" not in data:
        raise ValueError(
            "Weighted misorientation graph requires RVE_continuity to be stored "
            "in the JSON file. The topology graph can use the selected override."
        )
    backend = _load_voxel_graph_backend(path)
    builder = getattr(backend, "build_voxel_graph_cached", None)
    if builder is None:
        builder = getattr(backend, "build_voxel_graph")
    graph = builder(path, snapshot_index=int(snapshot_index), connectivity=int(connectivity))
    voxel_index = np.asarray(graph["voxel_index"], dtype=np.int32)
    edges = np.asarray(graph["edges"], dtype=np.int32)
    return {
        "voxel_index": voxel_index,
        "grain_id": np.asarray(graph["old_grain_id"], dtype=np.int32),
        "phase_id": np.asarray(graph["phase_id_per_voxel"], dtype=np.int32),
        "edges": edges,
        "edge_is_periodic": np.asarray(
            graph.get("edge_is_periodic", np.zeros(edges.shape[0], dtype=bool)),
            dtype=bool,
        ),
        "periodic": bool(graph.get("periodic", False)),
        "connectivity": int(connectivity),
        "weights": np.asarray(graph["weights"], dtype=float),
        "backend": "weighted_misorientation",
    }


def _graph_for_view(
    json_path: str | Path,
    snapshot_index: int,
    connectivity: int,
    rve_continuity: bool | None,
) -> tuple[dict[str, Any], str | None]:
    path, mtime_ns, size = _file_cache_key(json_path)
    data = _load_json_cached(path, mtime_ns, size)
    _rve_continuity_value(data, rve_continuity)
    try:
        return (
            _weighted_graph_cached(
                path,
                mtime_ns,
                size,
                int(snapshot_index),
                int(connectivity),
                rve_continuity,
            ),
            None,
        )
    except ModuleNotFoundError as exc:
        graph = _fast_graph(json_path, snapshot_index, connectivity, rve_continuity)
        graph["backend"] = "fast_topology"
        dependency = exc.name or "required graph dependency"
        return (
            graph,
            "Misorientation edge coloring requires the optional graph dependency "
            f"'{dependency}'. Install it (for this project: `pip install orix`) "
            "to show weighted edges and labels. Using topology-only edges for now.",
        )
    except ValueError as exc:
        if "RVE_continuity" not in data:
            graph = _fast_graph(json_path, snapshot_index, connectivity, rve_continuity)
            graph["backend"] = "fast_topology"
            return (
                graph,
                f"{exc} Using topology-only edges with the selected "
                f"RVE_continuity={bool(rve_continuity)}.",
            )
        raise


def _edge_line_coordinates(points: np.ndarray, edges: np.ndarray) -> tuple[Any, Any, Any]:
    if edges.size == 0:
        return [], [], []
    segments = points[edges]
    gap = np.full((segments.shape[0], 1, 3), np.nan)
    separated = np.concatenate([segments, gap], axis=1).reshape(-1, 3)
    return separated[:, 0], separated[:, 1], separated[:, 2]


def _weight_color_range(weights: np.ndarray) -> tuple[float, float] | None:
    finite = weights[np.isfinite(weights)]
    if finite.size == 0:
        return None
    cmin = float(np.min(finite))
    cmax = float(np.max(finite))
    if np.isclose(cmin, cmax):
        cmax = cmin + 1.0
    return cmin, cmax


def _edge_weight_bins(weights: np.ndarray, n_bins: int) -> list[tuple[np.ndarray, float | None]]:
    if weights.size == 0:
        return []
    finite = np.isfinite(weights)
    bins: list[tuple[np.ndarray, float | None]] = []
    finite_weights = weights[finite]
    if finite_weights.size:
        w_min = float(np.min(finite_weights))
        w_max = float(np.max(finite_weights))
        if np.isclose(w_min, w_max):
            bins.append((finite.copy(), w_min))
        else:
            edges = np.linspace(w_min, w_max, int(n_bins) + 1)
            for index in range(int(n_bins)):
                if index == int(n_bins) - 1:
                    mask = finite & (weights >= edges[index]) & (weights <= edges[index + 1])
                else:
                    mask = finite & (weights >= edges[index]) & (weights < edges[index + 1])
                if np.any(mask):
                    bins.append((mask, float((edges[index] + edges[index + 1]) / 2.0)))
    if np.any(~finite):
        bins.append((~finite, None))
    return bins


def _slice_graph(
    graph: dict[str, Any],
    slice_axis: str | None,
    slice_index: int | None,
) -> dict[str, Any]:
    if slice_axis is None:
        return graph
    axis = {"x": 0, "y": 1, "z": 2}.get(str(slice_axis).lower())
    if axis is None:
        raise ValueError("slice_axis must be one of x, y, z, or None")
    if slice_index is None:
        raise ValueError("slice_index is required for sliced graph scope")

    points = graph["voxel_index"]
    keep_nodes = points[:, axis] == int(slice_index)
    if not np.any(keep_nodes):
        lo = int(points[:, axis].min())
        hi = int(points[:, axis].max())
        raise ValueError(f"No nodes found for {slice_axis}={slice_index}; valid {lo}..{hi}")

    old_to_new = -np.ones(points.shape[0], dtype=np.int64)
    kept_old = np.where(keep_nodes)[0]
    old_to_new[kept_old] = np.arange(kept_old.size)
    edges = graph["edges"]
    edge_keep = keep_nodes[edges[:, 0]] & keep_nodes[edges[:, 1]]
    sliced = dict(graph)
    sliced["voxel_index"] = graph["voxel_index"][keep_nodes]
    sliced["grain_id"] = graph["grain_id"][keep_nodes]
    sliced["phase_id"] = graph["phase_id"][keep_nodes]
    sliced["edges"] = old_to_new[edges[edge_keep]].astype(np.int32)
    sliced["edge_is_periodic"] = graph["edge_is_periodic"][edge_keep]
    if isinstance(graph.get("weights"), np.ndarray):
        sliced["weights"] = graph["weights"][edge_keep]
    return sliced


def _filter_periodic_edges(graph: dict[str, Any], show_periodic_edges: bool) -> dict[str, Any]:
    if show_periodic_edges:
        return graph
    edge_is_periodic = graph.get("edge_is_periodic")
    if edge_is_periodic is None or not np.any(edge_is_periodic):
        return graph
    keep = ~edge_is_periodic
    filtered = dict(graph)
    filtered["edges"] = graph["edges"][keep]
    filtered["edge_is_periodic"] = edge_is_periodic[keep]
    if isinstance(graph.get("weights"), np.ndarray):
        filtered["weights"] = graph["weights"][keep]
    return filtered


def _filter_grain(
    arrays_or_graph: dict[str, Any],
    selected_grain_id: int | None,
) -> dict[str, Any]:
    if selected_grain_id is None:
        return arrays_or_graph
    keep = arrays_or_graph["grain_id"] == int(selected_grain_id)
    filtered = dict(arrays_or_graph)
    filtered["voxel_index"] = arrays_or_graph["voxel_index"][keep]
    filtered["grain_id"] = arrays_or_graph["grain_id"][keep]
    if "phase_id" in arrays_or_graph:
        filtered["phase_id"] = arrays_or_graph["phase_id"][keep]
    if "field_values" in arrays_or_graph:
        filtered["field_values"] = {
            field: np.asarray(values, dtype=object)[keep].tolist()
            for field, values in arrays_or_graph["field_values"].items()
        }
    if "edges" in arrays_or_graph:
        old_to_new = -np.ones(arrays_or_graph["grain_id"].shape[0], dtype=np.int64)
        kept_old = np.where(keep)[0]
        old_to_new[kept_old] = np.arange(kept_old.size)
        edges = arrays_or_graph["edges"]
        edge_keep = keep[edges[:, 0]] & keep[edges[:, 1]]
        filtered["edges"] = old_to_new[edges[edge_keep]].astype(np.int32)
        filtered["edge_is_periodic"] = arrays_or_graph["edge_is_periodic"][edge_keep]
        if isinstance(arrays_or_graph.get("weights"), np.ndarray):
            filtered["weights"] = arrays_or_graph["weights"][edge_keep]
    return filtered


def _scene_axis_layout(title: str, show_grid: bool) -> dict[str, Any]:
    return {
        "title": {"text": title if show_grid else ""},
        "visible": bool(show_grid),
        "showgrid": bool(show_grid),
        "zeroline": bool(show_grid),
        "showbackground": bool(show_grid),
        "showticklabels": bool(show_grid),
        "ticks": "outside" if show_grid else "",
        "gridcolor": "#dbe7f5",
        "zerolinecolor": "#9fb2ca",
        "backgroundcolor": "rgba(248, 250, 252, 0.50)"
        if show_grid
        else "rgba(0, 0, 0, 0)",
    }


def _scene_layout(title: str, *, show_grid: bool = True) -> dict[str, Any]:
    del title
    return {
        "xaxis": _scene_axis_layout("X / i", show_grid),
        "yaxis": _scene_axis_layout("Y / j", show_grid),
        "zaxis": _scene_axis_layout("Z / k", show_grid),
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}},
        "annotations": [],
        "domain": {},
    }


def _axis_bounds(*point_arrays: np.ndarray) -> list[list[float]]:
    usable = [
        np.asarray(points, dtype=float)
        for points in point_arrays
        if np.asarray(points).size and np.asarray(points).ndim == 2
    ]
    if not usable:
        return [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    stacked = np.vstack(usable + [np.array([[0.0, 0.0, 0.0]])])
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.06, 0.5)
    return [
        [float(mins[0] - pad[0]), float(maxs[0] + pad[0])],
        [float(mins[1] - pad[1]), float(maxs[1] + pad[1])],
        [float(mins[2] - pad[2]), float(maxs[2] + pad[2])],
    ]


def _add_origin_and_axes(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    bounds: list[list[float]],
    show_axes: bool = True,
    show_origin: bool = True,
    axis_width: int = 7,
    origin_size: int = 8,
) -> None:
    if not show_axes and not show_origin:
        return
    origin = np.array([0.0, 0.0, 0.0])
    endpoints = {
        "X": np.array([bounds[0][1], 0.0, 0.0]),
        "Y": np.array([0.0, bounds[1][1], 0.0]),
        "Z": np.array([0.0, 0.0, bounds[2][1]]),
    }
    colors = {"X": "red", "Y": "green", "Z": "blue"}

    if show_origin:
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers+text",
                marker={"size": origin_size, "color": "black", "symbol": "diamond"},
                text=["Origin<br>(0,0,0)"],
                textposition="top center",
                textfont={"size": 12, "color": "black"},
                hovertemplate="Origin (0,0,0)<extra></extra>",
                name="origin",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    if show_axes:
        for label, endpoint in endpoints.items():
            fig.add_trace(
                go.Scatter3d(
                    x=[origin[0], endpoint[0]],
                    y=[origin[1], endpoint[1]],
                    z=[origin[2], endpoint[2]],
                    mode="lines+text",
                    line={"color": colors[label], "width": axis_width},
                    text=["", label],
                    textposition="top center",
                    textfont={"size": 15, "color": colors[label]},
                    hoverinfo="skip",
                    name=f"{label} axis",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )


def build_microstructure_figure(
    json_path: str | Path,
    *,
    snapshot_index: int = 0,
    selected_grain_id: int | None = None,
    view_type: str = "rve",
    graph_scope: str = "entire",
    slice_axis: str | None = None,
    slice_index: int | None = None,
    connectivity: int = 6,
    show_periodic_edges: bool = True,
    show_edge_labels: bool = False,
    color_by: str = "grain_id",
    colormap: str = DEFAULT_COLORMAP,
    marker_size: float = 0.92,
    graph_node_size: int = 4,
    edge_width: int = 2,
    edge_color_bins: int = 18,
    show_axes: bool = True,
    show_origin: bool = True,
    show_grid: bool = True,
    detailed_hover: bool = MICROSTRUCTURE_DETAILED_HOVER_DEFAULT,
    rve_continuity: bool | None = None,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build a fast single-snapshot microstructure/graph figure."""
    if view_type not in VIEW_TYPES:
        raise ValueError("view_type must be one of rve, both, graph")
    if graph_scope not in GRAPH_SCOPES:
        raise ValueError("graph_scope must be entire or slice")

    state = build_control_state(
        json_path,
        plot="microstructure",
        snapshot_mode="one",
        view_type=view_type,
        graph_scope=graph_scope,
        snapshot_index=snapshot_index,
    )
    if not state["report"]["modules"]["microstructure"]:
        raise ValueError(state["report"]["microstructure"]["reason"])

    arrays = _snapshot_arrays(json_path, snapshot_index)
    arrays_view = _filter_grain(arrays, selected_grain_id)
    data = load_mimedo(json_path)
    subplot_titles = {
        "rve": ("RVE",),
        "graph": ("Equivalent graph",),
        "both": ("RVE", "Equivalent graph"),
    }[view_type]
    specs = [[{"type": "scene"}] * (2 if view_type == "both" else 1)]
    fig = make_subplots(
        rows=1,
        cols=2 if view_type == "both" else 1,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
    )

    warnings: list[str] = []
    rve_bounds = None
    graph_bounds = None
    graph_backend = None
    value_colorbar_shown = False
    if view_type in {"rve", "both"}:
        points = arrays_view["voxel_index"]
        _check_voxel_cube_limit(points, f"snapshot {snapshot_index}")
        rve_bounds = _axis_bounds(points)
        vertices, faces = _mesh_geometry_for_snapshot(
            json_path,
            snapshot_index,
            selected_grain_id,
            float(marker_size),
        )
        hover_kwargs = (
            {
                "text": [
                    f"voxel {i}<br>"
                    f"{f'grain {int(g)}<br>' if int(g) >= 0 else ''}"
                    f"index {tuple(idx)}"
                    for i, (g, idx) in enumerate(zip(arrays_view["grain_id"], points))
                    for _ in range(8)
                ],
                "hovertemplate": "%{text}<extra></extra>",
            }
            if detailed_hover
            else {"hoverinfo": "skip"}
        )
        resolved_color_by = _normalize_color_field(arrays_view, color_by)
        color_base, _ = _split_color_by(resolved_color_by)
        numeric_color = _classify_color_field(arrays_view, resolved_color_by) == "numeric"
        value_colorbar_shown = numeric_color
        # In "both" view the graph carries its own misorientation colourbar at
        # x=1.02, so the value colourbar is pushed further right to avoid overlap.
        colorbar = (
            {
                "title": {
                    "text": _colorbar_title(
                        resolved_color_by, _color_field_unit(data, color_base)
                    )
                },
                "x": 1.0 if view_type != "both" else 1.12,
                "len": 0.82,
                "thickness": 20,
            }
            if numeric_color
            else None
        )
        _add_voxel_mesh(
            fig,
            vertices=vertices,
            faces=faces,
            arrays=arrays_view,
            color_by=resolved_color_by,
            colorscale=colormap,
            category_map=None,
            cmin=None,
            cmax=None,
            hover_kwargs=hover_kwargs,
            name="RVE voxel cubes",
            row=1,
            col=1,
            show_colorbar=numeric_color,
            colorbar=colorbar,
        )

    if view_type in {"graph", "both"}:
        graph, graph_warning = _graph_for_view(
            json_path,
            snapshot_index,
            connectivity,
            rve_continuity,
        )
        if graph_warning:
            warnings.append(graph_warning)
        if graph_scope == "slice":
            graph = _slice_graph(graph, slice_axis or "z", slice_index)
        graph = _filter_periodic_edges(graph, show_periodic_edges)
        graph = _filter_grain(graph, selected_grain_id)
        graph_col = 2 if view_type == "both" else 1
        graph_points = graph["voxel_index"]
        graph_edges = graph["edges"]
        graph_backend = graph.get("backend", "weighted_misorientation")
        graph_bounds = _axis_bounds(graph_points)
        graph_weights = graph.get("weights")
        has_weights = (
            isinstance(graph_weights, np.ndarray)
            and graph_weights.shape[0] == graph_edges.shape[0]
        )
        weight_range = _weight_color_range(graph_weights) if has_weights else None
        if has_weights and weight_range is not None and graph_edges.size:
            cmin, cmax = weight_range
            span = cmax - cmin if not np.isclose(cmin, cmax) else 1.0
            for mask, representative_weight in _edge_weight_bins(
                graph_weights, edge_color_bins
            ):
                x_edge, y_edge, z_edge = _edge_line_coordinates(
                    graph_points, graph_edges[mask]
                )
                if representative_weight is None:
                    color = "rgba(70, 70, 70, 0.75)"
                else:
                    normalized = float(np.clip((representative_weight - cmin) / span, 0.0, 1.0))
                    color = pc.sample_colorscale("Viridis", [normalized])[0]
                fig.add_trace(
                    go.Scatter3d(
                        x=x_edge,
                        y=y_edge,
                        z=z_edge,
                        mode="lines",
                        line={"color": color, "width": edge_width},
                        hoverinfo="skip",
                        name="Graph edges",
                        showlegend=False,
                    ),
                    row=1,
                    col=graph_col,
                )
            fig.add_trace(
                go.Scatter3d(
                    x=[None, None],
                    y=[None, None],
                    z=[None, None],
                    mode="markers",
                    marker={
                        "size": 0.1,
                        "color": [cmin, cmax],
                        "colorscale": "Viridis",
                        "cmin": cmin,
                        "cmax": cmax,
                        "opacity": 0.0,
                        "colorbar": {
                            "title": "misorientation<br>(deg)",
                            "x": 1.02,
                            "len": 0.82,
                            "thickness": 24,
                        },
                    },
                    hoverinfo="skip",
                    name="misorientation scale",
                    showlegend=False,
                ),
                row=1,
                col=graph_col,
            )
        else:
            x_edge, y_edge, z_edge = _edge_line_coordinates(graph_points, graph_edges)
            fig.add_trace(
                go.Scatter3d(
                    x=x_edge,
                    y=y_edge,
                    z=z_edge,
                    mode="lines",
                    line={"color": "#4f6f8f", "width": edge_width},
                    hoverinfo="skip",
                    name="Graph edges",
                    showlegend=False,
                ),
                row=1,
                col=graph_col,
            )
        graph_color_map = _categorical_color_map(graph["grain_id"])
        graph_colors = [graph_color_map[int(value)] for value in graph["grain_id"]]
        fig.add_trace(
            go.Scatter3d(
                x=graph_points[:, 0],
                y=graph_points[:, 1],
                z=graph_points[:, 2],
                mode="markers",
                marker={"size": graph_node_size, "color": graph_colors, "opacity": 0.9},
                text=[
                    f"node {i}<br>"
                    f"{f'grain {int(g)}<br>' if int(g) >= 0 else ''}"
                    f"index {tuple(idx)}"
                    for i, (g, idx) in enumerate(zip(graph["grain_id"], graph_points))
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Graph nodes",
                showlegend=False,
            ),
            row=1,
            col=graph_col,
        )
        if show_edge_labels:
            if has_weights and graph_edges.shape[0] <= 2500:
                midpoints = (
                    graph_points[graph_edges[:, 0]] + graph_points[graph_edges[:, 1]]
                ) / 2.0
                labels = [
                    f"{float(weight):.1f} deg" if np.isfinite(weight) else "inf"
                    for weight in graph_weights
                ]
                fig.add_trace(
                    go.Scatter3d(
                        x=midpoints[:, 0],
                        y=midpoints[:, 1],
                        z=midpoints[:, 2],
                        mode="text",
                        text=labels,
                        textfont={"size": 10, "color": "black"},
                        hoverinfo="skip",
                        name="edge labels",
                        showlegend=False,
                    ),
                    row=1,
                    col=graph_col,
                )
            elif has_weights:
                warnings.append(
                    "Edge labels are hidden because the visible graph has more than "
                    "2500 edges."
                )
            else:
                warnings.append(
                    "Edge labels need misorientation weights; using topology-only graph."
                )

    if show_axes or show_origin:
        if view_type in {"rve", "both"}:
            _add_origin_and_axes(
                fig,
                row=1,
                col=1,
                bounds=rve_bounds or _axis_bounds(arrays_view["voxel_index"]),
                show_axes=show_axes,
                show_origin=show_origin,
            )
        if view_type == "both":
            _add_origin_and_axes(
                fig,
                row=1,
                col=2,
                bounds=graph_bounds or _axis_bounds(arrays_view["voxel_index"]),
                show_axes=show_axes,
                show_origin=show_origin,
            )
        elif view_type == "graph":
            _add_origin_and_axes(
                fig,
                row=1,
                col=1,
                bounds=graph_bounds or _axis_bounds(arrays_view["voxel_index"]),
                show_axes=show_axes,
                show_origin=show_origin,
            )

    title_bits = [f"snapshot {snapshot_index}"]
    if arrays.get("time") is not None:
        title_bits.append(f"time={arrays['time']}")
    if selected_grain_id is not None:
        title_bits.append(f"grain {int(selected_grain_id)}")
    if view_type in {"graph", "both"}:
        title_bits.append(f"connectivity={int(connectivity)}")
    fig.update_layout(
        template="plotly_white",
        height=760,
        margin={"l": 0, "r": 90 if value_colorbar_shown else 0, "t": 80, "b": 0},
        title=" | ".join(title_bits),
        scene=_scene_layout("scene", show_grid=show_grid),
    )
    if view_type == "both":
        fig.update_layout(scene2=_scene_layout("scene2", show_grid=show_grid))

    metadata = {
        "control_state": state,
        "warnings": warnings,
        "backend": graph_backend or "rve_only",
    }
    return fig, metadata


def _snapshot_selection(
    snapshot_count: int,
    start_snapshot: int,
    end_snapshot: int | None,
    count: int,
) -> list[int]:
    start = max(0, min(int(start_snapshot), snapshot_count - 1))
    end = snapshot_count - 1 if end_snapshot is None else int(end_snapshot)
    end = max(start, min(end, snapshot_count - 1))
    count = max(1, int(count))
    if count >= end - start + 1:
        return list(range(start, end + 1))
    selected = np.linspace(start, end, count)
    return sorted({int(round(value)) for value in selected})


def _validate_microstructure_snapshot_indices(
    json_path: str | Path,
    snapshot_indices: list[int] | tuple[int, ...],
    *,
    max_count: int | None = None,
) -> list[int]:
    report = inspect_mimedo(json_path)
    if not report["modules"]["microstructure"]:
        raise ValueError(report["microstructure"]["reason"])
    snapshot_total = int(report["microstructure"]["snapshot_count"])
    normalized: list[int] = []
    for snapshot_index in snapshot_indices:
        value = int(snapshot_index)
        if not (0 <= value < snapshot_total):
            raise ValueError(
                f"snapshot {value} is outside the available range 0..{snapshot_total - 1}."
            )
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("Select at least one snapshot.")
    if max_count is not None and len(normalized) > max_count:
        raise ValueError(
            f"Multi-snapshot microstructure views support at most {max_count} "
            f"snapshots; selected {len(normalized)}."
        )
    return normalized


def _grid_shape(panel_count: int, layout: str) -> tuple[int, int]:
    if layout == "row":
        return 1, panel_count
    cols = int(np.ceil(np.sqrt(panel_count)))
    rows = int(np.ceil(panel_count / cols))
    return rows, cols


def _available_color_values(json_path: str | Path, snapshot_index: int) -> set[str]:
    return {
        value
        for _, value in _color_options(json_path, snapshot_index)
        if _split_color_by(value)[0] in COLOR_FIELD_REQUIREMENTS
    }


def _default_compare_colors(json_path: str | Path, snapshot_index: int = 0) -> list[str]:
    ordered_values = [value for _, value in _color_options(json_path, snapshot_index)]
    if ordered_values:
        return ordered_values[:2]
    return ["grain_id"]


def _validate_compare_colors(
    json_path: str | Path,
    snapshots: list[int],
    color_by_rows: list[str],
) -> list[str]:
    if not color_by_rows:
        raise ValueError("Select at least one comparison color mode.")
    normalized = []
    for field in color_by_rows:
        if _split_color_by(field)[0] in COLOR_FIELD_REQUIREMENTS and field not in normalized:
            normalized.append(field)
    if not normalized:
        raise ValueError("No valid comparison color modes were selected.")

    missing: dict[str, list[int]] = {}
    for snapshot_index in snapshots:
        available = _available_color_values(json_path, snapshot_index)
        for field in normalized:
            if field not in available:
                missing.setdefault(field, []).append(snapshot_index)
    if missing:
        details = "; ".join(
            f"{field} missing in snapshots {indexes}" for field, indexes in missing.items()
        )
        raise ValueError(f"Cannot compare with unavailable color fields: {details}.")
    return normalized


def _comparison_color_scale_state(
    arrays_by_snapshot: dict[int, dict[str, Any]],
    color_by: str,
    same_color_scale: bool,
) -> tuple[dict[int, str] | None, float | None, float | None]:
    if not same_color_scale:
        return None, None, None
    base, _ = _split_color_by(color_by)
    if base in {"grain_id", "phase_id"}:
        values = np.concatenate(
            [arrays[base] for arrays in arrays_by_snapshot.values()]
        )
        return _categorical_color_map(values), None, None
    if base == "IPFcolor_(1 0 0)":
        return None, None, None

    numeric_arrays = [
        _field_scalar_array(arrays, color_by)
        for arrays in arrays_by_snapshot.values()
    ]
    finite_arrays = [arr[np.isfinite(arr)] for arr in numeric_arrays if arr.size]
    finite = np.concatenate(finite_arrays) if finite_arrays else np.array([])
    if finite.size:
        return None, float(np.min(finite)), float(np.max(finite))
    return None, None, None


def build_microstructure_evolution_figure(
    json_path: str | Path,
    *,
    snapshot_indices: list[int] | tuple[int, ...] | None = None,
    start_snapshot: int = 0,
    end_snapshot: int | None = None,
    count: int = 6,
    layout: str = "grid",
    selected_grain_id: int | None = None,
    color_by: str = "grain_id",
    colormap: str = DEFAULT_COLORMAP,
    same_camera: bool = True,
    same_color_scale: bool = True,
    marker_size: float = 0.92,
    show_axes: bool = True,
    show_origin: bool = True,
    show_grid: bool = True,
    detailed_hover: bool = MICROSTRUCTURE_DETAILED_HOVER_DEFAULT,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build a multiple-snapshot RVE-only evolution figure."""
    report = inspect_mimedo(json_path)
    if not report["modules"]["microstructure"]:
        raise ValueError(report["microstructure"]["reason"])
    snapshot_count = int(report["microstructure"]["snapshot_count"])
    selected = (
        _validate_microstructure_snapshot_indices(
            json_path,
            list(snapshot_indices),
            max_count=MICROSTRUCTURE_MAX_SNAPSHOT_PANELS,
        )
        if snapshot_indices is not None
        else _snapshot_selection(snapshot_count, start_snapshot, end_snapshot, count)
    )
    if len(selected) > MICROSTRUCTURE_MAX_SNAPSHOT_PANELS:
        raise ValueError(
            f"Multi-snapshot microstructure views support at most "
            f"{MICROSTRUCTURE_MAX_SNAPSHOT_PANELS} snapshots; selected {len(selected)}."
        )
    rows, cols = _grid_shape(len(selected), layout)
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]

    arrays_by_snapshot = {
        snapshot_index: _filter_grain(_snapshot_arrays(json_path, snapshot_index), selected_grain_id)
        for snapshot_index in selected
    }
    data = load_mimedo(json_path)
    color_base, _ = _split_color_by(color_by)
    field_unit = _color_field_unit(data, color_base)
    resolved_first = _normalize_color_field(arrays_by_snapshot[selected[0]], color_by)
    is_numeric = (
        _classify_color_field(arrays_by_snapshot[selected[0]], resolved_first) == "numeric"
    )
    category_map = None
    cmin = cmax = None
    if same_color_scale and not is_numeric and color_base in {"grain_id", "phase_id"}:
        values = np.concatenate(
            [arrays[color_base] for arrays in arrays_by_snapshot.values()]
        )
        category_map = _categorical_color_map(values)
    elif same_color_scale and is_numeric:
        numeric_arrays = [
            _field_scalar_array(arrays, color_by)
            for arrays in arrays_by_snapshot.values()
        ]
        finite_parts = [arr[np.isfinite(arr)] for arr in numeric_arrays if arr.size]
        finite = np.concatenate(finite_parts) if finite_parts else np.array([])
        if finite.size:
            cmin = float(np.min(finite))
            cmax = float(np.max(finite))

    titles = []
    for snapshot_index in selected:
        arrays = arrays_by_snapshot[snapshot_index]
        if arrays.get("time") is None:
            titles.append(f"snapshot {snapshot_index}")
        else:
            titles.append(f"snapshot {snapshot_index}, time={arrays['time']}")
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    shared_colorbar_shown = False
    for index, snapshot_index in enumerate(selected):
        row = index // cols + 1
        col = index % cols + 1
        arrays = arrays_by_snapshot[snapshot_index]
        points = arrays["voxel_index"]
        _check_voxel_cube_limit(points, f"snapshot {snapshot_index}")
        vertices, faces = _mesh_geometry_for_snapshot(
            json_path,
            snapshot_index,
            selected_grain_id,
            float(marker_size),
        )
        hover_kwargs = (
            {
                "text": [
                    f"snapshot {snapshot_index}<br>voxel {i}<br>"
                    f"{f'grain {int(g)}<br>' if int(g) >= 0 else ''}"
                    f"index {tuple(idx)}"
                    for i, (g, idx) in enumerate(zip(arrays["grain_id"], points))
                    for _ in range(8)
                ],
                "hovertemplate": "%{text}<extra></extra>",
            }
            if detailed_hover
            else {"hoverinfo": "skip"}
        )
        show_cb = False
        colorbar = None
        if is_numeric:
            title = _colorbar_title(color_by, field_unit)
            if same_color_scale:
                # One shared colourbar (global range) placed on the right.
                if not shared_colorbar_shown:
                    show_cb = True
                    colorbar = {
                        "title": {"text": title},
                        "x": 1.0,
                        "len": 0.9,
                        "thickness": 18,
                    }
                    shared_colorbar_shown = True
            else:
                # Per-snapshot self-scaling: a compact colourbar per panel.
                show_cb = True
                colorbar = _panel_colorbar(title, row, col, rows, cols)
        _add_voxel_mesh(
            fig,
            vertices=vertices,
            faces=faces,
            arrays=arrays,
            color_by=color_by,
            colorscale=colormap,
            category_map=category_map,
            cmin=cmin,
            cmax=cmax,
            hover_kwargs=hover_kwargs,
            name=f"snapshot {snapshot_index}",
            row=row,
            col=col,
            show_colorbar=show_cb,
            colorbar=colorbar,
        )
        if show_axes or show_origin:
            _add_origin_and_axes(
                fig,
                row=row,
                col=col,
                bounds=_axis_bounds(points),
                show_axes=show_axes,
                show_origin=show_origin,
            )

    camera = {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}}
    layout_updates: dict[str, Any] = {}
    for index in range(len(selected)):
        scene_name = "scene" if index == 0 else f"scene{index + 1}"
        layout_updates[scene_name] = _scene_layout(
            scene_name,
            show_grid=show_grid,
        )
        if same_camera:
            layout_updates[scene_name]["camera"] = camera

    fig.update_layout(
        template="plotly_white",
        height=max(520, rows * 360),
        margin={"l": 0, "r": 80 if is_numeric else 0, "t": 90, "b": 0},
        title="Microstructure evolution",
        **layout_updates,
    )
    state = build_control_state(
        json_path,
        plot="microstructure",
        snapshot_mode="multiple",
        view_type="rve",
    )
    return fig, {
        "control_state": state,
        "warnings": [],
        "snapshots": selected,
        "backend": "fast_rve_evolution",
    }


def build_microstructure_comparison_figure(
    json_path: str | Path,
    *,
    snapshot_indices: list[int] | tuple[int, ...] | None = None,
    start_snapshot: int = 0,
    end_snapshot: int | None = None,
    count: int = 4,
    color_by_rows: list[str] | tuple[str, ...] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    selected_grain_id: int | None = None,
    same_camera: bool = True,
    same_color_scale: bool = True,
    marker_size: float = 0.92,
    show_axes: bool = True,
    show_origin: bool = True,
    show_grid: bool = True,
    detailed_hover: bool = MICROSTRUCTURE_DETAILED_HOVER_DEFAULT,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build an RVE-only comparison grid with colors as rows and snapshots as columns.

    The panel count is capped by ``COMPARISON_MAX_PANELS``. This keeps notebook
    interaction responsive and avoids silently dropping selected snapshots or
    color rows.
    """
    report = inspect_mimedo(json_path)
    if not report["modules"]["microstructure"]:
        raise ValueError(report["microstructure"]["reason"])
    snapshot_total = int(report["microstructure"]["snapshot_count"])
    selected = (
        _validate_microstructure_snapshot_indices(json_path, list(snapshot_indices))
        if snapshot_indices is not None
        else _snapshot_selection(snapshot_total, start_snapshot, end_snapshot, count)
    )
    requested_colors = list(color_by_rows) if color_by_rows else _default_compare_colors(json_path, selected[0])
    color_rows = _validate_compare_colors(json_path, selected, requested_colors)
    panel_count = len(selected) * len(color_rows)
    if panel_count > COMPARISON_MAX_PANELS:
        raise ValueError(
            f"Comparison would create {panel_count} panels "
            f"({len(selected)} snapshots x {len(color_rows)} color rows). "
            f"The limit is {COMPARISON_MAX_PANELS}; reduce snapshot count or "
            "selected color modes."
        )

    rows = len(color_rows)
    cols = len(selected)
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = []
    for color_by in color_rows:
        for snapshot_index in selected:
            arrays = _snapshot_arrays(json_path, snapshot_index)
            time_text = "" if arrays.get("time") is None else f", time={arrays['time']}"
            subplot_titles.append(f"{color_by}<br>snapshot {snapshot_index}{time_text}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.015,
        vertical_spacing=0.055,
    )

    arrays_by_snapshot = {
        snapshot_index: _filter_grain(_snapshot_arrays(json_path, snapshot_index), selected_grain_id)
        for snapshot_index in selected
    }
    mesh_by_snapshot = {
        snapshot_index: _mesh_geometry_for_snapshot(
            json_path,
            snapshot_index,
            selected_grain_id,
            marker_size,
        )
        for snapshot_index in selected
    }
    hover_by_snapshot = (
        {
            snapshot_index: [
                f"snapshot {snapshot_index}<br>voxel {i}<br>"
                f"{f'grain {int(g)}<br>' if int(g) >= 0 else ''}"
                f"index {tuple(idx)}"
                for i, (g, idx) in enumerate(
                    zip(
                        arrays_by_snapshot[snapshot_index]["grain_id"],
                        arrays_by_snapshot[snapshot_index]["voxel_index"],
                    )
                )
                for _ in range(8)
            ]
            for snapshot_index in selected
        }
        if detailed_hover
        else {}
    )

    data = load_mimedo(json_path)
    any_numeric_row = False
    for row_index, color_by in enumerate(color_rows, start=1):
        category_map, cmin, cmax = _comparison_color_scale_state(
            arrays_by_snapshot,
            color_by,
            same_color_scale,
        )
        row_base, _ = _split_color_by(color_by)
        row_unit = _color_field_unit(data, row_base)
        row_resolved = _normalize_color_field(arrays_by_snapshot[selected[0]], color_by)
        row_is_numeric = (
            _classify_color_field(arrays_by_snapshot[selected[0]], row_resolved) == "numeric"
        )
        any_numeric_row = any_numeric_row or row_is_numeric
        for col_index, snapshot_index in enumerate(selected, start=1):
            arrays = arrays_by_snapshot[snapshot_index]
            points = arrays["voxel_index"]
            vertices, faces = mesh_by_snapshot[snapshot_index]
            hover_kwargs = (
                {
                    "text": hover_by_snapshot[snapshot_index],
                    "hovertemplate": "%{text}<extra></extra>",
                }
                if detailed_hover
                else {"hoverinfo": "skip"}
            )
            # One colourbar per row, on the last column, when the row is numeric.
            show_cb = row_is_numeric and col_index == cols
            colorbar = (
                _panel_colorbar(
                    _colorbar_title(color_by, row_unit),
                    row_index,
                    cols,
                    rows,
                    cols,
                )
                if show_cb
                else None
            )
            _add_voxel_mesh(
                fig,
                vertices=vertices,
                faces=faces,
                arrays=arrays,
                color_by=color_by,
                colorscale=colormap,
                category_map=category_map,
                cmin=cmin,
                cmax=cmax,
                hover_kwargs=hover_kwargs,
                name=f"{color_by} snapshot {snapshot_index}",
                row=row_index,
                col=col_index,
                show_colorbar=show_cb,
                colorbar=colorbar,
            )
            if show_axes or show_origin:
                _add_origin_and_axes(
                    fig,
                    row=row_index,
                    col=col_index,
                    bounds=_axis_bounds(points),
                    show_axes=show_axes,
                    show_origin=show_origin,
                    axis_width=6,
                    origin_size=7,
                )

    camera = {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}}
    layout_updates: dict[str, Any] = {}
    for index in range(panel_count):
        scene_name = "scene" if index == 0 else f"scene{index + 1}"
        layout_updates[scene_name] = _scene_layout(scene_name, show_grid=show_grid)
        if same_camera:
            layout_updates[scene_name]["camera"] = camera

    state = build_control_state(
        json_path,
        plot="microstructure",
        snapshot_mode="compare",
        view_type="rve",
    )
    fig.update_layout(
        template="plotly_white",
        height=max(560, rows * 390),
        margin={"l": 0, "r": 70 if any_numeric_row else 0, "t": 96, "b": 0},
        title=(
            f"Microstructure comparison | {len(selected)} snapshots x "
            f"{len(color_rows)} color rows"
        ),
        **layout_updates,
    )
    return fig, {
        "control_state": state,
        "warnings": [],
        "snapshots": selected,
        "color_rows": color_rows,
        "backend": "fast_rve_comparison",
    }


def _load_texture_dependencies() -> tuple[Any, Any, Any]:
    try:
        import matplotlib.pyplot as plt
        from orix import plot as _orix_plot  # noqa: F401 - registers projections
        from orix.crystal_map import Phase
        from orix.quaternion import Orientation
        from orix.vector import Miller
    except ModuleNotFoundError as exc:
        dependency = exc.name or "texture plotting dependency"
        raise ModuleNotFoundError(
            "Texture plotting requires `orix` and `matplotlib` in the active "
            f"Python environment. Missing dependency: {dependency}. Install with "
            "`pip install orix matplotlib`, then restart the kernel."
        ) from exc
    return plt, Orientation, Miller, Phase


def _texture_phase(data: dict[str, Any], Phase: Any) -> Any:
    phases = _phase_records(data)
    if not phases:
        raise ValueError("Texture plotting requires at least one phase in the JSON.")
    phase_data = phases[0]
    phase_name = phase_data.get("phase_name") or phase_data.get("name") or "phase"
    symmetry_name = _phase_symmetry_name(data)
    if not symmetry_name:
        raise ValueError("Texture plotting requires phase orientation.crystal_symmetry_group.")
    return Phase(name=str(phase_name), point_group=str(symmetry_name))


def _orientation_sample_indices(total: int, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or total <= max_points:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, int(max_points), dtype=np.int64)


@lru_cache(maxsize=24)
def _texture_payload_cached(
    path: str,
    mtime_ns: int,
    size: int,
    snapshot_index: int,
    max_points: int | None,
) -> dict[str, Any]:
    del mtime_ns, size
    _, Orientation, _, Phase = _load_texture_dependencies()
    data = _load_json_cached(*_file_cache_key(path))
    snapshot = _snapshot(data, snapshot_index)
    if not _is_texture_reference_snapshot(snapshot):
        statuses = " or ".join(TEXTURE_REFERENCE_STATUSES)
        raise ValueError(f"snapshot {snapshot_index} is not {statuses}.")
    voxels = snapshot.get("voxels")
    if not isinstance(voxels, list) or not voxels:
        raise ValueError(f"snapshot {snapshot_index} has no voxel data.")
    eulers = np.asarray([voxel.get("orientation") for voxel in voxels], dtype=float)
    if eulers.ndim != 2 or eulers.shape[1] != 3:
        raise ValueError(
            f"snapshot {snapshot_index} voxel orientations must be Nx3 Euler angles in radians."
        )
    indices = _orientation_sample_indices(eulers.shape[0], max_points)
    phase = _texture_phase(data, Phase)
    point_group = getattr(phase, "point_group", None)
    orientation_symmetry = getattr(point_group, "laue", point_group)
    orientations = Orientation.from_euler(
        eulers[indices],
        symmetry=orientation_symmetry,
        degrees=False,
    )
    return {
        "O": orientations,
        "phase": phase,
        "snapshot_index": int(snapshot_index),
        "time": snapshot.get("time_point"),
        "status": _snapshot_grid_status(snapshot),
        "n_total": int(eulers.shape[0]),
        "n_used": int(indices.size),
        "sampled": bool(indices.size != eulers.shape[0]),
    }


def _texture_payload(
    json_path: str | Path,
    snapshot_index: int,
    max_points: int | None,
) -> dict[str, Any]:
    return _texture_payload_cached(
        *_file_cache_key(json_path),
        int(snapshot_index),
        None if max_points is None else int(max_points),
    )


def _texture_snapshot_options(json_path: str | Path) -> list[tuple[str, int]]:
    report = inspect_mimedo(json_path)
    options = []
    for item in report["texture"].get("undeformed_snapshots", []):
        label = f"snapshot {item['index']}"
        if item.get("time") is not None:
            label += f" | time={item['time']}"
        label += " | undeformed"
        voxel_count = item.get("voxel_count")
        if voxel_count is not None:
            label += f" | voxels={voxel_count}"
        options.append((label, int(item["index"])))
    return options


def _default_texture_snapshots(json_path: str | Path) -> tuple[int, ...]:
    values = [value for _, value in _texture_snapshot_options(json_path)]
    if len(values) <= TEXTURE_MAX_SNAPSHOTS:
        return tuple(values)
    middle = values[len(values) // 2]
    selected = [values[0], middle, values[-1]]
    return tuple(dict.fromkeys(selected))


def _validate_texture_snapshots(json_path: str | Path, snapshot_indices: list[int]) -> list[int]:
    available = {value for _, value in _texture_snapshot_options(json_path)}
    selected = []
    for snapshot_index in snapshot_indices:
        value = int(snapshot_index)
        if value not in selected:
            selected.append(value)
    if not selected:
        raise ValueError("Select at least one undeformed texture snapshot.")
    unavailable = [value for value in selected if value not in available]
    if unavailable:
        raise ValueError(
            "Texture plotting only supports undeformed snapshots with voxel orientations; "
            f"invalid selection: {unavailable}."
        )
    return selected


def _texture_marker_style(orientation_count: int) -> tuple[float, float]:
    if orientation_count < 1:
        return 5.0, 0.4
    scale = 1.0 / np.sqrt(float(orientation_count))
    return float(np.clip(250 * scale, 0.2, 18)), float(np.clip(4 * scale, 0.04, 0.45))


def build_texture_figure(
    json_path: str | Path,
    *,
    snapshot_indices: list[int] | tuple[int, ...] | None = None,
    texture_view: str = "pf",
    max_points: int | None = TEXTURE_MAX_POINTS,
) -> tuple[Any, dict[str, Any]]:
    """Build a PF-only or PDF-only texture comparison from undeformed voxel orientations."""
    if texture_view not in TEXTURE_VIEWS:
        raise ValueError("texture_view must be 'pf' or 'pdf'")
    report = inspect_mimedo(json_path)
    if not report["modules"]["texture"]:
        raise ValueError(report["texture"]["reason"])
    selected = _validate_texture_snapshots(
        json_path,
        list(snapshot_indices) if snapshot_indices is not None else list(_default_texture_snapshots(json_path)),
    )
    plt, _, Miller, _ = _load_texture_dependencies()
    payloads = [
        _texture_payload(json_path, snapshot_index, max_points)
        for snapshot_index in selected
    ]
    rows = len(TEXTURE_POLE_FAMILIES)
    cols = len(payloads)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.8 * cols, 4.2 * rows),
        subplot_kw={"projection": "stereographic"},
        squeeze=False,
    )
    warnings: list[str] = []
    for payload in payloads:
        if payload["sampled"]:
            warnings.append(
                f"snapshot {payload['snapshot_index']} uses {payload['n_used']} of "
                f"{payload['n_total']} voxel orientations for texture plotting."
            )

    for row, (family_label, vector) in enumerate(TEXTURE_POLE_FAMILIES.items()):
        for col, payload in enumerate(payloads):
            ax = axes[row][col]
            phase = payload["phase"]
            orientations = payload["O"]
            miller = Miller(uvw=list(vector), phase=phase).symmetrise(unique=True)
            poles = orientations.inv().outer(miller)
            if texture_view == "pf":
                size, alpha = _texture_marker_style(int(payload["n_used"]))
                ax.scatter(poles, s=size, alpha=alpha)
                title_kind = "PF"
            else:
                ax.pole_density_function(poles)
                title_kind = "PDF"
            ax.set_labels("X", "Y", None)
            time_text = "" if payload["time"] is None else f" | time={payload['time']}"
            ax.set_title(
                f"{family_label} {title_kind}\n"
                f"snapshot {payload['snapshot_index']}{time_text}",
                fontsize=10,
            )
    view_title = "Pole figures" if texture_view == "pf" else "Pole density functions"
    fig.suptitle(
        f"{view_title} from undeformed voxel orientations",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    state = build_control_state(json_path, plot="texture")
    return fig, {
        "control_state": state,
        "warnings": warnings,
        "snapshots": selected,
        "pole_families": list(TEXTURE_POLE_FAMILIES.keys()),
        "backend": "orix_texture_voxels",
    }


def _normal_snapshot_index(data: dict[str, Any], snapshot_index: int) -> int:
    micro = data.get("microstructure")
    if not _is_snapshot_array(micro):
        raise ValueError("microstructure is not a valid snapshot array")
    index = int(snapshot_index)
    if index < 0:
        index += len(micro)
    if not (0 <= index < len(micro)):
        raise IndexError(f"snapshot_index={snapshot_index} out of range [0, {len(micro)})")
    return index


def _length_unit_scale_to_um(length_unit: Any) -> float:
    if length_unit is None:
        raise KeyError("Missing data['units']['Length']; cannot convert statistics to um.")
    unit = str(length_unit).strip().lower().replace("micro", "u")
    unit = unit.replace("micron", "um").replace("microns", "um")
    unit = "".join(unit.split())
    if unit in {"m", "meter", "meters", "metre", "metres"}:
        return 1e6
    if unit in {"um", "micrometer", "micrometers", "micrometre", "micrometres"}:
        return 1.0
    raise ValueError(f"Unsupported length unit for statistics: {length_unit!r}")


@dataclass
class _StatsVoxelMesh:
    nodes: np.ndarray
    voxel_dict: dict[int, list[int]]
    grain_dict: dict[int, list[int]]
    grain_phase_dict: dict[int, int]
    voxel_centers: dict[int, np.ndarray]
    voxel_spacing: np.ndarray

    @staticmethod
    def _grid_from_snapshot(snapshot: Mapping[str, Any]) -> tuple[np.ndarray, int, int, int]:
        grid = snapshot.get("grid")
        if not isinstance(grid, Mapping):
            raise KeyError("Snapshot is missing required key: grid.")
        status = str(grid.get("status", "")).strip().lower()
        if status != "undeformed":
            raise RuntimeError(
                "Statistics mesh creation requires an undeformed regular grid, "
                f"but got grid.status={grid.get('status')!r}."
            )
        size = np.asarray(grid.get("grid_size"), dtype=float)
        spacing = np.asarray(grid.get("grid_spacing"), dtype=float)
        if size.shape != (3,):
            raise ValueError(f"Expected grid_size length 3, got {grid.get('grid_size')!r}.")
        if spacing.shape != (3,):
            raise ValueError(
                f"Expected grid_spacing length 3, got {grid.get('grid_spacing')!r}."
            )
        ratio = size / spacing
        counts = np.rint(ratio).astype(int)
        if not np.allclose(ratio, counts, rtol=0, atol=1e-10):
            raise ValueError(
                "grid_size/grid_spacing is not near-integer: "
                f"ratio={ratio.tolist()}, rounded={counts.tolist()}."
            )
        return spacing, int(counts[0]), int(counts[1]), int(counts[2])

    @classmethod
    def from_data_object(
        cls,
        json_path: str | Path,
        *,
        snapshot_index: int,
        origin_mode: str = "centroid_min",
        convert_to_um: bool = True,
    ) -> "_StatsVoxelMesh":
        data = load_mimedo(json_path)
        index = _normal_snapshot_index(data, snapshot_index)
        snapshot = _snapshot(data, index)
        voxels = snapshot.get("voxels")
        if not isinstance(voxels, list) or not voxels:
            raise ValueError(f"Snapshot {index} has no voxel data.")

        spacing, nx, ny, nz = cls._grid_from_snapshot(snapshot)
        scale_to_um = 1.0
        if convert_to_um:
            units = data.get("units", {})
            if not isinstance(units, Mapping):
                raise KeyError("Missing or invalid units object in JSON.")
            scale_to_um = _length_unit_scale_to_um(units.get("Length"))

        spacing_um = spacing.astype(float) * scale_to_um
        dx = float(spacing[0])
        grid = snapshot.get("grid", {})
        origin_json = grid.get("origin") if isinstance(grid, Mapping) else None
        if isinstance(origin_json, (list, tuple)) and len(origin_json) == 3:
            origin = np.asarray(origin_json, dtype=float)
        else:
            centers = np.asarray([v["centroid_coordinates"] for v in voxels], dtype=float)
            origin = centers.min(axis=0) - 0.5 * dx
        if origin_mode == "centroid_min":
            centers = np.asarray([v["centroid_coordinates"] for v in voxels], dtype=float)
            origin = centers.min(axis=0) - 0.5 * dx

        ii, jj, kk = np.meshgrid(
            np.arange(nx + 1),
            np.arange(ny + 1),
            np.arange(nz + 1),
            indexing="ij",
        )
        nodes = np.column_stack(
            (
                origin[0] + ii.ravel() * dx,
                origin[1] + jj.ravel() * dx,
                origin[2] + kk.ravel() * dx,
            )
        ).astype(float)
        if convert_to_um:
            nodes *= scale_to_um

        grains = snapshot.get("grains") if isinstance(snapshot.get("grains"), list) else []
        grain_table = {
            int(grain["grain_id"]): grain
            for grain in grains
            if isinstance(grain, dict) and "grain_id" in grain
        }
        voxel_dict: dict[int, list[int]] = {}
        grain_dict: dict[int, list[int]] = {}
        grain_phase_dict: dict[int, int] = {}
        voxel_centers: dict[int, np.ndarray] = {}
        stride_j = nz + 1
        stride_i = (ny + 1) * (nz + 1)

        for voxel in voxels:
            vid = int(voxel["voxel_id"])
            i, j, k = (
                int(voxel["voxel_index"][0]) - 1,
                int(voxel["voxel_index"][1]) - 1,
                int(voxel["voxel_index"][2]) - 1,
            )
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                raise ValueError(
                    f"voxel_index out of bounds for grid ({nx}, {ny}, {nz}): "
                    f"voxel_id={vid}, voxel_index={voxel['voxel_index']}"
                )
            n000 = 1 + i * stride_i + j * stride_j + k
            n100 = 1 + (i + 1) * stride_i + j * stride_j + k
            n010 = 1 + i * stride_i + (j + 1) * stride_j + k
            n110 = 1 + (i + 1) * stride_i + (j + 1) * stride_j + k
            n001 = 1 + i * stride_i + j * stride_j + (k + 1)
            n101 = 1 + (i + 1) * stride_i + j * stride_j + (k + 1)
            n011 = 1 + i * stride_i + (j + 1) * stride_j + (k + 1)
            n111 = 1 + (i + 1) * stride_i + (j + 1) * stride_j + (k + 1)
            voxel_dict[vid] = [n101, n100, n000, n001, n111, n110, n010, n011]

            gid = int(voxel["grain_id"])
            grain_dict.setdefault(gid, []).append(vid)
            center = np.asarray(voxel.get("centroid_coordinates"), dtype=float)
            voxel_centers[vid] = center * scale_to_um if convert_to_um else center

        for gid in grain_dict:
            grain_phase_dict[gid] = int(grain_table.get(gid, {}).get("phase_id", 0))

        return cls(
            nodes=nodes,
            voxel_dict=voxel_dict,
            grain_dict=grain_dict,
            grain_phase_dict=grain_phase_dict,
            voxel_centers=voxel_centers,
            voxel_spacing=spacing_um,
        )


def _fit_lognormal_params(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        raise ValueError("Cannot fit lognormal parameters to an empty array.")
    logs = np.log(finite)
    sigma = float(np.std(logs, ddof=1)) if finite.size > 1 else 1e-6
    sigma = max(sigma, 1e-6)
    scale = float(np.exp(np.mean(logs)))
    return scale, sigma


def _stats_state_from_arrays(
    semi_axes: np.ndarray,
    eq_diameters: np.ndarray,
) -> dict[str, Any]:
    if semi_axes.shape[0] == 0:
        raise ValueError("No grains were available for statistics extraction.")
    a = semi_axes[:, 0]
    b = semi_axes[:, 1]
    c = semi_axes[:, 2]
    aspect = a / np.maximum(c, 1e-12)
    eqd = np.asarray(eq_diameters, dtype=float)
    a_scale, a_sig = _fit_lognormal_params(a)
    b_scale, b_sig = _fit_lognormal_params(b)
    c_scale, c_sig = _fit_lognormal_params(c)
    ar_scale, ar_sig = _fit_lognormal_params(aspect)
    eqd_scale, eqd_sig = _fit_lognormal_params(eqd)
    return {
        "a": a,
        "b": b,
        "c": c,
        "eqd": eqd,
        "ar": aspect,
        "a_scale": a_scale,
        "a_sig": a_sig,
        "b_scale": b_scale,
        "b_sig": b_sig,
        "c_scale": c_scale,
        "c_sig": c_sig,
        "ar_scale": ar_scale,
        "ar_sig": ar_sig,
        "eqd_scale": eqd_scale,
        "eqd_sig": eqd_sig,
        "ind_rot": int(np.argmin(np.median(semi_axes, axis=0))),
    }


def _get_stats_vox_fallback(mesh: _StatsVoxelMesh) -> dict[str, Any]:
    semi_axes = []
    eq_diameters = []
    voxel_volume = float(np.prod(mesh.voxel_spacing))
    for voxel_ids in mesh.grain_dict.values():
        centers = np.asarray([mesh.voxel_centers[vid] for vid in voxel_ids], dtype=float)
        lengths = (centers.max(axis=0) - centers.min(axis=0)) + mesh.voxel_spacing
        axes = np.sort(0.5 * lengths)[::-1]
        volume = max(float(len(voxel_ids)) * voxel_volume, 1e-24)
        eqd = 2.0 * ((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0)
        semi_axes.append(axes)
        eq_diameters.append(eqd)
    return _stats_state_from_arrays(
        np.asarray(semi_axes, dtype=float),
        np.asarray(eq_diameters, dtype=float),
    )


@lru_cache(maxsize=32)
def _stats_for_snapshot_cached(
    path: str,
    mtime_ns: int,
    size: int,
    snapshot_index: int,
    use_kanapy: bool,
) -> tuple[dict[str, Any], str, list[str]]:
    del mtime_ns, size
    mesh = _StatsVoxelMesh.from_data_object(path, snapshot_index=snapshot_index)
    warnings: list[str] = []
    if use_kanapy:
        try:
            kanapy = importlib.import_module("kanapy")
            rve_stats = kanapy.core.rve_stats
            state = rve_stats.get_stats_vox(mesh, show_plot=False)
            return state, "kanapy", warnings
        except ModuleNotFoundError as exc:
            warnings.append(
                f"Kanapy statistics backend is unavailable ({exc.name}); "
                "using the internal JSON voxel fallback."
            )
        except Exception as exc:
            warnings.append(
                "Kanapy statistics backend failed; using the internal JSON voxel "
                f"fallback. Details: {type(exc).__name__}: {exc}"
            )
    return _get_stats_vox_fallback(mesh), "internal_voxel_bbox", warnings


def _stats_for_snapshot(
    json_path: str | Path,
    snapshot_index: int,
    *,
    use_kanapy: bool = True,
) -> tuple[dict[str, Any], str, list[str]]:
    return _stats_for_snapshot_cached(
        *_file_cache_key(json_path),
        int(snapshot_index),
        bool(use_kanapy),
    )


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot derive plot range from an empty array.")
    return float(np.min(arr)), float(np.max(arr))


def _build_stats_data(
    initial_state: dict[str, Any],
    regridded_state: dict[str, Any],
) -> dict[str, Any]:
    a_i = np.asarray(initial_state.get("a", []), dtype=float).ravel()
    b_i = np.asarray(initial_state.get("b", []), dtype=float).ravel()
    c_i = np.asarray(initial_state.get("c", []), dtype=float).ravel()
    ar_i = np.asarray(initial_state.get("ar", []), dtype=float).ravel()
    eq_i = np.asarray(initial_state.get("eqd", []), dtype=float).ravel()
    a_r = np.asarray(regridded_state.get("a", []), dtype=float).ravel()
    b_r = np.asarray(regridded_state.get("b", []), dtype=float).ravel()
    c_r = np.asarray(regridded_state.get("c", []), dtype=float).ravel()
    ar_r = np.asarray(regridded_state.get("ar", []), dtype=float).ravel()
    eq_r = np.asarray(regridded_state.get("eqd", []), dtype=float).ravel()

    xmin_i, xmax_i = _finite_range(np.concatenate([a_i, b_i, c_i]))
    xmin_r, xmax_r = _finite_range(np.concatenate([a_r, b_r, c_r]))
    xmin_ar_i, xmax_ar_i = _finite_range(ar_i)
    xmin_ar_r, xmax_ar_r = _finite_range(ar_r)
    xmin_eq_i, xmax_eq_i = _finite_range(eq_i)
    xmin_eq_r, xmax_eq_r = _finite_range(eq_r)

    return {
        "semi_axes": {
            "initial": {
                "a": a_i,
                "b": b_i,
                "c": c_i,
                "a_scale": initial_state.get("a_scale"),
                "a_sig": initial_state.get("a_sig"),
                "b_scale": initial_state.get("b_scale"),
                "b_sig": initial_state.get("b_sig"),
                "c_scale": initial_state.get("c_scale"),
                "c_sig": initial_state.get("c_sig"),
                "xmin_gl": xmin_i,
                "xmax_gl": xmax_i,
            },
            "regridded": {
                "a": a_r,
                "b": b_r,
                "c": c_r,
                "a_scale": regridded_state.get("a_scale"),
                "a_sig": regridded_state.get("a_sig"),
                "b_scale": regridded_state.get("b_scale"),
                "b_sig": regridded_state.get("b_sig"),
                "c_scale": regridded_state.get("c_scale"),
                "c_sig": regridded_state.get("c_sig"),
                "xmin_gl": xmin_r,
                "xmax_gl": xmax_r,
            },
        },
        "aspect": {
            "initial": {
                "ar": ar_i,
                "ar_scale": initial_state.get("ar_scale"),
                "ar_sig": initial_state.get("ar_sig"),
                "xmin_ar_i": xmin_ar_i,
                "xmax_ar_i": xmax_ar_i,
            },
            "regridded": {
                "ar": ar_r,
                "ar_scale": regridded_state.get("ar_scale"),
                "ar_sig": regridded_state.get("ar_sig"),
                "xmin_ar_r": xmin_ar_r,
                "xmax_ar_r": xmax_ar_r,
            },
        },
        "eq_diam": {
            "initial": {
                "eqd": eq_i,
                "eqd_scale": initial_state.get("eqd_scale"),
                "eqd_sig": initial_state.get("eqd_sig"),
                "xmin_eq_i": xmin_eq_i,
                "xmax_eq_i": xmax_eq_i,
            },
            "regridded": {
                "eqd": eq_r,
                "eqd_scale": regridded_state.get("eqd_scale"),
                "eqd_sig": regridded_state.get("eqd_sig"),
                "xmin_eq_r": xmin_eq_r,
                "xmax_eq_r": xmax_eq_r,
            },
        },
    }


def _statistics_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    a = np.asarray(state.get("a", []), dtype=float).ravel()
    b = np.asarray(state.get("b", []), dtype=float).ravel()
    c = np.asarray(state.get("c", []), dtype=float).ravel()
    ar = np.asarray(state.get("ar", []), dtype=float).ravel()
    eqd = np.asarray(state.get("eqd", []), dtype=float).ravel()
    xmin_gl, xmax_gl = _finite_range(np.concatenate([a, b, c]))
    xmin_ar, xmax_ar = _finite_range(ar)
    xmin_eq, xmax_eq = _finite_range(eqd)
    return {
        "a": a,
        "b": b,
        "c": c,
        "a_scale": state.get("a_scale"),
        "a_sig": state.get("a_sig"),
        "b_scale": state.get("b_scale"),
        "b_sig": state.get("b_sig"),
        "c_scale": state.get("c_scale"),
        "c_sig": state.get("c_sig"),
        "ar": ar,
        "ar_scale": state.get("ar_scale"),
        "ar_sig": state.get("ar_sig"),
        "eqd": eqd,
        "eqd_scale": state.get("eqd_scale"),
        "eqd_sig": state.get("eqd_sig"),
        "xmin_gl": xmin_gl,
        "xmax_gl": xmax_gl,
        "xmin_ar": xmin_ar,
        "xmax_ar": xmax_ar,
        "xmin_eq": xmin_eq,
        "xmax_eq": xmax_eq,
    }


def _build_multi_stats_data(
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    if not states:
        raise ValueError("Select at least one undeformed snapshot for statistics.")

    snapshots = []
    semi_by_snapshot: dict[str, dict[str, Any]] = {}
    aspect_by_snapshot: dict[str, dict[str, Any]] = {}
    eq_by_snapshot: dict[str, dict[str, Any]] = {}

    for item in states:
        label = str(item["label"])
        payload = _statistics_state_payload(item["state"])
        snapshots.append(
            {
                "snapshot_index": int(item["snapshot_index"]),
                "label": label,
                "backend": str(item["backend"]),
                "grain_count": int(len(payload["a"])),
            }
        )
        semi_by_snapshot[label] = {
            key: payload[key]
            for key in (
                "a",
                "b",
                "c",
                "a_scale",
                "a_sig",
                "b_scale",
                "b_sig",
                "c_scale",
                "c_sig",
                "xmin_gl",
                "xmax_gl",
            )
        }
        aspect_by_snapshot[label] = {
            "ar": payload["ar"],
            "ar_scale": payload["ar_scale"],
            "ar_sig": payload["ar_sig"],
            "xmin_ar": payload["xmin_ar"],
            "xmax_ar": payload["xmax_ar"],
        }
        eq_by_snapshot[label] = {
            "eqd": payload["eqd"],
            "eqd_scale": payload["eqd_scale"],
            "eqd_sig": payload["eqd_sig"],
            "xmin_eq": payload["xmin_eq"],
            "xmax_eq": payload["xmax_eq"],
        }

    stats_data: dict[str, Any] = {
        "snapshots": snapshots,
        "semi_axes": {"by_snapshot": semi_by_snapshot},
        "aspect": {"by_snapshot": aspect_by_snapshot},
        "eq_diam": {"by_snapshot": eq_by_snapshot},
    }

    first_label = snapshots[0]["label"]
    stats_data["semi_axes"]["initial"] = semi_by_snapshot[first_label]
    stats_data["aspect"]["initial"] = {
        **aspect_by_snapshot[first_label],
        "xmin_ar_i": aspect_by_snapshot[first_label]["xmin_ar"],
        "xmax_ar_i": aspect_by_snapshot[first_label]["xmax_ar"],
    }
    stats_data["eq_diam"]["initial"] = {
        **eq_by_snapshot[first_label],
        "xmin_eq_i": eq_by_snapshot[first_label]["xmin_eq"],
        "xmax_eq_i": eq_by_snapshot[first_label]["xmax_eq"],
    }

    if len(snapshots) > 1:
        last_label = snapshots[-1]["label"]
        stats_data["semi_axes"]["regridded"] = semi_by_snapshot[last_label]
        stats_data["aspect"]["regridded"] = {
            **aspect_by_snapshot[last_label],
            "xmin_ar_r": aspect_by_snapshot[last_label]["xmin_ar"],
            "xmax_ar_r": aspect_by_snapshot[last_label]["xmax_ar"],
        }
        stats_data["eq_diam"]["regridded"] = {
            **eq_by_snapshot[last_label],
            "xmin_eq_r": eq_by_snapshot[last_label]["xmin_eq"],
            "xmax_eq_r": eq_by_snapshot[last_label]["xmax_eq"],
        }

    return stats_data


def _json_stats_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return str(value)


def _save_stats_data(
    stats_data: dict[str, Any],
    json_path: str | Path,
    filename: str,
) -> Path:
    out_path = Path(json_path).expanduser().resolve().parent / filename
    out_path.write_text(json.dumps(stats_data, default=_json_stats_default, indent=2))
    return out_path


def _statistics_snapshot_options(json_path: str | Path) -> list[tuple[str, int]]:
    report = inspect_mimedo(json_path)
    return [
        (str(item["label"]), int(item["value"]))
        for item in report["statistics"].get("undeformed_snapshots", [])
    ]


def _shared_undeformed_snapshot_options(json_path: str | Path) -> list[tuple[str, int]]:
    texture_options = _texture_snapshot_options(json_path)
    stats_options = _statistics_snapshot_options(json_path)
    if texture_options and stats_options:
        stats_values = {value for _, value in stats_options}
        shared = [(label, value) for label, value in texture_options if value in stats_values]
        if shared:
            return shared
    return texture_options or stats_options


def _default_shared_undeformed_snapshots(json_path: str | Path) -> tuple[int, ...]:
    options = _shared_undeformed_snapshot_options(json_path)
    if not options:
        return ()
    values = [value for _, value in options]
    default_texture = [
        value for value in _default_texture_snapshots(json_path) if value in values
    ]
    if default_texture:
        return tuple(default_texture)
    return _default_statistics_snapshots(json_path)


def _default_statistics_snapshots(json_path: str | Path) -> tuple[int, ...]:
    options = _statistics_snapshot_options(json_path)
    if not options:
        return ()
    if len(options) == 1:
        return (options[0][1],)
    return (options[0][1], options[-1][1])


def _validate_statistics_snapshots(
    json_path: str | Path,
    snapshot_indices: list[int] | tuple[int, ...] | None,
) -> list[int]:
    options = _statistics_snapshot_options(json_path)
    available = {value for _, value in options}
    selected = list(_default_statistics_snapshots(json_path)) if snapshot_indices is None else [
        int(value) for value in snapshot_indices
    ]
    normalized: list[int] = []
    for value in selected:
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("No undeformed snapshots are available for statistics.")
    missing = [idx for idx in normalized if idx not in available]
    if missing:
        raise ValueError(
            "Statistics can only compare undeformed snapshots; invalid selection: "
            f"{missing}."
        )
    return normalized


def _lognorm_pdf(x: np.ndarray, sigma: float, scale: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-12)
    scale = max(float(scale), 1e-12)
    y = np.zeros_like(x, dtype=float)
    mask = x > 0
    z = (np.log(x[mask]) - math.log(scale)) / sigma
    y[mask] = np.exp(-0.5 * z * z) / (x[mask] * sigma * math.sqrt(2.0 * math.pi))
    return y


def _load_statistics_plot_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.legend_handler import HandlerTuple
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        dependency = exc.name or "statistics plotting dependency"
        raise ModuleNotFoundError(
            "Statistics plotting requires `matplotlib` in the active Python "
            f"environment. Missing dependency: {dependency}. Install matplotlib, "
            "then restart the kernel/session."
        ) from exc
    return plt, GridSpec, Line2D, HandlerTuple, None


def _plot_stats_overlay(
    stats_data: dict[str, Any],
    *,
    label_initial: str = STATISTICS_LABEL_INITIAL,
    label_regridded: str = STATISTICS_LABEL_REGRIDDED,
    figsize: tuple[float, float] = (16, 11),
    n_points_sa: int = 300,
    n_points_other: int = 600,
    lw: float = 3.0,
    alpha_sa_initial: float = 0.18,
    alpha_sa_regridded: float = 0.09,
    alpha_line_initial: float = 1.0,
    alpha_line_regridded: float = 0.70,
    alpha_other_initial: float = 0.12,
    alpha_other_regridded: float = 0.05,
    legend_handlelength: float = 5.0,
    show: bool = False,
) -> Any:
    plt, GridSpec, Line2D, HandlerTuple, _ = _load_statistics_plot_dependencies()
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        pass

    sa_i = stats_data["semi_axes"]["initial"]
    sa_r = stats_data["semi_axes"]["regridded"]
    ar_i = stats_data["aspect"]["initial"]
    ar_r = stats_data["aspect"]["regridded"]
    eq_i = stats_data["eq_diam"]["initial"]
    eq_r = stats_data["eq_diam"]["regridded"]

    xmin_sa = float(min(sa_i["xmin_gl"], sa_r["xmin_gl"]))
    xmax_sa = float(max(sa_i["xmax_gl"], sa_r["xmax_gl"]))
    xmin_ar = float(min(ar_i["xmin_ar_i"], ar_r["xmin_ar_r"]))
    xmax_ar = float(max(ar_i["xmax_ar_i"], ar_r["xmax_ar_r"]))
    xmin_eq = float(min(eq_i["xmin_eq_i"], eq_r["xmin_eq_r"]))
    xmax_eq = float(max(eq_i["xmax_eq_i"], eq_r["xmax_eq_r"]))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(nrows=3, ncols=1, figure=fig, hspace=0.38)
    ax_sa = fig.add_subplot(gs[0, 0])
    ax_ar = fig.add_subplot(gs[1, 0])
    ax_eq = fig.add_subplot(gs[2, 0])

    x_sa = np.linspace(max(xmin_sa, 1e-12), xmax_sa, n_points_sa, endpoint=True)
    axis_colors: dict[str, str] = {}
    max_pdf_sa = 0.0
    for key in ("a", "b", "c"):
        y = _lognorm_pdf(x_sa, sa_i[f"{key}_sig"], sa_i[f"{key}_scale"])
        line_i, = ax_sa.plot(
            x_sa,
            y,
            linewidth=lw,
            linestyle="-",
            alpha=alpha_line_initial,
        )
        axis_colors[key] = line_i.get_color()
        ax_sa.fill_between(x_sa, y, alpha=alpha_sa_initial, color=axis_colors[key])
        max_pdf_sa = max(max_pdf_sa, float(np.max(y)))

    for key in ("a", "b", "c"):
        y = _lognorm_pdf(x_sa, sa_r[f"{key}_sig"], sa_r[f"{key}_scale"])
        ax_sa.plot(
            x_sa,
            y,
            linewidth=lw,
            linestyle="--",
            color=axis_colors[key],
            alpha=alpha_line_regridded,
        )
        ax_sa.fill_between(x_sa, y, alpha=alpha_sa_regridded, color=axis_colors[key])
        max_pdf_sa = max(max_pdf_sa, float(np.max(y)))

    ax_sa.set_title(f"Semi-axes (a, b, c) - {label_initial} vs {label_regridded}", fontsize=16)
    ax_sa.set_xlabel("Length of semi-axis (um)", fontsize=14)
    ax_sa.set_ylabel("Density", fontsize=14)
    ax_sa.set_xlim(xmin_sa, xmax_sa)
    ax_sa.set_ylim(0.0, 1.05 * max_pdf_sa if max_pdf_sa > 0 else 1.0)

    handles = []
    labels = []
    for key in ("a", "b", "c"):
        col = axis_colors[key]
        handles.append(
            (
                Line2D([0], [0], color=col, lw=lw, linestyle="-", alpha=alpha_line_initial),
                Line2D([0], [0], color=col, lw=lw, linestyle="--", alpha=alpha_line_regridded),
            )
        )
        labels.append(key)
    ax_sa.legend(
        handles=handles,
        labels=labels,
        title=f"Semi-axis (solid={label_initial}, dashed={label_regridded})",
        loc="upper right",
        fontsize=11,
        title_fontsize=11,
        frameon=True,
        fancybox=True,
        handlelength=legend_handlelength,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.9)},
    )

    x_ar = np.linspace(max(xmin_ar, 1e-12), xmax_ar, n_points_other, endpoint=True)
    y_ar_i = _lognorm_pdf(x_ar, ar_i["ar_sig"], ar_i["ar_scale"])
    y_ar_r = _lognorm_pdf(x_ar, ar_r["ar_sig"], ar_r["ar_scale"])
    max_pdf_ar = max(float(np.max(y_ar_i)), float(np.max(y_ar_r)))
    ax_ar.plot(x_ar, y_ar_i, linewidth=lw, linestyle="-", label=label_initial)
    ax_ar.plot(
        x_ar,
        y_ar_r,
        linewidth=lw,
        linestyle="--",
        label=label_regridded,
        alpha=alpha_line_regridded,
    )
    ax_ar.fill_between(x_ar, y_ar_i, alpha=alpha_other_initial)
    ax_ar.fill_between(x_ar, y_ar_r, alpha=alpha_other_regridded)
    ax_ar.set_title(f"Aspect ratio - {label_initial} vs {label_regridded}", fontsize=16)
    ax_ar.set_xlabel("Aspect ratio (-)", fontsize=14)
    ax_ar.set_ylabel("Density", fontsize=14)
    ax_ar.set_xlim(xmin_ar, xmax_ar)
    ax_ar.set_ylim(0.0, 1.05 * max_pdf_ar if max_pdf_ar > 0 else 1.0)
    ax_ar.legend(loc="upper right", fontsize=11, frameon=True, fancybox=True)

    x_eq = np.linspace(max(xmin_eq, 1e-12), xmax_eq, n_points_other, endpoint=True)
    y_eq_i = _lognorm_pdf(x_eq, eq_i["eqd_sig"], eq_i["eqd_scale"])
    y_eq_r = _lognorm_pdf(x_eq, eq_r["eqd_sig"], eq_r["eqd_scale"])
    max_pdf_eq = max(float(np.max(y_eq_i)), float(np.max(y_eq_r)))
    ax_eq.plot(x_eq, y_eq_i, linewidth=lw, linestyle="-", label=label_initial)
    ax_eq.plot(
        x_eq,
        y_eq_r,
        linewidth=lw,
        linestyle="--",
        label=label_regridded,
        alpha=alpha_line_regridded,
    )
    ax_eq.fill_between(x_eq, y_eq_i, alpha=alpha_other_initial)
    ax_eq.fill_between(x_eq, y_eq_r, alpha=alpha_other_regridded)
    ax_eq.set_title(f"Equivalent diameter - {label_initial} vs {label_regridded}", fontsize=16)
    ax_eq.set_xlabel("Equivalent diameter (um)", fontsize=14)
    ax_eq.set_ylabel("Density", fontsize=14)
    ax_eq.set_xlim(xmin_eq, xmax_eq)
    ax_eq.set_ylim(0.0, 1.05 * max_pdf_eq if max_pdf_eq > 0 else 1.0)
    ax_eq.legend(loc="upper right", fontsize=11, frameon=True, fancybox=True)

    for axis in (ax_sa, ax_ar, ax_eq):
        axis.tick_params(labelsize=12)

    if show:
        plt.show()
    return fig


def _hex_to_rgba(color: str, alpha: float) -> str:
    value = color.strip().lstrip("#")
    if len(value) != 6:
        return f"rgba(80, 100, 120, {float(alpha):.3f})"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {float(alpha):.3f})"


def _style_statistics_axes(
    fig: go.Figure,
    *,
    row: int,
    x_title: str,
    y_title: str,
    x_range: list[float],
    y_max: float,
) -> None:
    fig.update_xaxes(
        title_text=x_title,
        range=x_range,
        showgrid=True,
        gridcolor="#e6eef8",
        zeroline=True,
        zerolinecolor="#b8c7da",
        linecolor="#8aa2bf",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickformat=".6~f",
        hoverformat=".6~f",
        exponentformat="none",
        showexponent="none",
        tickfont={"size": 11},
        title_font={"size": 13},
        row=row,
        col=1,
    )
    fig.update_yaxes(
        title_text=y_title,
        range=[0.0, 1.05 * y_max if y_max > 0 else 1.0],
        showgrid=True,
        gridcolor="#e6eef8",
        zeroline=True,
        zerolinecolor="#b8c7da",
        linecolor="#8aa2bf",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickformat=".6~f",
        hoverformat=".6~f",
        exponentformat="none",
        showexponent="none",
        tickfont={"size": 11},
        title_font={"size": 13},
        row=row,
        col=1,
    )


def _plot_statistics_snapshots(
    stats_data: dict[str, Any],
    *,
    figsize: tuple[float, float] = (16, 11),
    n_points_sa: int = 300,
    n_points_other: int = 600,
    lw: float = 2.7,
    show: bool = False,
) -> Any:
    del figsize, show
    snapshots = list(stats_data.get("snapshots", []))
    if not snapshots:
        raise ValueError("Statistics plot needs at least one selected snapshot.")
    semi = stats_data["semi_axes"]["by_snapshot"]
    aspect = stats_data["aspect"]["by_snapshot"]
    eq_diam = stats_data["eq_diam"]["by_snapshot"]
    labels = [str(item["label"]) for item in snapshots]

    xmin_sa = min(float(semi[label]["xmin_gl"]) for label in labels)
    xmax_sa = max(float(semi[label]["xmax_gl"]) for label in labels)
    xmin_ar = min(float(aspect[label]["xmin_ar"]) for label in labels)
    xmax_ar = max(float(aspect[label]["xmax_ar"]) for label in labels)
    xmin_eq = min(float(eq_diam[label]["xmin_eq"]) for label in labels)
    xmax_eq = max(float(eq_diam[label]["xmax_eq"]) for label in labels)

    subplot_title_suffix = labels[0] if len(labels) == 1 else f"{len(labels)} snapshots"
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            f"Semi-axes (a, b, c) - {subplot_title_suffix}",
            f"Aspect ratio - {subplot_title_suffix}",
            f"Equivalent diameter - {subplot_title_suffix}",
        ],
    )

    axis_colors = {"a": "#2563eb", "b": "#ea580c", "c": "#059669"}
    line_dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    state_colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#be185d",
        "#475569",
    ]
    fill_alpha = 0.10 if len(labels) == 1 else 0.035

    x_sa = np.linspace(max(xmin_sa, 1e-12), xmax_sa, n_points_sa, endpoint=True)
    max_pdf_sa = 0.0
    for state_index, label in enumerate(labels):
        dash = line_dashes[state_index % len(line_dashes)]
        alpha = 1.0 if state_index == 0 or len(labels) == 1 else 0.78
        for key in ("a", "b", "c"):
            y = _lognorm_pdf(
                x_sa,
                semi[label][f"{key}_sig"],
                semi[label][f"{key}_scale"],
            )
            fig.add_trace(
                go.Scatter(
                    x=x_sa,
                    y=y,
                    mode="lines",
                    line={
                        "color": axis_colors[key],
                        "width": lw,
                        "dash": dash,
                    },
                    opacity=alpha,
                    fill="tozeroy" if fill_alpha > 0 else None,
                    fillcolor=_hex_to_rgba(axis_colors[key], fill_alpha),
                    name=f"{key} - {label}",
                    legendgroup=f"semi-{key}",
                    **({"legend": "legend"} if _PLOTLY_MULTI_LEGEND else {}),
                    hovertemplate=(
                        f"{label}<br>semi-axis {key}: %{{x:.6~f}} um<br>"
                        "density: %{y:.6~f}<extra></extra>"
                    ),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            max_pdf_sa = max(max_pdf_sa, float(np.max(y)))

    x_ar = np.linspace(max(xmin_ar, 1e-12), xmax_ar, n_points_other, endpoint=True)
    max_pdf_ar = 0.0
    for state_index, label in enumerate(labels):
        y = _lognorm_pdf(
            x_ar,
            aspect[label]["ar_sig"],
            aspect[label]["ar_scale"],
        )
        color = state_colors[state_index % len(state_colors)]
        dash = line_dashes[state_index % len(line_dashes)]
        fig.add_trace(
            go.Scatter(
                x=x_ar,
                y=y,
                mode="lines",
                line={"color": color, "width": lw, "dash": dash},
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.08 if len(labels) == 1 else 0.035),
                name=label,
                legendgroup=f"state-{state_index}",
                **({"legend": "legend2"} if _PLOTLY_MULTI_LEGEND else {}),
                hovertemplate=(
                    f"{label}<br>aspect ratio: %{{x:.6~f}}<br>"
                    "density: %{y:.6~f}<extra></extra>"
                ),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        max_pdf_ar = max(max_pdf_ar, float(np.max(y)))

    x_eq = np.linspace(max(xmin_eq, 1e-12), xmax_eq, n_points_other, endpoint=True)
    max_pdf_eq = 0.0
    for state_index, label in enumerate(labels):
        y = _lognorm_pdf(
            x_eq,
            eq_diam[label]["eqd_sig"],
            eq_diam[label]["eqd_scale"],
        )
        color = state_colors[state_index % len(state_colors)]
        dash = line_dashes[state_index % len(line_dashes)]
        fig.add_trace(
            go.Scatter(
                x=x_eq,
                y=y,
                mode="lines",
                line={"color": color, "width": lw, "dash": dash},
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.08 if len(labels) == 1 else 0.035),
                name=label,
                legendgroup=f"state-{state_index}",
                **({"legend": "legend3"} if _PLOTLY_MULTI_LEGEND else {}),
                hovertemplate=(
                    f"{label}<br>equivalent diameter: %{{x:.6~f}} um<br>"
                    "density: %{y:.6~f}<extra></extra>"
                ),
                # Without multiple legends these labels already appear via the
                # aspect-ratio traces that share the same legendgroup.
                showlegend=_PLOTLY_MULTI_LEGEND,
            ),
            row=3,
            col=1,
        )
        max_pdf_eq = max(max_pdf_eq, float(np.max(y)))

    title = "Microstructure statistics"
    subtitle = labels[0] if len(labels) == 1 else " | ".join(labels[:3])
    if len(labels) > 3:
        subtitle += f" | +{len(labels) - 3} more"
    fig.update_layout(
        template="plotly_white",
        height=920,
        title={
            "text": f"<b>{title}</b><br><sup>{subtitle}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfdff",
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#102a43"},
        margin={"l": 78, "r": 36, "t": 104, "b": 70},
        hovermode="closest",
        legend={
            "orientation": "v",
            "x": 0.985,
            "xanchor": "right",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 10},
            "bgcolor": "rgba(255,255,255,0.78)",
            "bordercolor": "#d9e2ec",
            "borderwidth": 1,
            # A "Semi-axes" title is wrong when all traces share one legend.
            **(
                {"title": {"text": "Semi-axes"}}
                if _PLOTLY_MULTI_LEGEND
                else {}
            ),
        },
        **(
            {
                "legend2": {
                    "orientation": "v",
                    "x": 0.985,
                    "xanchor": "right",
                    "y": 0.585,
                    "yanchor": "top",
                    "font": {"size": 10},
                    "bgcolor": "rgba(255,255,255,0.78)",
                    "bordercolor": "#d9e2ec",
                    "borderwidth": 1,
                    "title": {"text": "Aspect ratio"},
                },
                "legend3": {
                    "orientation": "v",
                    "x": 0.985,
                    "xanchor": "right",
                    "y": 0.205,
                    "yanchor": "top",
                    "font": {"size": 10},
                    "bgcolor": "rgba(255,255,255,0.78)",
                    "bordercolor": "#d9e2ec",
                    "borderwidth": 1,
                    "title": {"text": "Eq. diameter"},
                },
            }
            if _PLOTLY_MULTI_LEGEND
            else {}
        ),
    )
    _style_statistics_axes(
        fig,
        row=1,
        x_title="Length of semi-axis [um]",
        y_title="Density",
        x_range=[xmin_sa, xmax_sa],
        y_max=max_pdf_sa,
    )
    _style_statistics_axes(
        fig,
        row=2,
        x_title="Aspect ratio [-]",
        y_title="Density",
        x_range=[xmin_ar, xmax_ar],
        y_max=max_pdf_ar,
    )
    _style_statistics_axes(
        fig,
        row=3,
        x_title="Equivalent diameter [um]",
        y_title="Density",
        x_range=[xmin_eq, xmax_eq],
        y_max=max_pdf_eq,
    )
    return fig


def build_statistics_figure(
    json_path: str | Path,
    *,
    snapshot_indices: list[int] | tuple[int, ...] | None = None,
    initial_snapshot_index: int | None = None,
    regridded_snapshot_index: int | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
    label_initial: str = STATISTICS_LABEL_INITIAL,
    label_regridded: str = STATISTICS_LABEL_REGRIDDED,
    use_kanapy: bool = True,
    save_stats_data: bool = False,
    stats_filename: str = "stats_data.json",
    show: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build statistics PDFs for one or more undeformed microstructure snapshots."""
    report = inspect_mimedo(json_path)
    if not report["modules"]["statistics"]:
        raise ValueError(report["statistics"]["reason"])
    old_pair_mode = snapshot_indices is None and (
        initial_snapshot_index is not None or regridded_snapshot_index is not None
    )
    if old_pair_mode:
        snapshot_indices = [
            value
            for value in (initial_snapshot_index, regridded_snapshot_index)
            if value is not None
        ]
    selected = _validate_statistics_snapshots(json_path, snapshot_indices)
    option_labels = {value: label for label, value in _statistics_snapshot_options(json_path)}
    if labels is not None:
        display_labels = [str(value) for value in labels]
        if len(display_labels) != len(selected):
            raise ValueError("Statistics labels must match the number of selected snapshots.")
    elif len(selected) == 2 and old_pair_mode:
        display_labels = [label_initial, label_regridded]
    else:
        display_labels = [
            option_labels.get(index, f"snapshot {index}") for index in selected
        ]

    seen_labels: dict[str, int] = {}
    unique_labels = []
    for label in display_labels:
        count = seen_labels.get(label, 0) + 1
        seen_labels[label] = count
        unique_labels.append(label if count == 1 else f"{label} ({count})")

    states = []
    warnings: list[str] = []
    backends = []
    for snapshot_index, label in zip(selected, unique_labels):
        state, backend, snapshot_warnings = _stats_for_snapshot(
            json_path,
            snapshot_index,
            use_kanapy=use_kanapy,
        )
        warnings.extend(snapshot_warnings)
        backends.append(backend)
        states.append(
            {
                "snapshot_index": snapshot_index,
                "label": label,
                "state": state,
                "backend": backend,
            }
        )

    stats_data = _build_multi_stats_data(states)
    saved_path = None
    if save_stats_data:
        saved_path = _save_stats_data(stats_data, json_path, stats_filename)
    fig = _plot_statistics_snapshots(
        stats_data,
        show=show,
    )
    state = build_control_state(json_path, plot="statistics")
    return fig, {
        "control_state": state,
        "warnings": warnings,
        "snapshots": selected,
        "stats_data": stats_data,
        "saved_path": str(saved_path) if saved_path is not None else None,
        "backend": backends[0] if len(set(backends)) == 1 else "+".join(backends),
    }


def _unit_text(value: Any) -> str:
    if value in (None, "", 1, "1"):
        return "-"
    return str(value)


def _axis_title(quantity: str, component: str, unit: str) -> str:
    return f"{quantity} {component} [{unit}]"


def _strain_quantity_label(strain_source: str) -> str:
    if strain_source == "plastic_strain":
        return "Plastic strain"
    return "Total strain"


def _strain_component_prefix(strain_source: str) -> str:
    if strain_source == "plastic_strain":
        return "Ep"
    return "E"


def _stress_component_label(stress_key: str) -> str:
    if stress_key == "equivalent_stress":
        return "Seq"
    return f"S{_component_suffix(stress_key)}"


def _strain_component_label(strain_source: str, strain_key: str) -> str:
    if strain_key == "equivalent_strain":
        return "Eeq"
    if strain_key == "equivalent_plastic_strain":
        return "Epeq"
    return f"{_strain_component_prefix(strain_source)}{_component_suffix(strain_key)}"


def _strain_key(strain_source: str, suffix: str) -> str:
    if strain_source == "plastic_strain":
        return f"plastic_strain_{suffix}"
    return f"strain_{suffix}"


def _component_suffix(component_key: str) -> str:
    return component_key.rsplit("_", 1)[-1]


def _component_pair(suffix: str, strain_source: str) -> tuple[str, str]:
    return f"stress_{suffix}", _strain_key(strain_source, suffix)


def _has_series(module: Mapping[str, Any], key: str) -> bool:
    value = module.get(key)
    return isinstance(value, list) and len(value) > 0


def _first_component_pair(
    stress: Mapping[str, Any],
    strain: Mapping[str, Any],
    strain_source: str,
) -> tuple[str, str] | None:
    pairs = _mechanical_pairs(
        _mechanical_lengths(stress),
        _mechanical_lengths(strain),
        strain_source,
    )
    return pairs[0] if pairs else None


def _resolve_component_pair(
    stress: Mapping[str, Any],
    strain: Mapping[str, Any],
    strain_source: str,
    stress_key: str,
    strain_key: str,
) -> tuple[str, str, str | None]:
    if _has_series(stress, stress_key) and _has_series(strain, strain_key):
        return stress_key, strain_key, None

    paired_strain_key = _paired_strain_key(stress_key, strain_source)
    if paired_strain_key and _has_series(stress, stress_key) and _has_series(
        strain,
        paired_strain_key,
    ):
        return (
            stress_key,
            paired_strain_key,
            f"{strain_key} is not available; plotted {stress_key} vs "
            f"{paired_strain_key}.",
        )

    fallback = _first_component_pair(stress, strain, strain_source)
    if fallback is None:
        raise ValueError(
            f"No plottable stress/{strain_source} component pairs are available"
        )
    fallback_stress, fallback_strain = fallback
    return (
        fallback_stress,
        fallback_strain,
        f"{stress_key} vs {strain_key} is not available; plotted "
        f"{fallback_stress} vs {fallback_strain}.",
    )


def _component_array(module: dict[str, Any], key: str) -> np.ndarray:
    value = module.get(key)
    if not isinstance(value, list):
        raise KeyError(f"{key} is not available")
    return np.asarray(value, dtype=float)


def _valid_pair(
    stress: dict[str, Any],
    strain: dict[str, Any],
    stress_key: str,
    strain_key: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if stress_key not in stress:
        return None, None, f"{stress_key} is not available"
    if strain_key not in strain:
        return None, None, f"{strain_key} is not available"
    stress_values = _component_array(stress, stress_key)
    strain_values = _component_array(strain, strain_key)
    if stress_values.shape[0] != strain_values.shape[0]:
        common_length = min(stress_values.shape[0], strain_values.shape[0])
        return (
            stress_values[:common_length],
            strain_values[:common_length],
            f"{stress_key} length {stress_values.shape[0]} does not match "
            f"{strain_key} length {strain_values.shape[0]}; plotted first "
            f"{common_length} paired values.",
        )
    return stress_values, strain_values, None


def _voxel_strain_field(strain_source: str) -> str:
    """Per-voxel field name backing a top-level strain module."""
    return "plastic_strain" if strain_source == "plastic_strain" else "strain"


def _snapshot_marker_candidates(
    json_path: str | Path,
    strain_source: str = "total_strain",
) -> list[int]:
    """Snapshots that can be homogenized onto the curve for ``strain_source``.

    A snapshot qualifies only if its voxels carry both per-voxel ``stress`` and the
    strain tensor backing the selected strain module. The undeformed snapshot has
    neither, so it is naturally excluded.
    """
    data = load_mimedo(json_path)
    micro = data.get("microstructure")
    if not _is_snapshot_array(micro):
        return []
    strain_field = _voxel_strain_field(strain_source)
    candidates: list[int] = []
    for index, snapshot in enumerate(micro):
        keys = _sample_keys(snapshot.get("voxels"), limit=1)
        if "stress" in keys and strain_field in keys:
            candidates.append(index)
    return candidates


def _homogenized_snapshot_point(
    json_path: str | Path,
    snapshot_index: int,
    stress_key: str,
    strain_key: str,
    strain_source: str,
) -> tuple[float, float] | None:
    """Homogenize one snapshot's per-voxel tensors onto the (strain, stress) curve.

    The exporter defines the top-level response as the *arithmetic mean* of the
    per-voxel tensor (equal reference voxel volumes on a structured grid), with the
    equivalent value taken as the Mises of that mean tensor -- not the mean of the
    per-voxel Mises. Reducing the mean tensor here reproduces the curve value
    exactly, so no snapshot-to-increment index mapping is needed.

    Returns ``None`` when the snapshot lacks the required per-voxel tensors (e.g.
    the undeformed snapshot, or an export without the ``sigma`` quantity).
    """
    try:
        arrays = _snapshot_arrays(json_path, snapshot_index)
    except (ValueError, IndexError):
        return None
    field_values = arrays.get("field_values", {})
    strain_field = _voxel_strain_field(strain_source)
    stress_stack = _stack_tensors(field_values.get("stress", []))
    strain_stack = _stack_tensors(field_values.get(strain_field, []))
    if stress_stack is None or strain_stack is None:
        return None

    def _reduce(stack: np.ndarray, key: str, base: str) -> float | None:
        mean_tensor = stack.mean(axis=0)[None, :, :]
        if key.startswith("equivalent"):
            return float(_reduce_stacked_tensors(mean_tensor, base, "mises")[0])
        suffix = _component_suffix(key)
        if suffix not in TENSOR_COMPONENT_INDEX:
            return None
        return float(_reduce_stacked_tensors(mean_tensor, base, suffix)[0])

    y = _reduce(stress_stack, stress_key, "stress")
    x = _reduce(strain_stack, strain_key, strain_field)
    if x is None or y is None:
        return None
    return x, y


def build_mechanical_figure(
    json_path: str | Path,
    *,
    plot_mode: str = "single",
    strain_source: str = "total_strain",
    stress_component: str = "stress_11",
    strain_component: str = "strain_11",
    multi_group: str = "normal",
    snapshot_markers: Sequence[int] | None = None,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build a mechanical response figure from stress and selected strain module.

    ``snapshot_markers`` optionally overlays each listed microstructure snapshot as
    a point on the curve, computed by homogenizing that snapshot's per-voxel
    tensors. Disabled by default; only drawn for ``plot_mode="single"``.
    """
    data = load_mimedo(json_path)
    state = build_control_state(json_path, plot="mechanical")
    if not state["report"]["modules"]["mechanical_response"]:
        raise ValueError(state["report"]["mechanical"]["reason"])
    if strain_source not in STRAIN_SOURCES:
        raise ValueError("strain_source must be 'total_strain' or 'plastic_strain'")
    if strain_source not in data or not isinstance(data.get(strain_source), dict):
        raise ValueError(f"{strain_source} is not available in this MiMeDO")

    stress = data["stress"]
    strain = data[strain_source]
    units = data.get("units", {}) if isinstance(data.get("units"), dict) else {}
    stress_unit = _unit_text(units.get("Stress"))
    strain_unit = _unit_text(units.get("Strain"))
    strain_quantity = _strain_quantity_label(strain_source)
    strain_prefix = _strain_component_prefix(strain_source)
    warnings: list[str] = []

    def _style_figure(fig: go.Figure, *, title: str, subtitle: str, height: int) -> None:
        fig.update_layout(
            template="plotly_white",
            height=height,
            title={
                "text": f"<b>{title}</b><br><sup>{subtitle}</sup>",
                "x": 0.02,
                "xanchor": "left",
            },
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fbfdff",
            font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": "#102a43"},
            margin={"l": 78, "r": 36, "t": 86, "b": 70},
            hovermode="closest",
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#e6eef8",
            zeroline=True,
            zerolinecolor="#b8c7da",
            linecolor="#8aa2bf",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickformat=".6~f",
            hoverformat=".6~f",
            exponentformat="none",
            showexponent="none",
            tickfont={"size": 11},
            title_font={"size": 13},
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e6eef8",
            zeroline=True,
            zerolinecolor="#b8c7da",
            linecolor="#8aa2bf",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickformat=".6~f",
            hoverformat=".6~f",
            exponentformat="none",
            showexponent="none",
            tickfont={"size": 11},
            title_font={"size": 13},
        )

    if plot_mode == "single":
        fig = go.Figure()
        stress_component, strain_component, selection_warning = _resolve_component_pair(
            stress,
            strain,
            strain_source,
            stress_component,
            strain_component,
        )
        if selection_warning:
            warnings.append(selection_warning)
        stress_label = _stress_component_label(stress_component)
        strain_label = _strain_component_label(strain_source, strain_component)
        y_values, x_values, warning = _valid_pair(
            stress, strain, stress_component, strain_component
        )
        if warning:
            warnings.append(warning)
        if y_values is not None and x_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{stress_label} vs {strain_label}",
                    line={"color": MECHANICAL_LINE_COLORS[0], "width": 3.2},
                    hovertemplate=(
                        f"{strain_label}: %{{x:.6~f}} {strain_unit}<br>"
                        f"{stress_label}: %{{y:.6~f}} {stress_unit}<extra></extra>"
                    ),
                )
            )
        if snapshot_markers:
            points: list[tuple[float, float]] = []
            labels: list[str] = []
            unavailable: list[int] = []
            for snapshot_index in snapshot_markers:
                point = _homogenized_snapshot_point(
                    json_path,
                    int(snapshot_index),
                    stress_component,
                    strain_component,
                    strain_source,
                )
                if point is None:
                    unavailable.append(int(snapshot_index))
                    continue
                points.append(point)
                labels.append(f"snapshot {int(snapshot_index)}")
            if unavailable:
                warnings.append(
                    f"No snapshot marker for snapshots {unavailable}: they lack "
                    f"per-voxel 'stress' and/or '{_voxel_strain_field(strain_source)}'."
                )
            if points:
                fig.add_trace(
                    go.Scatter(
                        x=[point[0] for point in points],
                        y=[point[1] for point in points],
                        mode="markers",
                        name="Microstructure snapshots",
                        text=labels,
                        marker={
                            "size": 9,
                            "color": "#ffffff",
                            "line": {"color": "#dc2626", "width": 2},
                            "symbol": "circle",
                        },
                        hovertemplate=(
                            "%{text}<br>"
                            f"{strain_label}: %{{x:.6~f}} {strain_unit}<br>"
                            f"{stress_label}: %{{y:.6~f}} {stress_unit}<extra></extra>"
                        ),
                    )
                )
        _style_figure(
            fig,
            title="Mechanical response",
            subtitle=f"{stress_label} vs {strain_quantity} {strain_label}",
            height=560,
        )
        fig.update_xaxes(
            title_text=_axis_title(strain_quantity, strain_label, strain_unit)
        )
        fig.update_yaxes(
            title_text=_axis_title("Stress", stress_label, stress_unit)
        )
        return fig, {"control_state": state, "warnings": warnings}

    if multi_group not in MULTI_COMPONENT_GROUPS:
        raise ValueError("multi_group must be normal, shear, or all")
    if snapshot_markers:
        warnings.append(
            "snapshot_markers are drawn only for plot_mode='single'; ignored here."
        )
    suffixes = {
        "normal": NORMAL_SUFFIXES,
        "shear": SHEAR_SUFFIXES,
        "all": MECHANICAL_COMPONENT_SUFFIXES,
    }[multi_group]
    rows = 2 if multi_group == "all" else 1
    cols = 3 if multi_group == "all" else len(suffixes)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"S{s} vs {strain_prefix}{s}" for s in suffixes],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    for index, suffix in enumerate(suffixes):
        stress_key, strain_key = _component_pair(suffix, strain_source)
        y_values, x_values, warning = _valid_pair(stress, strain, stress_key, strain_key)
        row = index // 3 + 1 if multi_group == "all" else 1
        col = index % 3 + 1 if multi_group == "all" else index + 1
        fig.update_xaxes(
            title_text=_axis_title(strain_quantity, f"{strain_prefix}{suffix}", strain_unit),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=_axis_title("Stress", f"S{suffix}", stress_unit),
            row=row,
            col=col,
        )
        if warning:
            warnings.append(warning)
        if y_values is None or x_values is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=f"S{suffix} vs {strain_prefix}{suffix}",
                line={
                    "color": MECHANICAL_LINE_COLORS[index % len(MECHANICAL_LINE_COLORS)],
                    "width": 2.8,
                },
                hovertemplate=(
                    f"{strain_prefix}{suffix}: %{{x:.6~f}} {strain_unit}<br>"
                    f"S{suffix}: %{{y:.6~f}} {stress_unit}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    _style_figure(
        fig,
        title="Mechanical response",
        subtitle=f"{multi_group.title()} components against {strain_quantity.lower()}",
        height=760 if multi_group == "all" else 560,
    )
    return fig, {"control_state": state, "warnings": warnings}


def _object_label(json_path: str | Path) -> str:
    data = load_mimedo(json_path)
    identifier = data.get("identifier")
    if isinstance(identifier, str) and identifier.strip():
        return identifier.strip()
    return Path(json_path).name


def _plot_options_for_path(json_path: str | Path) -> list[tuple[str, str]]:
    report = inspect_mimedo(json_path)
    options = []
    if report["modules"]["microstructure"]:
        options.append(("Microstructure", "microstructure"))
    if report["modules"]["texture"]:
        options.append(("Texture", "texture"))
    if report["modules"]["statistics"]:
        options.append(("Statistics", "statistics"))
    if report["modules"]["mechanical_response"]:
        options.append(("Mechanical response", "mechanical"))
    return options or [("No plottable module", "none")]


def _snapshot_options(json_path: str | Path) -> list[tuple[str, int]]:
    report = inspect_mimedo(json_path)
    if not report["modules"]["microstructure"]:
        return [("snapshot 0", 0)]
    data = load_mimedo(json_path)
    options = []
    for index, snapshot in enumerate(data["microstructure"]):
        label = f"snapshot {index}"
        if isinstance(snapshot, dict) and snapshot.get("time_point") is not None:
            label += f" | time={snapshot['time_point']}"
        options.append((label, index))
    return options


def _grain_options(json_path: str | Path, snapshot_index: int) -> list[tuple[str, int | None]]:
    try:
        arrays = _snapshot_arrays(json_path, snapshot_index)
    except Exception:
        return [("All voxels", None)]
    label = "All grains" if arrays["grain_ids"] else "All voxels"
    return [(label, None)] + [
        (f"grain {grain_id}", int(grain_id)) for grain_id in arrays["grain_ids"]
    ]


def _tensor_menu_options(base: str) -> list[tuple[str, str]]:
    """Abaqus-style component/Mises menu entries for one tensor field."""
    name = TENSOR_MENU_NAME.get(base, base)
    symbol = TENSOR_FIELD_SYMBOL.get(base, base)
    options = [(f"{name} - Mises", f"{base}:mises")]
    for comp in ("11", "22", "33", "12", "13", "23"):
        options.append((f"{name} - {symbol}{comp}", f"{base}:{comp}"))
    return options


def _color_options(json_path: str | Path, snapshot_index: int) -> list[tuple[str, str]]:
    report = inspect_mimedo(json_path, snapshot_index=snapshot_index)
    fields = report["microstructure"].get("color_fields", {})
    present_tensors = {
        base for base in TENSOR_REDUCTION_FIELDS if fields.get(base)
    }

    options: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    processed_fields: set[str] = set()

    def _emit(field: str) -> None:
        if field in processed_fields or not fields.get(field):
            return
        processed_fields.add(field)
        # A standalone Mises scalar is replaced by the tensor's ":mises" reduction.
        if field in MISES_SCALAR_FOR_TENSOR and MISES_SCALAR_FOR_TENSOR[field] in present_tensors:
            return
        entries = (
            _tensor_menu_options(field)
            if field in TENSOR_REDUCTION_FIELDS
            else [(COLOR_FIELD_LABELS.get(field, field), field)]
        )
        for label, value in entries:
            if value in seen_values:
                continue
            seen_values.add(value)
            options.append((label, value))

    for field in COLOR_FIELD_DEFAULT_ORDER:
        _emit(field)
    for field, enabled in fields.items():
        if enabled:
            _emit(field)

    return options or [("grain_id", "grain_id")]


def _default_color_value(json_path: str | Path, snapshot_index: int) -> str:
    options = _color_options(json_path, snapshot_index)
    if options:
        option = options[0]
        return str(option[1] if isinstance(option, tuple) else option)
    return "grain_id"


def _slice_bounds(json_path: str | Path, snapshot_index: int, axis: str) -> tuple[int, int]:
    arrays = _snapshot_arrays(json_path, snapshot_index)
    col = {"x": 0, "y": 1, "z": 2}[axis]
    values = arrays["voxel_index"][:, col]
    return int(values.min()), int(values.max())


def _rve_continuity_options(json_path: str | Path) -> tuple[list[tuple[str, bool | None]], bool | None, bool]:
    data = load_mimedo(json_path)
    if "RVE_continuity" in data and isinstance(data["RVE_continuity"], bool):
        value = bool(data["RVE_continuity"])
        return [(f"From JSON: {value}", value)], value, True
    return [("Choose...", None), ("True", True), ("False", False)], None, False


def build_microstructure_snapshots_figure(
    json_path: str | Path,
    *,
    snapshot_indices: list[int] | tuple[int, ...] | None = None,
    start_snapshot: int = 0,
    end_snapshot: int | None = None,
    count: int = 1,
    layout: str = "grid",
    selected_grain_id: int | None = None,
    color_by: str = "grain_id",
    colormap: str = DEFAULT_COLORMAP,
    same_camera: bool = True,
    same_color_scale: bool = True,
    marker_size: float = 0.92,
    show_axes: bool = True,
    show_origin: bool = True,
    show_grid: bool = True,
    view_type: str = "rve",
    graph_scope: str = "entire",
    slice_axis: str | None = None,
    slice_index: int | None = None,
    connectivity: int = 6,
    show_periodic_edges: bool = True,
    show_edge_labels: bool = False,
    graph_node_size: int = 4,
    edge_width: int = 2,
    edge_color_bins: int = 18,
    detailed_hover: bool = MICROSTRUCTURE_DETAILED_HOVER_DEFAULT,
    rve_continuity: bool | None = None,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build the unified snapshots view.

    A range/count resolving to one snapshot uses the full single-snapshot RVE
    builder, including graph options. Ranges resolving to multiple snapshots use
    the evolution grid.
    """
    report = inspect_mimedo(json_path)
    if not report["modules"]["microstructure"]:
        raise ValueError(report["microstructure"]["reason"])
    snapshot_total = int(report["microstructure"]["snapshot_count"])
    selected = (
        _validate_microstructure_snapshot_indices(
            json_path,
            list(snapshot_indices),
            max_count=MICROSTRUCTURE_MAX_SNAPSHOT_PANELS,
        )
        if snapshot_indices is not None
        else _snapshot_selection(snapshot_total, start_snapshot, end_snapshot, count)
    )

    if len(selected) == 1:
        fig, metadata = build_microstructure_figure(
            json_path,
            snapshot_index=selected[0],
            selected_grain_id=selected_grain_id,
            view_type=view_type,
            graph_scope=graph_scope,
            slice_axis=slice_axis,
            slice_index=slice_index,
            connectivity=connectivity,
            show_periodic_edges=show_periodic_edges,
            show_edge_labels=show_edge_labels,
            color_by=color_by,
            colormap=colormap,
            marker_size=marker_size,
            graph_node_size=graph_node_size,
            edge_width=edge_width,
            edge_color_bins=edge_color_bins,
            show_axes=show_axes,
            show_origin=show_origin,
            show_grid=show_grid,
            detailed_hover=detailed_hover,
            rve_continuity=rve_continuity,
        )
        metadata["snapshots"] = selected
        metadata["mode"] = "single_snapshot"
        return fig, metadata

    fig, metadata = build_microstructure_evolution_figure(
        json_path,
        snapshot_indices=selected,
        start_snapshot=start_snapshot,
        end_snapshot=end_snapshot,
        count=count,
        layout=layout,
        selected_grain_id=selected_grain_id,
        color_by=color_by,
        colormap=colormap,
        same_camera=same_camera,
        same_color_scale=same_color_scale,
        marker_size=marker_size,
        show_axes=show_axes,
        show_origin=show_origin,
        show_grid=show_grid,
        detailed_hover=detailed_hover,
    )
    metadata["mode"] = "snapshot_evolution"
    if view_type != "rve" or graph_scope != "entire":
        metadata.setdefault("warnings", []).append(
            "Graph controls apply only when the snapshot selection resolves to "
            "one snapshot; showing RVE evolution for the selected snapshots."
        )
    return fig, metadata


def mimedat_viewer_static(
    json_path: str | Path,
    *,
    plot: str = "auto",
    show: bool = True,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Convenience dispatcher for static viewer plotting."""
    state = build_control_state(json_path, plot=plot)
    active = state["controls"]["plot_family"].get("active")
    if plot != "auto":
        active = plot
    if active == "microstructure":
        snapshot_mode = kwargs.pop("snapshot_mode", "snapshots")
        if snapshot_mode == "compare":
            fig, metadata = build_microstructure_comparison_figure(json_path, **kwargs)
        elif snapshot_mode in {"snapshots", "multiple"}:
            fig, metadata = build_microstructure_snapshots_figure(json_path, **kwargs)
        else:
            fig, metadata = build_microstructure_figure(json_path, **kwargs)
    elif active == "texture":
        fig, metadata = build_texture_figure(json_path, **kwargs)
    elif active == "statistics":
        fig, metadata = build_statistics_figure(json_path, **kwargs)
    elif active == "mechanical":
        fig, metadata = build_mechanical_figure(json_path, **kwargs)
    else:
        raise ValueError(
            "No plottable microstructure, texture, statistics, or mechanical "
            "response module found"
        )
    metadata.setdefault("control_state", state)
    if show:
        fig.show()
    return fig, metadata


def mimedat_viewer(
    *json_paths: str | Path,
    show: bool = True,
    renderer: str | None = None,
    return_widgets: bool = False,
) -> Any:
    """Create an interactive MiMeDO viewer for one or more JSON objects.

    Example
    -------
    ``mimedat_viewer("c779a1c5.json", "c779c1a4.json")``

    The first control chooses the active data object. The remaining controls are
    kept visible and are enabled/frozen according to the modules available in
    that object and the current plot mode. Microstructure modes are snapshots
    and RVE-only comparison. In snapshots mode, a selection resolving to one
    snapshot enables the graph controls; multiple selected snapshots render the
    evolution grid. In comparison mode, snapshots become columns and selected
    color modes become rows, capped by ``COMPARISON_MAX_PANELS`` for
    responsiveness.
    """
    if not json_paths:
        raise ValueError("Pass at least one MiMeDO JSON path")

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Interactive mimedat_viewer requires ipywidgets in the active Python "
            "environment. Install it with:\n\n"
            "    pip install ipywidgets\n\n"
            "Then restart the kernel/session and rerun mimedat_viewer(...)."
        ) from exc

    paths = [str(Path(path).expanduser()) for path in json_paths]
    # Lazy labels: parsing every JSON just for its identifier took ~50 s and
    # ~1 GB for five files. Only paths[0] is loaded at startup; other objects
    # are parsed when first selected (_on_object_change -> _sync_object_controls).
    path_options = [(Path(path).name, path) for path in paths]

    object_dropdown = widgets.Dropdown(
        options=path_options,
        value=paths[0],
        description="Data object:",
        layout=widgets.Layout(width="460px"),
    )
    plot_dropdown = widgets.Dropdown(
        options=_plot_options_for_path(paths[0]),
        description="Plot:",
        layout=widgets.Layout(width="300px"),
    )

    snapshot_mode = widgets.ToggleButtons(
        options=[
            ("Snapshots", "snapshots"),
            ("Compare", "compare"),
        ],
        value="snapshots",
        description="Mode:",
    )
    snapshot_selection_mode = widgets.ToggleButtons(
        options=[
            ("Single", "single"),
            ("Range", "range"),
            ("Manual", "manual"),
        ],
        value="single",
        description="Selection:",
    )
    snapshot_dropdown = widgets.Dropdown(
        options=_snapshot_options(paths[0]),
        value=0,
        description="Snapshot:",
        layout=widgets.Layout(width="420px"),
    )
    previous_snapshot_button = widgets.Button(
        description="<",
        tooltip="Previous snapshot",
        layout=widgets.Layout(width="42px"),
    )
    next_snapshot_button = widgets.Button(
        description=">",
        tooltip="Next snapshot",
        layout=widgets.Layout(width="42px"),
    )
    grain_dropdown = widgets.Dropdown(
        options=_grain_options(paths[0], 0),
        value=None,
        description="Grain:",
        layout=widgets.Layout(width="260px"),
    )
    view_dropdown = widgets.Dropdown(
        options=[("RVE only", "rve"), ("RVE + Graph", "both"), ("Graph only", "graph")],
        value="rve",
        description="View:",
        layout=widgets.Layout(width="240px"),
    )
    initial_color_options = _color_options(paths[0], 0)
    color_dropdown = widgets.Dropdown(
        options=initial_color_options,
        value=_default_color_value(paths[0], 0),
        description="Color:",
        layout=widgets.Layout(width="320px"),
    )
    colormap_dropdown = widgets.Dropdown(
        options=list(COLORMAP_OPTIONS),
        value=DEFAULT_COLORMAP,
        description="Colormap:",
        layout=widgets.Layout(width="260px"),
    )
    compare_color_select = widgets.SelectMultiple(
        options=_color_options(paths[0], 0),
        value=tuple(_default_compare_colors(paths[0], 0)),
        description="Compare colors:",
        rows=4,
        layout=widgets.Layout(width="420px"),
    )

    graph_scope = widgets.Dropdown(
        options=[("Entire graph", "entire"), ("Slice graph", "slice")],
        value="entire",
        description="Graph:",
        layout=widgets.Layout(width="240px"),
    )
    slice_axis = widgets.Dropdown(
        options=[("X", "x"), ("Y", "y"), ("Z", "z")],
        value="z",
        description="Axis:",
        layout=widgets.Layout(width="180px"),
    )
    slice_index = widgets.BoundedIntText(
        value=1,
        min=1,
        max=1,
        description="Slice:",
        layout=widgets.Layout(width="200px"),
    )
    connectivity = widgets.Dropdown(
        options=[("6 face", 6), ("18 face+edge", 18), ("26 face+edge+corner", 26)],
        value=6,
        description="Connectivity:",
        layout=widgets.Layout(width="260px"),
    )
    rve_continuity_dropdown = widgets.Dropdown(
        options=_rve_continuity_options(paths[0])[0],
        value=_rve_continuity_options(paths[0])[1],
        description="RVE continuity:",
        layout=widgets.Layout(width="260px"),
    )
    show_periodic_edges = widgets.Checkbox(
        value=True,
        description="Show periodic wrap edges",
        indent=False,
        layout=widgets.Layout(width="240px"),
    )
    show_edge_labels = widgets.Checkbox(
        value=False,
        description="Show edge labels",
        indent=False,
        layout=widgets.Layout(width="180px"),
    )
    marker_size = widgets.FloatSlider(
        value=0.92,
        min=0.2,
        max=1.5,
        step=0.05,
        description="Voxel size:",
        continuous_update=False,
        layout=widgets.Layout(width="280px"),
    )
    show_axes = widgets.Checkbox(
        value=True,
        description="Show axes",
        indent=False,
        layout=widgets.Layout(width="130px"),
    )
    show_origin = widgets.Checkbox(
        value=True,
        description="Show origin",
        indent=False,
        layout=widgets.Layout(width="140px"),
    )
    show_grid = widgets.Checkbox(
        value=True,
        description="Show grid",
        indent=False,
        layout=widgets.Layout(width="130px"),
    )
    detailed_hover = widgets.Checkbox(
        value=MICROSTRUCTURE_DETAILED_HOVER_DEFAULT,
        description="Detailed hover",
        indent=False,
        tooltip="Build per-voxel hover labels. Leave off for faster 3D updates.",
        layout=widgets.Layout(width="170px"),
    )
    graph_node_size = widgets.IntSlider(
        value=4,
        min=1,
        max=12,
        step=1,
        description="Node size:",
        continuous_update=False,
        layout=widgets.Layout(width="280px"),
    )
    edge_width = widgets.IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description="Edge width:",
        continuous_update=False,
        layout=widgets.Layout(width="280px"),
    )

    range_start_snapshot = widgets.Dropdown(
        options=_snapshot_options(paths[0]),
        value=0,
        description="From:",
        layout=widgets.Layout(width="360px"),
    )
    range_end_snapshot = widgets.Dropdown(
        options=_snapshot_options(paths[0]),
        value=0,
        description="To:",
        layout=widgets.Layout(width="360px"),
    )
    range_sample_count = widgets.IntSlider(
        value=1,
        min=1,
        max=12,
        step=1,
        description="Samples:",
        continuous_update=False,
        layout=widgets.Layout(width="260px"),
    )
    manual_snapshot_select = widgets.SelectMultiple(
        options=_snapshot_options(paths[0]),
        value=(0,),
        description="Snapshots:",
        rows=6,
        layout=widgets.Layout(width="540px"),
    )
    selected_snapshot_preview = widgets.HTML(
        value="",
        layout=widgets.Layout(width="100%", margin="0 0 8px 0"),
    )
    snapshot_limit_note = widgets.HTML(
        value=(
            "<span style='color:#52616f;font-size:12px;'>"
            f"Limits: evolution shows up to {MICROSTRUCTURE_MAX_SNAPSHOT_PANELS} "
            f"snapshots; compare mode allows up to {COMPARISON_MAX_PANELS} panels "
            "(snapshots x color rows). Graph is available only for a single selected snapshot."
            "</span>"
        ),
        layout=widgets.Layout(width="100%", margin="0 0 8px 0"),
    )
    evolution_layout = widgets.Dropdown(
        options=[("Grid", "grid"), ("Row", "row")],
        value="grid",
        description="Layout:",
        layout=widgets.Layout(width="180px"),
    )
    same_camera = widgets.Checkbox(
        value=True,
        description="Same camera",
        indent=False,
        layout=widgets.Layout(width="160px"),
    )
    same_color_scale = widgets.Checkbox(
        value=True,
        description="Global color range",
        indent=False,
        tooltip=(
            "On: one value range shared across all shown snapshots. "
            "Off: each snapshot self-scales to its own min/max."
        ),
        layout=widgets.Layout(width="200px"),
    )

    texture_view = widgets.ToggleButtons(
        options=[("PF only", "pf"), ("PDF only", "pdf")],
        value="pf",
        description="Texture view:",
    )
    undeformed_snapshot_select = widgets.SelectMultiple(
        options=_shared_undeformed_snapshot_options(paths[0]),
        value=_default_shared_undeformed_snapshots(paths[0]),
        description="Undeformed snapshots:",
        rows=4,
        layout=widgets.Layout(width="540px"),
    )
    texture_max_points = widgets.Dropdown(
        options=[("5,000", 5000), ("10,000", 10000), ("20,000", 20000)],
        value=TEXTURE_MAX_POINTS,
        description="Max points:",
        layout=widgets.Layout(width="220px"),
    )
    stats_save_data = widgets.Checkbox(
        value=False,
        description="Save stats_data.json",
        indent=False,
        layout=widgets.Layout(width="220px"),
    )

    mechanical_plot_mode = widgets.ToggleButtons(
        options=[("Single", "single"), ("Multi", "multi")],
        value="single",
        description="Plot mode:",
    )
    strain_source_dropdown = widgets.Dropdown(
        options=[("Total strain", "total_strain")],
        value="total_strain",
        description="Strain type:",
        layout=widgets.Layout(width="260px"),
    )
    stress_dropdown = widgets.Dropdown(
        options=[],
        description="Stress:",
        layout=widgets.Layout(width="260px"),
    )
    strain_dropdown = widgets.Dropdown(
        options=[],
        description="Strain:",
        layout=widgets.Layout(width="260px"),
    )
    multi_group = widgets.Dropdown(
        options=[("Normal", "normal"), ("Shear", "shear"), ("All", "all")],
        value="normal",
        description="Group:",
        layout=widgets.Layout(width="220px"),
    )
    show_snapshot_points = widgets.Checkbox(
        value=False,
        description="Show snapshot points on curve",
        indent=False,
        tooltip=(
            "Overlay each microstructure snapshot as a point on the curve, "
            "homogenized onto the selected stress/strain component. "
            "Single plot mode only; snapshots without per-voxel stress/strain "
            "(e.g. the undeformed snapshot) are skipped."
        ),
        layout=widgets.Layout(width="300px"),
    )
    snapshot_points_select = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Points:",
        rows=4,
        layout=widgets.Layout(width="320px"),
    )

    output = widgets.Output()
    update_button = widgets.Button(
        description="Update plot",
        button_style="primary",
        tooltip="Rebuild the plot using the current controls",
        layout=widgets.Layout(width="130px"),
    )
    auto_update = widgets.Checkbox(
        value=False,
        description="Auto update",
        indent=False,
        tooltip="When enabled, every control change redraws the plot immediately.",
        layout=widgets.Layout(width="140px"),
    )
    render_status = widgets.HTML(
        value="",
        layout=widgets.Layout(width="320px"),
    )
    busy = {"depth": 0}
    render_pending = {"value": False}

    def _is_busy() -> bool:
        return busy["depth"] > 0

    def _begin_busy() -> None:
        busy["depth"] += 1

    def _end_busy() -> None:
        busy["depth"] = max(0, busy["depth"] - 1)

    micro_widgets = [
        snapshot_mode,
        snapshot_selection_mode,
        snapshot_dropdown,
        previous_snapshot_button,
        next_snapshot_button,
        range_start_snapshot,
        range_end_snapshot,
        range_sample_count,
        manual_snapshot_select,
        grain_dropdown,
        view_dropdown,
        color_dropdown,
        colormap_dropdown,
        compare_color_select,
        marker_size,
        show_axes,
        show_origin,
        show_grid,
        detailed_hover,
        evolution_layout,
        same_camera,
        same_color_scale,
    ]
    graph_widgets = [
        graph_scope,
        slice_axis,
        slice_index,
        connectivity,
        rve_continuity_dropdown,
        show_periodic_edges,
        show_edge_labels,
        graph_node_size,
        edge_width,
    ]
    mechanical_widgets = [
        mechanical_plot_mode,
        strain_source_dropdown,
        stress_dropdown,
        strain_dropdown,
        multi_group,
    ]
    texture_widgets = [
        texture_view,
        texture_max_points,
    ]
    undeformed_snapshot_widgets = [
        undeformed_snapshot_select,
    ]
    statistics_widgets = [
        stats_save_data,
    ]

    def _set_widget_options(widget: Any, options: list[Any], preferred: Any = None) -> None:
        widget.options = options
        values = [option[1] if isinstance(option, tuple) else option for option in options]
        if preferred in values:
            widget.value = preferred
        elif values:
            widget.value = values[0]

    def _set_multi_widget_options(
        widget: Any,
        options: list[Any],
        preferred: list[Any] | tuple[Any, ...] | None = None,
        default_values: list[Any] | tuple[Any, ...] | None = None,
    ) -> None:
        widget.options = options
        values = [option[1] if isinstance(option, tuple) else option for option in options]
        selected = [value for value in (preferred or []) if value in values]
        if not selected and default_values is not None:
            selected = [value for value in default_values if value in values]
        if not selected:
            selected = [
                value
                for value in _default_compare_colors(
                    object_dropdown.value,
                    _reference_snapshot_index(),
                )
                if value in values
            ]
        if not selected and values:
            selected = [values[0]]
        widget.value = tuple(selected)

    def _snapshot_option_values() -> list[int]:
        return [
            int(option[1]) if isinstance(option, tuple) else int(option)
            for option in snapshot_dropdown.options
        ]

    def _snapshot_label(snapshot_index: int) -> str:
        labels = {
            int(option[1]) if isinstance(option, tuple) else int(option): str(option[0])
            if isinstance(option, tuple)
            else str(option)
            for option in snapshot_dropdown.options
        }
        return labels.get(int(snapshot_index), f"snapshot {int(snapshot_index)}")

    def _selected_micro_snapshots() -> list[int]:
        path = object_dropdown.value
        try:
            report = inspect_mimedo(path)
            snapshot_total = int(report["microstructure"]["snapshot_count"])
            mode = snapshot_selection_mode.value
            if mode == "single":
                return _validate_microstructure_snapshot_indices(
                    path,
                    [int(snapshot_dropdown.value)],
                )
            if mode == "manual":
                selected = list(manual_snapshot_select.value)
                if not selected:
                    selected = [int(snapshot_dropdown.value)]
                return _validate_microstructure_snapshot_indices(path, selected)
            return _snapshot_selection(
                snapshot_total,
                int(range_start_snapshot.value),
                int(range_end_snapshot.value),
                int(range_sample_count.value),
            )
        except Exception:
            return [0]

    def _reference_snapshot_index() -> int:
        selected = _selected_micro_snapshots()
        return int(selected[0]) if selected else 0

    def _sync_snapshot_preview() -> None:
        selected = _selected_micro_snapshots()
        chips = " ".join(
            "<span style='display:inline-block;background:#e6f0ff;border:1px solid #9bbcf3;"
            "border-radius:10px;padding:2px 8px;margin:2px 4px 2px 0;font-size:12px;'>"
            f"{html.escape(_snapshot_label(snapshot_index))}</span>"
            for snapshot_index in selected
        )
        selected_snapshot_preview.value = (
            f"<div style='font-size:13px;color:#243b53;'>"
            f"<b>Selected ({len(selected)}):</b> {chips}</div>"
        )

    def _sync_snapshot_control_visibility() -> None:
        mode = snapshot_selection_mode.value
        single_display = None if mode == "single" else "none"
        range_display = None if mode == "range" else "none"
        manual_display = None if mode == "manual" else "none"
        for widget in [previous_snapshot_button, snapshot_dropdown, next_snapshot_button]:
            widget.layout.display = single_display
        for widget in [range_start_snapshot, range_end_snapshot, range_sample_count]:
            widget.layout.display = range_display
        manual_snapshot_select.layout.display = manual_display

    def _selected_texture_snapshots() -> list[int]:
        selected = [int(value) for value in undeformed_snapshot_select.value]
        available = {value for _, value in _texture_snapshot_options(object_dropdown.value)}
        return [value for value in selected if value in available]

    def _selected_statistics_snapshots() -> list[int]:
        selected = [int(value) for value in undeformed_snapshot_select.value]
        available = {value for _, value in _statistics_snapshot_options(object_dropdown.value)}
        return [value for value in selected if value in available]

    def _sync_slice_bounds() -> None:
        path = object_dropdown.value
        try:
            lo, hi = _slice_bounds(path, _reference_snapshot_index(), slice_axis.value)
        except Exception:
            lo, hi = 0, 0
        slice_index.min = lo
        slice_index.max = hi
        if not (lo <= int(slice_index.value) <= hi):
            slice_index.value = int(round((lo + hi) / 2))

    def _sync_snapshot_dependent_controls() -> None:
        path = object_dropdown.value
        if int(range_end_snapshot.value) < int(range_start_snapshot.value):
            range_end_snapshot.value = range_start_snapshot.value
        reference_snapshot = _reference_snapshot_index()
        snapshot_dropdown.value = reference_snapshot
        _set_widget_options(
            grain_dropdown,
            _grain_options(path, reference_snapshot),
            grain_dropdown.value,
        )
        _set_widget_options(
            color_dropdown,
            _color_options(path, reference_snapshot),
            color_dropdown.value,
        )
        _set_multi_widget_options(
            compare_color_select,
            _color_options(path, reference_snapshot),
            compare_color_select.value,
            _default_compare_colors(path, reference_snapshot),
        )
        _sync_snapshot_control_visibility()
        _sync_snapshot_preview()
        _sync_slice_bounds()

    def _sync_snapshot_point_options() -> None:
        """Offer only snapshots that carry the per-voxel tensors the overlay needs.

        Defaults to every available snapshot, so ticking the checkbox alone gives
        the "all available snapshots" behaviour.
        """
        path = object_dropdown.value
        try:
            candidates = _snapshot_marker_candidates(path, strain_source_dropdown.value)
        except Exception:
            candidates = []
        keep = tuple(value for value in snapshot_points_select.value if value in candidates)
        snapshot_points_select.options = [(f"snapshot {index}", index) for index in candidates]
        snapshot_points_select.value = keep or tuple(candidates)

    def _sync_object_controls() -> None:
        path = object_dropdown.value
        report = inspect_mimedo(path)
        old_plot = plot_dropdown.value
        _set_widget_options(plot_dropdown, _plot_options_for_path(path), old_plot)
        _sync_snapshot_point_options()

        if report["modules"]["microstructure"]:
            snapshot_options = _snapshot_options(path)
            _set_widget_options(snapshot_dropdown, snapshot_options, snapshot_dropdown.value)
            _set_widget_options(range_start_snapshot, snapshot_options, range_start_snapshot.value)
            _set_widget_options(range_end_snapshot, snapshot_options, range_end_snapshot.value)
            _set_multi_widget_options(
                manual_snapshot_select,
                snapshot_options,
                manual_snapshot_select.value,
                (snapshot_dropdown.value,),
            )
            range_sample_count.max = max(
                1,
                min(MICROSTRUCTURE_MAX_SNAPSHOT_PANELS, len(snapshot_options)),
            )
            _sync_snapshot_dependent_controls()
            _set_multi_widget_options(
                undeformed_snapshot_select,
                _shared_undeformed_snapshot_options(path),
                undeformed_snapshot_select.value,
                _default_shared_undeformed_snapshots(path),
            )
            continuity_options, continuity_value, _ = _rve_continuity_options(path)
            _set_widget_options(
                rve_continuity_dropdown,
                continuity_options,
                continuity_value,
            )
            _sync_slice_bounds()

        if report["modules"]["mechanical_response"]:
            mechanical = report["mechanical"]
            strain_source_options = [("Total strain", "total_strain")]
            if mechanical["plastic_component_pairs"]:
                strain_source_options.append(("Plastic strain", "plastic_strain"))
            _set_widget_options(
                strain_source_dropdown,
                strain_source_options,
                strain_source_dropdown.value,
            )
            stress_options = [(key, key) for key in mechanical["stress_components"]]
            if strain_source_dropdown.value == "plastic_strain":
                strain_options = [(key, key) for key in mechanical["plastic_components"]]
            else:
                strain_options = [(key, key) for key in mechanical["strain_components"]]
            _set_widget_options(stress_dropdown, stress_options, stress_dropdown.value)
            _set_widget_options(strain_dropdown, strain_options, strain_dropdown.value)

    def _sync_enabled_state() -> None:
        path = object_dropdown.value
        reference_snapshot = _reference_snapshot_index()
        report = inspect_mimedo(path, snapshot_index=reference_snapshot)
        plot_value = plot_dropdown.value
        has_micro = report["modules"]["microstructure"]
        has_texture = report["modules"]["texture"]
        has_stats = report["modules"]["statistics"]
        has_mech = report["modules"]["mechanical_response"]
        micro_active = plot_value == "microstructure" and has_micro
        texture_active = plot_value == "texture" and has_texture
        stats_active = plot_value == "statistics" and has_stats
        mech_active = plot_value == "mechanical" and has_mech
        snapshots_active = micro_active and snapshot_mode.value == "snapshots"
        compare_active = micro_active and snapshot_mode.value == "compare"
        selection_active = snapshots_active or compare_active
        selected_snapshots = _selected_micro_snapshots() if has_micro else []
        single_active = snapshots_active and len(selected_snapshots) == 1
        if snapshots_active and not single_active and view_dropdown.value != "rve":
            view_dropdown.value = "rve"
        graph_active = single_active and view_dropdown.value in {"both", "graph"}
        slice_active = graph_active and graph_scope.value == "slice"

        for widget in micro_widgets:
            widget.disabled = not micro_active
        snapshot_selection_mode.disabled = not selection_active
        snapshot_dropdown.disabled = not selection_active or snapshot_selection_mode.value != "single"
        previous_snapshot_button.disabled = snapshot_dropdown.disabled
        next_snapshot_button.disabled = snapshot_dropdown.disabled
        range_start_snapshot.disabled = not selection_active or snapshot_selection_mode.value != "range"
        range_end_snapshot.disabled = range_start_snapshot.disabled
        range_sample_count.disabled = range_start_snapshot.disabled
        manual_snapshot_select.disabled = not selection_active or snapshot_selection_mode.value != "manual"
        grain_dropdown.disabled = not micro_active or not bool(report["microstructure"].get("has_grain_ids"))
        view_dropdown.disabled = not single_active
        color_dropdown.disabled = not snapshots_active
        colormap_dropdown.disabled = not (snapshots_active or compare_active)
        compare_color_select.disabled = not compare_active
        evolution_layout.disabled = not snapshots_active or single_active
        same_camera.disabled = not selection_active or (snapshots_active and single_active)
        same_color_scale.disabled = not selection_active or (snapshots_active and single_active)
        _sync_snapshot_control_visibility()
        _sync_snapshot_preview()

        for widget in graph_widgets:
            widget.disabled = not graph_active
        _, _, has_rve_continuity = _rve_continuity_options(path)
        rve_continuity_dropdown.disabled = not graph_active or has_rve_continuity
        slice_axis.disabled = not slice_active
        slice_index.disabled = not slice_active

        for widget in texture_widgets:
            widget.disabled = not texture_active

        for widget in undeformed_snapshot_widgets:
            widget.disabled = not (texture_active or stats_active)

        for widget in statistics_widgets:
            widget.disabled = not stats_active

        for widget in mechanical_widgets:
            widget.disabled = not mech_active
        strain_source_dropdown.disabled = not mech_active
        stress_dropdown.disabled = not mech_active or mechanical_plot_mode.value != "single"
        strain_dropdown.disabled = not mech_active or mechanical_plot_mode.value != "single"
        multi_group.disabled = not mech_active or mechanical_plot_mode.value != "multi"
        # Snapshot markers need per-voxel stress/strain and are drawn only in the
        # single-component plot, so the control stays off and disabled otherwise.
        mech_single = mech_active and mechanical_plot_mode.value == "single"
        markers_possible = bool(snapshot_points_select.options)
        show_snapshot_points.disabled = not (mech_single and markers_possible)
        snapshot_points_select.disabled = (
            show_snapshot_points.disabled or not show_snapshot_points.value
        )
        show_snapshot_points.description = (
            "Show snapshot points on curve"
            if mech_single or not mech_active
            else "Show snapshot points (single plot mode only)"
        )

    def _render_current(*_: Any) -> None:
        if _is_busy():
            return
        _begin_busy()
        update_button.disabled = True
        render_status.value = (
            "<span style='color:#334155;font-size:12px;'>Rendering...</span>"
        )
        try:
            _sync_enabled_state()
            path = object_dropdown.value
            warnings: list[str] = []
            with output:
                output.clear_output(wait=True)
                try:
                    if plot_dropdown.value == "microstructure":
                        active_micro_snapshots = _selected_micro_snapshots()
                        if snapshot_mode.value == "compare":
                            fig, meta = build_microstructure_comparison_figure(
                                path,
                                snapshot_indices=active_micro_snapshots,
                                color_by_rows=list(compare_color_select.value),
                                colormap=colormap_dropdown.value,
                                selected_grain_id=grain_dropdown.value,
                                same_camera=same_camera.value,
                                same_color_scale=same_color_scale.value,
                                marker_size=marker_size.value,
                                show_axes=show_axes.value,
                                show_origin=show_origin.value,
                                show_grid=show_grid.value,
                                detailed_hover=detailed_hover.value,
                            )
                        else:
                            active_slice_axis = slice_axis.value if graph_scope.value == "slice" else None
                            active_slice_index = slice_index.value if graph_scope.value == "slice" else None
                            fig, meta = build_microstructure_snapshots_figure(
                                path,
                                snapshot_indices=active_micro_snapshots,
                                layout=evolution_layout.value,
                                selected_grain_id=grain_dropdown.value,
                                color_by=color_dropdown.value,
                                colormap=colormap_dropdown.value,
                                same_camera=same_camera.value,
                                same_color_scale=same_color_scale.value,
                                marker_size=marker_size.value,
                                show_axes=show_axes.value,
                                show_origin=show_origin.value,
                                show_grid=show_grid.value,
                                view_type=view_dropdown.value,
                                graph_scope=graph_scope.value,
                                slice_axis=active_slice_axis,
                                slice_index=active_slice_index,
                                connectivity=connectivity.value,
                                show_periodic_edges=show_periodic_edges.value,
                                show_edge_labels=show_edge_labels.value,
                                graph_node_size=graph_node_size.value,
                                edge_width=edge_width.value,
                                detailed_hover=detailed_hover.value,
                                rve_continuity=rve_continuity_dropdown.value,
                            )
                    elif plot_dropdown.value == "texture":
                        active_texture_snapshots = _selected_texture_snapshots()
                        if not active_texture_snapshots:
                            raise ValueError(
                                "Select at least one undeformed snapshot with voxel "
                                "orientations for texture plotting."
                            )
                        fig, meta = build_texture_figure(
                            path,
                            snapshot_indices=active_texture_snapshots,
                            texture_view=texture_view.value,
                            max_points=texture_max_points.value,
                        )
                    elif plot_dropdown.value == "statistics":
                        active_stats_snapshots = _selected_statistics_snapshots()
                        if not active_stats_snapshots:
                            raise ValueError(
                                "Select at least one undeformed snapshot with voxel, "
                                "grain, and grid data for statistics plotting."
                            )
                        fig, meta = build_statistics_figure(
                            path,
                            snapshot_indices=active_stats_snapshots,
                            save_stats_data=stats_save_data.value,
                        )
                    elif plot_dropdown.value == "mechanical":
                        active_markers = (
                            list(snapshot_points_select.value)
                            if (
                                show_snapshot_points.value
                                and mechanical_plot_mode.value == "single"
                            )
                            else None
                        )
                        fig, meta = build_mechanical_figure(
                            path,
                            plot_mode=mechanical_plot_mode.value,
                            strain_source=strain_source_dropdown.value,
                            stress_component=stress_dropdown.value,
                            strain_component=strain_dropdown.value,
                            multi_group=multi_group.value,
                            snapshot_markers=active_markers,
                        )
                    else:
                        raise ValueError("No plottable module is available for this object")
                    warnings = meta.get("warnings", [])
                    if isinstance(fig, go.Figure):
                        fig.show(renderer=renderer)
                    else:
                        display(fig)
                    if warnings:
                        display(_warning_box(warnings))
                except Exception as exc:
                    display(
                        _warning_box(
                            [f"Viewer could not build this plot: {type(exc).__name__}: {exc}"]
                        )
                    )
            render_pending["value"] = False
            render_status.value = (
                "<span style='color:#166534;font-size:12px;'>Plot is up to date.</span>"
            )
        finally:
            update_button.disabled = False
            _end_busy()

    def _set_render_pending() -> None:
        render_pending["value"] = True
        render_status.value = (
            "<span style='color:#9a3412;font-size:12px;'>"
            "Changes pending. Click <b>Update plot</b>."
            "</span>"
        )

    def _request_render(*_: Any, force: bool = False) -> None:
        if _is_busy():
            return
        if force or auto_update.value:
            _render_current()
            return
        _begin_busy()
        try:
            _sync_enabled_state()
            _set_render_pending()
        finally:
            _end_busy()

    def _on_object_change(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        if _is_busy():
            return
        _begin_busy()
        try:
            _sync_object_controls()
        finally:
            _end_busy()
        _request_render()

    def _on_snapshot_range_change(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        if _is_busy():
            return
        _begin_busy()
        try:
            _sync_snapshot_dependent_controls()
        finally:
            _end_busy()
        _request_render()

    def _on_slice_axis_change(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        if _is_busy():
            return
        _begin_busy()
        try:
            _sync_slice_bounds()
        finally:
            _end_busy()
        _request_render()

    object_dropdown.observe(_on_object_change, names="value")
    for widget in [
        snapshot_mode,
        snapshot_selection_mode,
        snapshot_dropdown,
        range_start_snapshot,
        range_end_snapshot,
        range_sample_count,
        manual_snapshot_select,
    ]:
        widget.observe(_on_snapshot_range_change, names="value")
    slice_axis.observe(_on_slice_axis_change, names="value")

    def _move_single_snapshot(delta: int) -> None:
        values = _snapshot_option_values()
        if not values:
            return
        current = int(snapshot_dropdown.value)
        index = values.index(current) if current in values else 0
        index = max(0, min(len(values) - 1, index + int(delta)))
        snapshot_dropdown.value = values[index]

    previous_snapshot_button.on_click(lambda _: _move_single_snapshot(-1))
    next_snapshot_button.on_click(lambda _: _move_single_snapshot(1))

    def _on_strain_source_change(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        if _is_busy():
            return
        path = object_dropdown.value
        report = inspect_mimedo(path)
        mechanical = report["mechanical"]
        _begin_busy()
        try:
            if change["new"] == "plastic_strain":
                strain_options = [(key, key) for key in mechanical["plastic_components"]]
            else:
                strain_options = [(key, key) for key in mechanical["strain_components"]]
            _set_widget_options(strain_dropdown, strain_options, None)
            # The overlay reads a different per-voxel tensor per strain source.
            _sync_snapshot_point_options()
        finally:
            _end_busy()
        _request_render()

    strain_source_dropdown.observe(_on_strain_source_change, names="value")

    for widget in [
        plot_dropdown,
        grain_dropdown,
        view_dropdown,
        color_dropdown,
        colormap_dropdown,
        compare_color_select,
        graph_scope,
        slice_index,
        connectivity,
        rve_continuity_dropdown,
        show_periodic_edges,
        show_edge_labels,
        marker_size,
        graph_node_size,
        edge_width,
        show_axes,
        show_origin,
        show_grid,
        detailed_hover,
        evolution_layout,
        same_camera,
        same_color_scale,
        texture_view,
        undeformed_snapshot_select,
        texture_max_points,
        stats_save_data,
        mechanical_plot_mode,
        stress_dropdown,
        strain_dropdown,
        multi_group,
        show_snapshot_points,
        snapshot_points_select,
    ]:
        widget.observe(_request_render, names="value")

    update_button.on_click(lambda _: _request_render(force=True))

    def _on_auto_update_change(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        if change.get("new") and render_pending["value"]:
            _request_render(force=True)

    auto_update.observe(_on_auto_update_change, names="value")

    def _spread_row(children: list[Any]) -> Any:
        return widgets.HBox(
            children,
            layout=widgets.Layout(
                width="100%",
                justify_content="space-between",
                align_items="center",
                flex_flow="row wrap",
                margin="0 0 8px 0",
            ),
        )

    def _section_title(label: str) -> Any:
        return widgets.HTML(
            f"<b>{label}</b>",
            layout=widgets.Layout(width="100%", margin="8px 0 4px 0"),
        )

    def _warning_box(warnings: list[str]) -> Any:
        items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
        return widgets.HTML(
            "<div style='background:#fff7ed;border:1px solid #fdba74;"
            "border-left:5px solid #f59e0b;color:#7c2d12;padding:10px 12px;"
            "margin:12px 0 0 0;font-size:13px;line-height:1.45;'>"
            "<div style='font-weight:700;margin-bottom:4px;'>Warnings</div>"
            f"<ul style='margin:0;padding-left:18px;'>{items}</ul>"
            "</div>",
            layout=widgets.Layout(width="100%"),
        )

    controls = widgets.VBox(
        [
            _spread_row([object_dropdown, plot_dropdown]),
            _spread_row([update_button, auto_update, render_status]),
            _section_title("Microstructure"),
            _spread_row([snapshot_mode, grain_dropdown]),
            _section_title("Snapshots"),
            _spread_row([snapshot_selection_mode, evolution_layout]),
            _spread_row(
                [previous_snapshot_button, snapshot_dropdown, next_snapshot_button]
            ),
            _spread_row([range_start_snapshot, range_end_snapshot, range_sample_count]),
            _spread_row([manual_snapshot_select]),
            selected_snapshot_preview,
            snapshot_limit_note,
            _spread_row([same_camera, same_color_scale]),
            _spread_row([view_dropdown, color_dropdown, colormap_dropdown]),
            _spread_row([marker_size]),
            _spread_row([show_axes, show_origin, show_grid, detailed_hover]),
            _section_title("Comparison"),
            _spread_row([compare_color_select]),
            _section_title("Graph"),
            _spread_row(
                [graph_scope, slice_axis, slice_index, connectivity, rve_continuity_dropdown]
            ),
            _spread_row(
                [show_periodic_edges, show_edge_labels, graph_node_size, edge_width]
            ),
            _section_title("Undeformed snapshots"),
            _spread_row([undeformed_snapshot_select]),
            _section_title("Texture"),
            _spread_row([texture_view, texture_max_points]),
            _section_title("Statistics"),
            _spread_row([stats_save_data]),
            _section_title("Mechanical response"),
            _spread_row(
                [
                    mechanical_plot_mode,
                    strain_source_dropdown,
                    stress_dropdown,
                    strain_dropdown,
                    multi_group,
                ]
            ),
            _spread_row([show_snapshot_points, snapshot_points_select]),
        ],
        layout=widgets.Layout(width="100%"),
    )

    _sync_object_controls()
    _sync_enabled_state()
    if show:
        display(controls, output)
        _render_current()
    if return_widgets:
        return controls, output
    return None


def clear_viewer_caches() -> None:
    """Clear JSON, snapshot, and graph caches."""
    _load_json_cached.cache_clear()
    _snapshot_arrays_cached.cache_clear()
    _voxel_mesh_geometry_cached.cache_clear()
    _texture_payload_cached.cache_clear()
    _stats_for_snapshot_cached.cache_clear()
    _fast_graph_cached.cache_clear()
    _weighted_graph_cached.cache_clear()
