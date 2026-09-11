"""
MiMeDO identifier generation utilities.

This module creates a deterministic short identifier for a MiMeDO data object
using selected mandatory fields. The identifier is intended to be inserted into
``data["identifier"]`` before saving the JSON file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


DEFAULT_MIMEDO_IDENTIFIER_FIELDS: tuple[str, ...] = (
    "title",
    "creator",
    "creator_affiliation",
    "date",
    "shared_with",
    "rights",
    "rights_holder",
    "software",
    "software_version",
    "system",
    "system_version",
    "processor_specifications",
    "RVE_size",
    "RVE_continuity",
    "discretization_type",
    "discretization_unit_size",
    "discretization_count",
    "mechanical_BC",
    "phase",
    "microstructure",
    "units",
)


def create_mimedo_identifier(
    data: Mapping[str, Any],
    hash_length: int = 8,
    mandatory_fields: Sequence[str] | None = None,
) -> str:
    """
    Create a deterministic short identifier for a MiMeDO data object.

    Parameters
    ----------
    data
        MiMeDO data dictionary.
    hash_length
        Number of hexadecimal characters returned from the SHA-256 hash.
        The default is 8.
    mandatory_fields
        Optional custom list of fields used to build the hash payload.
        If omitted, ``DEFAULT_MIMEDO_IDENTIFIER_FIELDS`` is used.

    Returns
    -------
    str
        Deterministic short identifier, for example ``"a46fde6c"``.

    Notes
    -----
    - The existing ``identifier`` field is excluded from the hash payload.
    - Values ``0``, ``0.0``, and ``False`` are kept because they are valid data.
    - Missing or empty mandatory fields raise ``ValueError``.
    - The ``"microstructure"`` field is bound to its first entry
      (``microstructure[0]``, the initial simulation-ready snapshot) rather
      than hashed wholesale. ``microstructure[0]`` never changes once
      created -- it always carries its own ``microstructure_state_id``,
      which Kanapy's ``write_data`` assigns unconditionally before this
      function is ever called -- whereas later workflow stages (running the
      simulation, post-processing) append further snapshots to the array.
      Hashing the whole array would make this identifier drift if it were
      ever recomputed from a completed/updated MiMeDO, breaking
      reproducibility from the stored object. Binding to index 0 keeps the
      identifier sensitive to both the initial snapshot's raw content
      (grid/grains/voxels) and its own id together, while staying fixed
      regardless of what gets appended afterward. ``"phase"`` needs no
      equivalent narrowing: it is not a growing time series, and each phase
      entry already carries its own ``orientation_identifier`` alongside its
      raw orientation content.
    """
    if not isinstance(data, Mapping):
        raise TypeError("data must be a dictionary-like mapping.")

    if hash_length <= 0:
        raise ValueError("hash_length must be a positive integer.")

    fields = tuple(mandatory_fields or DEFAULT_MIMEDO_IDENTIFIER_FIELDS)

    cleaned_data = _remove_empty_entries(data)
    cleaned_data.pop("identifier", None)

    microstructure = cleaned_data.get("microstructure")
    if isinstance(microstructure, list) and microstructure:
        initial_snapshot = microstructure[0]
        # Whitelist the same "input" fields create_microstructure_identifier
        # itself hashes -- not the raw snapshot dict as currently stored. This
        # keeps the identifier reproducible even after post_processing enriches
        # snapshot 0 in place (e.g. add_ipf_color adds IPFcolor_* to every
        # snapshot's voxels, snapshot 0 included): those additions are not
        # "input" and must not change the identifier.
        grain_input_fields = {"grain_id", "phase_id", "grain_volume", "orientation"}
        voxel_input_fields = {
            "voxel_id", "grain_id", "phase_id", "centroid_coordinates",
            "voxel_index", "voxel_volume", "orientation",
        }
        cleaned_data["microstructure"] = {
            "grid": initial_snapshot.get("grid"),
            "grains": [
                {k: v for k, v in grain.items() if k in grain_input_fields}
                for grain in initial_snapshot.get("grains", [])
            ],
            "voxels": [
                {k: v for k, v in voxel.items() if k in voxel_input_fields}
                for voxel in initial_snapshot.get("voxels", [])
            ],
            "microstructure_state_id": initial_snapshot.get("microstructure_state_id"),
        }

    payload: dict[str, Any] = {}
    missing_fields: list[str] = []

    for field in fields:
        value = cleaned_data.get(field)

        if _is_empty_value(value):
            missing_fields.append(field)
        else:
            payload[field] = value

    if missing_fields:
        raise ValueError(
            "Cannot generate MiMeDO identifier. "
            f"The following mandatory fields are missing or empty: {missing_fields}"
        )

    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )

    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()[:hash_length]


def _is_empty_value(value: Any) -> bool:
    """
    Return True only for truly empty values.

    NumPy-safe:
    - keeps 0, 0.0, and False
    - avoids value == [] and value == {} on arrays
    """
    if value is None:
        return True

    if isinstance(value, str):
        return value == ""

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.size == 0

        if isinstance(value, np.generic):
            return False

    if isinstance(value, Mapping):
        return len(value) == 0

    if isinstance(value, (list, tuple, set)):
        return len(value) == 0

    return False


def _make_json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-native Python objects into JSON-safe values.
    """
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {str(key): _make_json_safe(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]

    return value


def _remove_empty_entries(value: Any) -> Any:
    """
    Recursively remove empty values from dictionaries and lists.
    """
    value = _make_json_safe(value)

    if isinstance(value, Mapping):
        cleaned_dict: dict[str, Any] = {}

        for key, item in value.items():
            cleaned_item = _remove_empty_entries(item)

            if not _is_empty_value(cleaned_item):
                cleaned_dict[str(key)] = cleaned_item

        return cleaned_dict

    if isinstance(value, list):
        cleaned_list = []

        for item in value:
            cleaned_item = _remove_empty_entries(item)

            if not _is_empty_value(cleaned_item):
                cleaned_list.append(cleaned_item)

        return cleaned_list

    return value

def create_orientation_identifier(
    eulers: Any,
    hash_length: int = 5,
    decimals: int = 6,
) -> str:
    """
    Create a deterministic short identifier for a phase orientation block.

    Parameters
    ----------
    eulers
        Euler-angle array with shape (n_grains, 3), ordered as:
        [Phi1, Phi, Phi2].
    hash_length
        Number of hexadecimal characters returned from the SHA-256 hash.
        The default is 5.
    decimals
        Number of decimals used before hashing.
        The default is 6.

    Returns
    -------
    str
        Short deterministic orientation identifier, for example ``"a46fd"``.
    """
    if hash_length <= 0:
        raise ValueError("hash_length must be a positive integer.")

    if np is None:
        raise ImportError("NumPy is required to create an orientation identifier.")

    eulers_array = np.asarray(eulers, dtype=float)

    if eulers_array.ndim != 2 or eulers_array.shape[1] != 3:
        raise ValueError(
            "eulers must be a 2D array with shape (n_grains, 3), "
            "ordered as [Phi1, Phi, Phi2]."
        )

    eulers_array = np.round(eulers_array, decimals=decimals)

    payload = json.dumps(
        eulers_array.tolist(),
        separators=(",", ":"),
        sort_keys=False,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:hash_length]
