"""Build voxel-neighbor graphs from MiMeDO-style microstructure JSON files."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
from orix.quaternion import Orientation, symmetry as orix_sym


def _build_symmetry_lookup() -> dict[str, object]:
    """Map Hermann-Mauguin symmetry names, e.g. ``m-3m``, to orix objects."""
    table = {}
    for attr_name in dir(orix_sym):
        if attr_name.startswith("_"):
            continue
        obj = getattr(orix_sym, attr_name)
        nm = getattr(obj, "name", None)
        if isinstance(nm, str) and nm:
            table.setdefault(nm, obj)
    return table


_SYM_NAME_TO_OBJ = _build_symmetry_lookup()


def build_voxel_graph(
    file_path: Union[str, Path],
    *,
    snapshot_index: int = 0,
    connectivity: int = 6,
) -> dict:
    """
    Convert one snapshot of a microstructure JSON into a graph.

    Periodicity is determined automatically from the top-level
    ``RVE_continuity`` field in the JSON (True → wrap, False → no wrap).
    If ``RVE_continuity`` is missing, a KeyError is raised — silently
    guessing would risk producing a wrong segmentation.

    Parameters
    ----------
    file_path : str or Path
        Path to the microstructure JSON file.
    snapshot_index : int, default 0
        Which snapshot under ``data["microstructure"]`` to convert.
    connectivity : int, default 6
        Voxel neighbourhood in 3D:
          6  — face neighbors only (face-sharing, smallest graph)
         18  — face + edge neighbors (intermediate)
         26  — face + edge + corner (densest graph)

    Returns
    -------
    dict with keys:
        N                  : int                       — number of voxels
        edges              : (E, 2) int32              — undirected edge endpoints
                                                          (each edge appears once,
                                                           with edges[i,0] < edges[i,1])
        weights            : (E,) float64              — ORIX misorientation in degrees
                                                          (np.inf for cross-phase edges)
        edge_is_periodic   : (E,) bool                 — True for wrap-around edges
                                                          added by periodic boundaries
        orientations       : ORIX Orientation, len N   — built with the (unique or
                                                          dominant) phase symmetry
        voxel_volume       : (N,) float64              — per-voxel volume
        voxel_index        : (N, 3) int32              — original 1-based grid indices
        old_grain_id       : (N,) int32                — stale grain_id from the JSON
        pos_to_idx         : dict[(int,int,int) -> int]— (i,j,k) -> node index
        phase_id_per_voxel : (N,) int32                — phase id resolved via
                                                          voxel.grain_id -> grain.phase_id
        symmetry_per_phase : dict[int -> orix Symmetry]— phase_id -> point-group object
        is_multi_phase     : bool                      — True if >1 phase present
        eulers             : (N, 3) float64            — raw Euler angles, radians
                                                          (Bunge convention, lab->crystal)
        periodic           : bool                      — value of RVE_continuity from
                                                          the JSON; True means the
                                                          graph includes wrap-around
                                                          edges between opposite faces
        snapshot_index     : int                       — echoed back for traceability
        snapshot_time      : float or None             — simulation time of this snapshot
        connectivity       : int                       — echoed back

    Notes
    -----
    Phase resolution chain:
        voxel.grain_id  ->  grains[].phase_id  ->  phase[].orientation.crystal_symmetry_group
        -> ORIX point-group object via Hermann-Mauguin name match.

    Multi-phase handling:
        Edge weights are computed *per phase* using the correct symmetry.
        For an edge whose two endpoints belong to different phases, the
        misorientation is not crystallographically meaningful, so weight = inf.
        Downstream segmentation that respects this will treat phase boundaries
        as hard cuts (which is physically correct for grain segmentation).

        The single 'orientations' Orientation object uses the *unique* phase's
        symmetry for single-phase data, or the *dominant* phase's symmetry for
        multi-phase data (with a warning). For accurate per-phase mean-orientation
        computations downstream you can rebuild Orientation objects from
        ``eulers[mask]`` using ``symmetry_per_phase[pid]``.

    Periodicity:
        Determined from the JSON's top-level ``RVE_continuity`` field.
        When True, the bounding box of voxel_index along each axis is
        wrapped using modular arithmetic, so a voxel at the minimum index
        of an axis is connected to the voxel at the maximum index (and
        equivalent corner/edge wraps for connectivity 18 or 26).
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be one of 6, 18, or 26")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "microstructure" not in data:
        raise KeyError("Top-level key 'microstructure' not found in JSON.")
    if "phase" not in data:
        raise KeyError("Top-level key 'phase' not found in JSON.")
    if "RVE_continuity" not in data:
        raise KeyError(
            "Top-level key 'RVE_continuity' not found in JSON. "
            "This field must be present (True for a periodic RVE, "
            "False for a non-periodic one). The graph builder uses it "
            "to decide whether to add wrap-around edges between opposite "
            "faces of the grid; silently guessing would risk corrupting "
            "downstream segmentation."
        )
    periodic_raw = data["RVE_continuity"]
    if not isinstance(periodic_raw, bool):
        raise TypeError(
            "Top-level key 'RVE_continuity' must be a JSON boolean "
            f"(true/false), got {type(periodic_raw).__name__}: "
            f"{periodic_raw!r}."
        )
    periodic = periodic_raw

    snaps = data["microstructure"]
    if not (0 <= snapshot_index < len(snaps)):
        raise IndexError(
            f"snapshot_index={snapshot_index} out of range [0, {len(snaps)})."
        )

    snap = snaps[snapshot_index]
    voxels = snap.get("voxels")
    grains = snap.get("grains")
    if voxels is None:
        raise KeyError(f"Snapshot {snapshot_index} has no 'voxels'.")
    if grains is None:
        raise KeyError(f"Snapshot {snapshot_index} has no 'grains'.")
    if len(voxels) == 0:
        raise ValueError(f"Snapshot {snapshot_index} has zero voxels.")

    symmetry_per_phase: dict[int, object] = {}
    for ph in data["phase"]:
        pid = int(ph["phase_id"])
        try:
            sym_name = ph["orientation"]["crystal_symmetry_group"]
        except KeyError as e:
            raise KeyError(
                f"Phase {pid}: missing 'orientation.crystal_symmetry_group'."
            ) from e
        if sym_name not in _SYM_NAME_TO_OBJ:
            raise ValueError(
                f"Phase {pid}: unknown crystal_symmetry_group {sym_name!r}. "
                f"Known names: {sorted(_SYM_NAME_TO_OBJ.keys())}"
            )
        symmetry_per_phase[pid] = _SYM_NAME_TO_OBJ[sym_name]

    grain_to_phase = {int(g["grain_id"]): int(g["phase_id"]) for g in grains}
    grain_phase_ids = set(grain_to_phase.values())
    missing_phase_ids = sorted(grain_phase_ids.difference(symmetry_per_phase.keys()))
    if missing_phase_ids:
        raise KeyError(
            "Snapshot grains[] references phase_id value(s) that are absent "
            f"from top-level phase[]: {missing_phase_ids}. "
            "Every grain phase_id must have a matching entry in phase[] so "
            "the crystal symmetry can be resolved before computing "
            "misorientation."
        )

    N = len(voxels)
    eulers = np.empty((N, 3), dtype=np.float64)
    voxel_volume = np.empty(N, dtype=np.float64)
    voxel_index = np.empty((N, 3), dtype=np.int32)
    old_grain_id = np.empty(N, dtype=np.int32)
    phase_id_per_voxel = np.empty(N, dtype=np.int32)
    pos_to_idx: dict[tuple[int, int, int], int] = {}

    missing_grains = set()
    for pos, v in enumerate(voxels):
        ijk = (
            int(v["voxel_index"][0]),
            int(v["voxel_index"][1]),
            int(v["voxel_index"][2]),
        )
        if ijk in pos_to_idx:
            raise ValueError(
                f"Duplicate voxel_index {ijk} encountered at positions "
                f"{pos_to_idx[ijk]} and {pos}."
            )
        pos_to_idx[ijk] = pos
        voxel_index[pos] = ijk
        eulers[pos] = np.asarray(v["orientation"], dtype=np.float64)
        voxel_volume[pos] = float(v["voxel_volume"])

        gid = int(v["grain_id"])
        old_grain_id[pos] = gid
        pid = grain_to_phase.get(gid)
        if pid is None:
            missing_grains.add(gid)
            phase_id_per_voxel[pos] = -1
        else:
            phase_id_per_voxel[pos] = pid

    if missing_grains:
        preview = sorted(missing_grains)[:10]
        suffix = "..." if len(missing_grains) > 10 else ""
        raise KeyError(
            f"{len(missing_grains)} voxel grain_id(s) not present in grains[]: "
            f"{preview}{suffix}"
        )

    unique_phases = np.unique(phase_id_per_voxel)
    is_multi_phase = unique_phases.size > 1

    if connectivity == 6:
        offsets = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
    elif connectivity == 18:
        offsets = [
            (di, dj, dk)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            for dk in (-1, 0, 1)
            if 1 <= abs(di) + abs(dj) + abs(dk) <= 2
        ]
    else:
        offsets = [
            (di, dj, dk)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            for dk in (-1, 0, 1)
            if not (di == 0 and dj == 0 and dk == 0)
        ]

    # Bounding box of the voxel grid (used for periodic wrap).
    i_min = int(voxel_index[:, 0].min())
    i_max = int(voxel_index[:, 0].max())
    j_min = int(voxel_index[:, 1].min())
    j_max = int(voxel_index[:, 1].max())
    k_min = int(voxel_index[:, 2].min())
    k_max = int(voxel_index[:, 2].max())
    i_ext = i_max - i_min + 1
    j_ext = j_max - j_min + 1
    k_ext = k_max - k_min + 1

    eu: list[int] = []
    ev: list[int] = []
    edge_is_periodic_list: list[bool] = []
    # `edge_to_idx` guards against duplicate edges that can arise in the
    # periodic case when an axis has extent <= 2 (both +1 and -1 offsets land
    # on the same neighbor). If the same pair is reachable through a normal
    # and a wrapped offset, it is marked periodic so the viewer can hide it.
    edge_to_idx: dict[tuple[int, int], int] = {}
    for (i, j, k), pos in pos_to_idx.items():
        for di, dj, dk in offsets:
            raw_ni, raw_nj, raw_nk = i + di, j + dj, k + dk
            ni, nj, nk = raw_ni, raw_nj, raw_nk
            is_periodic_edge = False
            if periodic:
                ni = ((ni - i_min) % i_ext) + i_min
                nj = ((nj - j_min) % j_ext) + j_min
                nk = ((nk - k_min) % k_ext) + k_min
                is_periodic_edge = (
                    ni != raw_ni or nj != raw_nj or nk != raw_nk
                )
            nb_pos = pos_to_idx.get((ni, nj, nk))
            if nb_pos is None or nb_pos == pos:
                continue
            a, b = (pos, nb_pos) if pos < nb_pos else (nb_pos, pos)
            existing = edge_to_idx.get((a, b))
            if existing is not None:
                edge_is_periodic_list[existing] = (
                    edge_is_periodic_list[existing] or is_periodic_edge
                )
                continue
            edge_to_idx[(a, b)] = len(eu)
            eu.append(a)
            ev.append(b)
            edge_is_periodic_list.append(is_periodic_edge)

    edges = (
        np.column_stack(
            [np.asarray(eu, dtype=np.int32), np.asarray(ev, dtype=np.int32)]
        )
        if eu
        else np.zeros((0, 2), dtype=np.int32)
    )
    edge_is_periodic = np.asarray(edge_is_periodic_list, dtype=bool)
    E = edges.shape[0]

    if is_multi_phase:
        dominant_phase = int(
            Counter(phase_id_per_voxel.tolist()).most_common(1)[0][0]
        )
        warnings.warn(
            f"Multi-phase data: phases={unique_phases.tolist()}. "
            f"'orientations' uses dominant phase {dominant_phase} symmetry. "
            "Cross-phase edge weights are set to np.inf.",
            stacklevel=2,
        )
        global_sym = symmetry_per_phase[dominant_phase]
    else:
        global_sym = symmetry_per_phase[int(unique_phases[0])]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=np.exceptions.ComplexWarning,
            module=r"quaternion(\.|$)",
        )
        O_global = Orientation.from_euler(
            eulers, symmetry=global_sym, direction="lab2crystal", degrees=False
        )

    weights = np.full(E, np.inf, dtype=np.float64)
    if E > 0:
        same_phase = (
            phase_id_per_voxel[edges[:, 0]] == phase_id_per_voxel[edges[:, 1]]
        )

        if not is_multi_phase:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=np.exceptions.ComplexWarning,
                    module=r"quaternion(\.|$)",
                )
                weights[:] = np.asarray(
                    O_global[edges[:, 0]].angle_with(
                        O_global[edges[:, 1]], degrees=True
                    ),
                    dtype=np.float64,
                )
        else:
            for pid in unique_phases.tolist():
                pid = int(pid)
                idxs = np.where(phase_id_per_voxel == pid)[0]
                if idxs.size == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=np.exceptions.ComplexWarning,
                        module=r"quaternion(\.|$)",
                    )
                    O_p = Orientation.from_euler(
                        eulers[idxs],
                        symmetry=symmetry_per_phase[pid],
                        direction="lab2crystal",
                        degrees=False,
                    )

                global_to_local = -np.ones(N, dtype=np.int64)
                global_to_local[idxs] = np.arange(idxs.size)
                edge_mask = same_phase & (phase_id_per_voxel[edges[:, 0]] == pid)
                if not np.any(edge_mask):
                    continue
                lu = global_to_local[edges[edge_mask, 0]]
                lv = global_to_local[edges[edge_mask, 1]]
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=np.exceptions.ComplexWarning,
                        module=r"quaternion(\.|$)",
                    )
                    weights[edge_mask] = np.asarray(
                        O_p[lu].angle_with(O_p[lv], degrees=True),
                        dtype=np.float64,
                    )

    return {
        "N": int(N),
        "edges": edges,
        "weights": weights,
        "edge_is_periodic": edge_is_periodic,
        "orientations": O_global,
        "voxel_volume": voxel_volume,
        "voxel_index": voxel_index,
        "old_grain_id": old_grain_id,
        "pos_to_idx": pos_to_idx,
        "phase_id_per_voxel": phase_id_per_voxel,
        "symmetry_per_phase": symmetry_per_phase,
        "is_multi_phase": bool(is_multi_phase),
        "eulers": eulers,
        "periodic": bool(periodic),
        "snapshot_index": int(snapshot_index),
        "snapshot_time": snap.get("time"),
        "connectivity": int(connectivity),
    }


@lru_cache(maxsize=12)
def _cached_build_voxel_graph(
    file_path: str,
    snapshot_index: int,
    connectivity: int,
    mtime_ns: int,
    size: int,
) -> dict:
    # mtime_ns and size are cache-key guards; build_voxel_graph reads file_path.
    # RVE_continuity is part of the file contents, so any change to it is
    # already covered by the mtime/size guards.
    _ = (mtime_ns, size)
    return build_voxel_graph(
        file_path, snapshot_index=snapshot_index, connectivity=connectivity
    )


def build_voxel_graph_cached(
    file_path: Union[str, Path],
    *,
    snapshot_index: int = 0,
    connectivity: int = 6,
) -> dict:
    """
    Cached variant of build_voxel_graph.

    The cache key includes the resolved path, file modification time, file size,
    snapshot index, and connectivity. Re-running the same viewer call avoids
    reparsing the large JSON and recomputing all misorientations.
    """
    path = Path(file_path).resolve()
    stat = path.stat()
    return _cached_build_voxel_graph(
        str(path), int(snapshot_index), int(connectivity), stat.st_mtime_ns, stat.st_size
    )


def clear_voxel_graph_cache() -> None:
    """Clear cached graph builds."""
    _cached_build_voxel_graph.cache_clear()


def summarize_voxel_graph(g: dict) -> None:
    """Print a compact summary of a graph dict produced by build_voxel_graph."""
    print("=" * 70)
    print(
        f"Voxel graph summary (snapshot {g['snapshot_index']}, "
        f"time={g['snapshot_time']}, connectivity={g['connectivity']})"
    )
    print("=" * 70)
    print(f"  N (voxels)                : {g['N']}")
    print(f"  E (edges, undirected)     : {g['edges'].shape[0]}")
    print(f"  periodic (RVE_continuity) : {g['periodic']}")
    print(f"  is_multi_phase            : {g['is_multi_phase']}")
    print(f"  phases present            : {sorted(set(g['phase_id_per_voxel'].tolist()))}")
    for pid, sym in g["symmetry_per_phase"].items():
        n_p = int(np.sum(g["phase_id_per_voxel"] == pid))
        print(f"    phase {pid}: symmetry={sym.name!r}, voxels={n_p}")

    vi = g["voxel_index"]
    print(
        "  voxel_index range         : "
        f"i={vi[:, 0].min()}..{vi[:, 0].max()}, "
        f"j={vi[:, 1].min()}..{vi[:, 1].max()}, "
        f"k={vi[:, 2].min()}..{vi[:, 2].max()}"
    )

    w = g["weights"]
    finite = np.isfinite(w)
    if np.any(finite):
        print(
            "  weights (deg)             : "
            f"min={w[finite].min():.4f}, "
            f"mean={w[finite].mean():.4f}, "
            f"median={np.median(w[finite]):.4f}, "
            f"max={w[finite].max():.4f}"
        )
    else:
        print("  weights (deg)             : no finite weights")
    print(f"  cross-phase edges (w=inf) : {int((~finite).sum())}")
    print(f"  voxel_volume sum          : {g['voxel_volume'].sum():.6e}")
    print(f"  unique old_grain_ids      : {len(np.unique(g['old_grain_id']))}")
    print("=" * 70)
