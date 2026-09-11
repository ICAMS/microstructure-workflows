"""
Standalone 3D MiMeDO grain segmentation pipeline.

This file combines the current working pipeline into one module:

    MiMeDO snapshot -> 3D periodic BFS labels
    -> 3D region adjacency graph
    -> boundary-artifact pre-merge
    -> updated MiMeDO JSON

The public entry point is ``segment_microstructure``. By default it writes a
copy, leaving the source file untouched. Set ``write_mode="inplace"`` to write
the segmented grain ids back into the same MiMeDO file.
"""

import argparse
import heapq
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from orix.quaternion import Orientation
    from orix.quaternion.symmetry import Oh
except ImportError:
    Orientation = None
    Oh = None

try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:
    adjusted_rand_score = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _require_segmentation_dependencies():
    """Fail clearly when the active Python environment lacks required packages."""
    missing = []
    if nx is None:
        missing.append("networkx")
    if Orientation is None or Oh is None:
        missing.append("orix")
    if missing:
        raise ImportError(
            "Missing required segmentation package(s): "
            + ", ".join(missing)
            + ". Install them in the Python environment used to run this file."
        )


# ----------------------------------------------------------------------
# 1. 3D periodic neighbor rule
# ----------------------------------------------------------------------
def neighbor_offsets_3d(connectivity=6):
    """Return 3D neighbor offsets for 6, 18, or 26 connectivity."""
    offs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                manhattan = abs(di) + abs(dj) + abs(dk)
                if connectivity == 6 and manhattan == 1:
                    offs.append((di, dj, dk))
                elif connectivity == 18 and manhattan <= 2:
                    offs.append((di, dj, dk))
                elif connectivity == 26:
                    offs.append((di, dj, dk))
    if not offs:
        raise ValueError("connectivity must be one of {6, 18, 26}")
    return offs


def _as_triple(periodic):
    """Accept one bool or a per-axis (px, py, pz) periodicity tuple."""
    if isinstance(periodic, (bool, np.bool_)):
        return (bool(periodic), bool(periodic), bool(periodic))
    px, py, pz = periodic
    return (bool(px), bool(py), bool(pz))


def iter_neighbors_3d(i, j, k, shape, offsets, periodic):
    """Yield valid wrapped/non-wrapped neighbors of one voxel."""
    nx_, ny_, nz_ = shape
    px, py, pz = periodic
    for offset_id, (di, dj, dk) in enumerate(offsets):
        coord = []
        valid = True
        for c, n, p in (
            (i + di, nx_, px),
            (j + dj, ny_, py),
            (k + dk, nz_, pz),
        ):
            if p:
                c %= n
            elif c < 0 or c >= n:
                valid = False
                break
            coord.append(c)
        if valid:
            ni, nj, nk = coord
            if (ni, nj, nk) != (i, j, k):
                yield offset_id, ni, nj, nk


# ----------------------------------------------------------------------
# 2. MiMeDO snapshot adapter
# ----------------------------------------------------------------------
def snapshot_to_grid(snapshot, symmetry=Oh, degrees=False):
    """
    Convert one MiMeDO snapshot into valid voxel mask and flat orientation data.

    Grid shape is derived from voxel_index, not grid_size, so deformed and
    regridded snapshots stay correct.
    """
    voxels = snapshot["voxels"]

    idx = np.array([v["voxel_index"] for v in voxels], dtype=int)
    nx_, ny_, nz_ = (
        int(idx[:, 0].max()),
        int(idx[:, 1].max()),
        int(idx[:, 2].max()),
    )
    shape = (nx_, ny_, nz_)

    eul = np.array([v["orientation"] for v in voxels], dtype=float)
    ori = Orientation.from_euler(eul, symmetry, degrees=degrees)
    q = ori.data

    quat_grid = np.full(shape + (4,), np.nan, dtype=float)
    valid_mask = np.zeros(shape, dtype=bool)
    ii, jj, kk = idx[:, 0] - 1, idx[:, 1] - 1, idx[:, 2] - 1
    quat_grid[ii, jj, kk] = q
    valid_mask[ii, jj, kk] = True

    flat_q = np.nan_to_num(quat_grid.reshape(-1, 4))
    ori_flat = Orientation(flat_q, symmetry)
    return valid_mask, shape, ori_flat


# ----------------------------------------------------------------------
# 3. BFS segmentation
# ----------------------------------------------------------------------
def precompute_neighbor_misorientation(ori_flat, shape, offsets, periodic):
    """Precompute symmetry-reduced misorientation to every neighbor offset."""
    nx_, ny_, nz_ = shape
    px, py, pz = periodic

    I, J, K = np.meshgrid(
        np.arange(nx_), np.arange(ny_), np.arange(nz_), indexing="ij"
    )
    miso_table = np.empty((len(offsets), nx_, ny_, nz_), dtype=float)

    for offset_id, (di, dj, dk) in enumerate(offsets):
        valid = np.ones(shape, dtype=bool)
        nbr_coords = []
        for C, d, n, p in (
            (I, di, nx_, px),
            (J, dj, ny_, py),
            (K, dk, nz_, pz),
        ):
            Cn = C + d
            if p:
                Cn = Cn % n
            else:
                valid &= (Cn >= 0) & (Cn < n)
                Cn = np.clip(Cn, 0, n - 1)
            nbr_coords.append(Cn)

        Ni, Nj, Nk = nbr_coords
        nbr_lin = ((Ni * ny_ + Nj) * nz_ + Nk).ravel()
        ang = np.asarray(ori_flat.angle_with(ori_flat[nbr_lin]), dtype=float)
        ang = ang.reshape(shape)
        ang[~valid] = np.inf
        miso_table[offset_id] = ang

    return miso_table


def segment_snapshot_bfs(
    valid_mask,
    shape,
    ori_flat,
    tolerance,
    connectivity=6,
    periodic=True,
):
    """
    Identify grains by BFS flood-fill on the 3D voxel-index grid.

    A neighbor joins the current grain when its local pairwise misorientation
    to the current voxel is <= tolerance.
    """
    nx_, ny_, nz_ = shape
    offsets = neighbor_offsets_3d(connectivity)
    periodic = _as_triple(periodic)
    miso_table = precompute_neighbor_misorientation(
        ori_flat, shape, offsets, periodic
    )

    labels = np.zeros(shape, dtype=int)
    visited = np.zeros(shape, dtype=bool)

    current_label = 1
    for i0 in range(nx_):
        for j0 in range(ny_):
            for k0 in range(nz_):
                if visited[i0, j0, k0] or not valid_mask[i0, j0, k0]:
                    continue

                queue = deque([(i0, j0, k0)])
                visited[i0, j0, k0] = True
                labels[i0, j0, k0] = current_label

                while queue:
                    i, j, k = queue.popleft()
                    for offset_id, ni, nj, nk in iter_neighbors_3d(
                        i, j, k, shape, offsets, periodic
                    ):
                        if visited[ni, nj, nk] or not valid_mask[ni, nj, nk]:
                            continue
                        if miso_table[offset_id, i, j, k] > tolerance:
                            continue
                        visited[ni, nj, nk] = True
                        labels[ni, nj, nk] = current_label
                        queue.append((ni, nj, nk))

                current_label += 1

    return labels, current_label - 1


def old_labels_grid(snapshot, shape):
    """Read inherited voxel grain_id values onto a grid for ARI comparison."""
    g = np.zeros(shape, dtype=int)
    for v in snapshot["voxels"]:
        i, j, k = v["voxel_index"]
        g[i - 1, j - 1, k - 1] = v["grain_id"]
    return g


def snapshot_grid_status(snapshot):
    """Return normalized MiMeDO grid status for one snapshot."""
    return str(snapshot.get("grid", {}).get("status", "")).strip().lower()


def _snapshot_segmentation_plan(snapshots, include_reference=True):
    """
    Segment snapshot 0 as ARI-only reference and deformed snapshots as writes.
    """
    plan = []
    skipped = []
    for s_idx, snap in enumerate(snapshots):
        status = snapshot_grid_status(snap)
        if s_idx == 0 and include_reference:
            plan.append(
                {
                    "snapshot": s_idx,
                    "snap": snap,
                    "status": status,
                    "write_labels": False,
                    "role": "reference ARI only",
                }
            )
        elif status == "deformed":
            plan.append(
                {
                    "snapshot": s_idx,
                    "snap": snap,
                    "status": status,
                    "write_labels": True,
                    "role": "deformed write target",
                }
            )
        else:
            skipped.append(
                {
                    "snapshot": s_idx,
                    "status": status or "missing",
                    "reason": "status is not deformed",
                }
            )
    return plan, skipped


def _iter_with_progress(items, desc="Segmenting snapshots", enabled=True):
    """Yield items with tqdm when available, otherwise draw a compact text bar."""
    if not enabled:
        yield from items
        return

    total = len(items)
    if total == 0:
        return

    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc, unit="snapshot")
        return

    width = 32

    def draw(done):
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r{desc}: |{bar}| {done}/{total}")
        sys.stdout.flush()

    draw(0)
    for done, item in enumerate(items, start=1):
        yield item
        draw(done)
    sys.stdout.write("\n")
    sys.stdout.flush()


# ----------------------------------------------------------------------
# 4. 3D RAG and boundary-artifact merge
# ----------------------------------------------------------------------
_PROPER_OPS_CACHE = {}


def _proper_sym_ops(sym):
    """
    Cached proper-rotation operators (quaternions) for a crystal symmetry.

    orix recomputes ``sym.proper_subgroup`` from scratch on every access
    (~16 ms for Oh). ``mean_orientation_data`` is called once per grain and once
    per merge, so that single attribute access dominated the whole pipeline.
    Caching by object identity makes it a one-time cost. The returned operators
    are identical, so results are unchanged.
    """
    key = id(sym)
    ops = _PROPER_OPS_CACHE.get(key)
    if ops is None:
        proper = sym.proper_subgroup if getattr(sym, "contains_inversion", False) else sym
        ops = np.asarray(proper.data, dtype=float)
        _PROPER_OPS_CACHE[key] = ops
    return ops


def mean_orientation_data(pixel_orientations, sym):
    """Normalized quaternion mean after symmetry-equivalent alignment."""
    quat = np.asarray(pixel_orientations, dtype=float)
    quat = quat.reshape((-1, quat.shape[-1]))
    quat = quat / np.linalg.norm(quat, axis=1)[:, None]

    ref = quat[0]
    sym_ops = _proper_sym_ops(sym)

    best_quat = quat.copy()
    best_score = np.abs(best_quat @ ref)

    for sym_op in sym_ops:
        w1, x1, y1, z1 = sym_op
        w2, x2, y2, z2 = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        candidate = np.column_stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )
        candidate = candidate / np.linalg.norm(candidate, axis=1)[:, None]
        score = np.abs(candidate @ ref)
        update = score > best_score
        best_quat[update] = candidate[update]
        best_score[update] = score[update]

    signs = np.sign(best_quat @ ref)
    signs[signs == 0.0] = 1.0
    best_quat = best_quat * signs[:, None]

    accumulator = best_quat.T @ best_quat
    eigvals, eigvecs = np.linalg.eigh(accumulator)
    mean_quat = eigvecs[:, int(np.argmax(eigvals))]
    if mean_quat[0] < 0.0:
        mean_quat = -mean_quat
    return mean_quat / np.linalg.norm(mean_quat)


def summarize_labels(label_array, rotations, sym, wanted_labels=None):
    """Build per-grain node attributes for a label grid."""
    lab = label_array.ravel()
    rot = rotations

    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]
    uniq, starts, counts = np.unique(
        lab_sorted, return_index=True, return_counts=True
    )

    rot_sorted = rot[order]
    sq = rot_sorted * rot_sorted
    sums_sq = np.add.reduceat(sq, starts, axis=0)
    sums = np.add.reduceat(rot_sorted, starts, axis=0)
    means_component = sums / counts[:, None]
    vars_ = sums_sq / counts[:, None] - means_component**2
    stds = np.sqrt(np.maximum(vars_, 0.0))

    pixels_list = np.split(order, starts[1:])

    if wanted_labels is None:
        valid = uniq > 0
        labels_out = uniq[valid]
        idx = np.arange(len(uniq))[valid]
    else:
        labels_out = np.asarray(wanted_labels)
        pos = {int(label): i for i, label in enumerate(uniq)}
        idx = np.array([pos[int(label)] for label in labels_out], dtype=int)

    nodes = []
    for label, i in zip(labels_out, idx):
        pix = pixels_list[i]
        nodes.append(
            (
                int(label),
                {
                    "npix": int(counts[i]),
                    "pixels": pix,
                    "ori_av": mean_orientation_data(rot[pix], sym),
                    "ori_std": stds[i],
                },
            )
        )
    return nodes


def _adjacent_label_pairs(label_array, offsets, periodic):
    """Return all unordered touching label pairs in the 3D grid."""
    nx_, ny_, nz_ = label_array.shape
    px, py, pz = periodic
    a_all, b_all = [], []
    for di, dj, dk in offsets:
        nb = np.roll(label_array, shift=(-di, -dj, -dk), axis=(0, 1, 2))
        valid = np.ones(label_array.shape, dtype=bool)
        if not px:
            if di == 1:
                valid[nx_ - 1, :, :] = False
            elif di == -1:
                valid[0, :, :] = False
        if not py:
            if dj == 1:
                valid[:, ny_ - 1, :] = False
            elif dj == -1:
                valid[:, 0, :] = False
        if not pz:
            if dk == 1:
                valid[:, :, nz_ - 1] = False
            elif dk == -1:
                valid[:, :, 0] = False
        mask = valid & (label_array > 0) & (nb > 0) & (label_array != nb)
        a_all.append(label_array[mask])
        b_all.append(nb[mask])
    if not a_all:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.concatenate(a_all), np.concatenate(b_all)


def build_grain_rag_3d(
    label_array,
    rot,
    sym,
    connectivity=6,
    periodic=True,
    spacing=(1.0, 1.0, 1.0),
):
    """Build a 3D periodic region adjacency graph from grain labels."""
    offsets = neighbor_offsets_3d(connectivity)
    per = _as_triple(periodic)
    dx, dy, dz = spacing

    nodes = summarize_labels(label_array, rot, sym)
    G = nx.Graph(
        label_map=label_array,
        symmetry=sym,
        rotations=rot,
        dx=dx,
        dy=dy,
        dz=dz,
        connectivity=connectivity,
        periodic=per,
    )
    G.add_nodes_from(nodes)

    a, b = _adjacent_label_pairs(label_array, offsets, per)
    G.add_edges_from(zip(a.tolist(), b.tolist()))
    return G


def merge_nodes(G, node1, node2):
    """Merge node1 into node2 and update graph labels in place."""
    G.nodes[node2]["pixels"] = np.concatenate(
        (G.nodes[node1]["pixels"], G.nodes[node2]["pixels"])
    )
    ntot = G.nodes[node1]["npix"] + G.nodes[node2]["npix"]

    if "rotations" in G.graph:
        pix = G.nodes[node2]["pixels"]
        G.nodes[node2]["ori_av"] = mean_orientation_data(
            G.graph["rotations"][pix], G.graph["symmetry"]
        )
    else:
        ori_av = (
            G.nodes[node1]["ori_av"] * G.nodes[node1]["npix"]
            + G.nodes[node2]["ori_av"] * G.nodes[node2]["npix"]
        ) / ntot
        G.nodes[node2]["ori_av"] = ori_av / np.linalg.norm(ori_av)

    G.nodes[node2]["npix"] = ntot

    for neigh in G.adj[node1]:
        if node2 != neigh:
            G.add_edge(node2, neigh)

    G.remove_node(node1)
    G.graph["label_map"][G.graph["label_map"] == node1] = node2


def find_largest_neighbor(G, node):
    """Return the largest neighbor by voxel count."""
    size_ln, num_ln = 0, -1
    for neigh in G.adj[node]:
        if G.nodes[neigh]["npix"] > size_ln:
            size_ln = G.nodes[neigh]["npix"]
            num_ln = neigh
    if num_ln < 0:
        raise ValueError(f"Grain {node} has no neighbors.")
    if num_ln == node:
        raise ValueError(f"Corrupted graph with circular edges: {node, num_ln}.")
    return num_ln


def find_sim_neighbor(G, node):
    """Return the most orientation-similar neighbor."""
    sym = G.graph["symmetry"]
    ori0 = Orientation(G.nodes[node]["ori_av"], sym)
    angles, neighbors = [], []
    for neigh in G.adj[node]:
        ang = ori0.angle_with(Orientation(G.nodes[neigh]["ori_av"], sym))[0]
        angles.append(ang)
        neighbors.append(neigh)
    k = int(np.argmin(angles))
    return neighbors[k], angles[k]


def _boundary_voxel_mask(label_array, offsets, periodic):
    """Mark voxels touching another label or a non-periodic domain boundary."""
    nx_, ny_, nz_ = label_array.shape
    px, py, pz = periodic
    is_boundary = np.zeros(label_array.shape, dtype=bool)
    for di, dj, dk in offsets:
        nb = np.roll(label_array, shift=(-di, -dj, -dk), axis=(0, 1, 2))
        valid = np.ones(label_array.shape, dtype=bool)
        if not px:
            if di == 1:
                valid[nx_ - 1, :, :] = False
            elif di == -1:
                valid[0, :, :] = False
        if not py:
            if dj == 1:
                valid[:, ny_ - 1, :] = False
            elif dj == -1:
                valid[:, 0, :] = False
        if not pz:
            if dk == 1:
                valid[:, :, nz_ - 1] = False
            elif dk == -1:
                valid[:, :, 0] = False
        is_boundary |= (valid & (nb != label_array)) | (~valid)
    is_boundary &= label_array > 0
    return is_boundary


def _batch_sim_neighbors(G, node_ids):
    """Return best orientation-similar neighbor for many nodes."""
    sym = G.graph["symmetry"]
    owners, neighs, q_self, q_neigh = [], [], [], []
    for candidate in node_ids:
        for neigh in G.adj[candidate]:
            owners.append(candidate)
            neighs.append(neigh)
            q_self.append(G.nodes[candidate]["ori_av"])
            q_neigh.append(G.nodes[neigh]["ori_av"])
    if not owners:
        return {}

    angles = np.asarray(
        Orientation(np.array(q_self), sym).angle_with(
            Orientation(np.array(q_neigh), sym)
        ),
        dtype=float,
    )
    best = {}
    for candidate, neigh, angle in zip(owners, neighs, angles):
        if candidate not in best or angle < best[candidate][1]:
            best[candidate] = (neigh, float(angle))
    return best


def _node_boundary_fraction(pixels, label_map, offsets, periodic):
    """
    Boundary-voxel mask for one node's voxels, computed on demand from the
    current ``label_map``.

    Identical definition to ``_boundary_voxel_mask`` restricted to these voxels
    (a voxel is a boundary voxel if any neighbor has a different label, or lies
    off-grid on a non-periodic axis), so ``mask.mean()`` matches the per-pass
    ``bmask[pixels].mean()`` exactly.
    """
    shape = label_map.shape
    nx_, ny_, nz_ = shape
    px, py, pz = periodic
    ci, cj, ck = np.unravel_index(pixels, shape)
    self_lab = label_map[ci, cj, ck]
    is_b = np.zeros(len(pixels), dtype=bool)
    all_periodic = px and py and pz
    for di, dj, dk in offsets:
        if all_periodic:
            # fast path: every wrapped neighbor is valid, so no masking/clipping;
            # axes with a zero offset component keep the same coordinate array
            ni = (ci + di) % nx_ if di else ci
            nj = (cj + dj) % ny_ if dj else cj
            nk = (ck + dk) % nz_ if dk else ck
            is_b |= label_map[ni, nj, nk] != self_lab
            continue
        valid = np.ones(len(pixels), dtype=bool)
        ni = ci + di
        if px:
            ni = ni % nx_
        else:
            off = (ni < 0) | (ni >= nx_)
            valid &= ~off
            ni = np.clip(ni, 0, nx_ - 1)
        nj = cj + dj
        if py:
            nj = nj % ny_
        else:
            off = (nj < 0) | (nj >= ny_)
            valid &= ~off
            nj = np.clip(nj, 0, ny_ - 1)
        nk = ck + dk
        if pz:
            nk = nk % nz_
        else:
            off = (nk < 0) | (nk >= nz_)
            valid &= ~off
            nk = np.clip(nk, 0, nz_ - 1)
        nb_lab = label_map[ni, nj, nk]
        is_b |= (valid & (nb_lab != self_lab)) | (~valid)
    return is_b


def merge_boundary_artifact_nodes_3d(
    G,
    gs_min,
    connectivity=6,
    periodic=True,
    npix_factor=2.0,
    boundary_fraction_min=0.85,
):
    """
    Merge small, mostly-boundary sliver grains into their most similar neighbor.

    Incremental implementation: produces the IDENTICAL merge sequence (and thus
    identical labels) as a per-pass rescan, but after each merge only the
    affected nodes -- the merge target and its neighbors -- are re-evaluated,
    instead of rescanning every node and recomputing every sim-neighbor on every
    merge. Candidates are kept in a lazy priority heap keyed exactly as the old
    sort: ``(-boundary_fraction, npix, angle, node_id)``.
    """
    offsets = neighbor_offsets_3d(connectivity)
    per = _as_triple(periodic)
    label_map = G.graph["label_map"]
    shape = label_map.shape
    npix_limit = max(int(npix_factor * gs_min), 40)

    version = {}
    heap = []
    bf_cache = {}  # node_id -> current boundary fraction (only for candidate-sized nodes)

    def node_bf(node_id):
        return float(_node_boundary_fraction(G.nodes[node_id]["pixels"],
                                             label_map, offsets, per).mean())

    def push(node_ids):
        """(Re)evaluate sim-neighbor for nodes whose bf is already in bf_cache and
        push fresh heap entries (one batched orix call for all of them)."""
        cand = [n for n in node_ids
                if n in bf_cache
                and n in G.nodes
                and int(G.nodes[n]["npix"]) <= npix_limit
                and G.degree[n] > 0
                and bf_cache[n] >= boundary_fraction_min]
        if not cand:
            return
        best = _batch_sim_neighbors(G, cand)
        for nid in cand:
            if nid not in best:
                continue
            sim_neigh, angle = best[nid]
            version[nid] = version.get(nid, 0) + 1
            heapq.heappush(
                heap,
                (-bf_cache[nid], int(G.nodes[nid]["npix"]), angle, int(nid),
                 int(sim_neigh), version[nid]),
            )

    # initial boundary fractions from one global mask pass (cheap, vectorised)
    bmask = _boundary_voxel_mask(label_map, offsets, per).ravel()
    for nid in G.nodes:
        if int(G.nodes[nid]["npix"]) <= npix_limit and G.degree[nid] > 0:
            bf_cache[nid] = float(bmask[G.nodes[nid]["pixels"]].mean())
    push(list(bf_cache))

    debug_records = []
    while heap:
        neg_bf, npix, angle, node_id, sim_neigh, ver = heapq.heappop(heap)
        if version.get(node_id) != ver:
            continue  # stale: node changed (or was removed) since this was pushed
        if node_id not in G.nodes or sim_neigh not in G.nodes:
            continue

        # a non-stale entry reflects the current global-best candidate; its key
        # (bf, npix, angle) is still current, so reuse bf_cache for the record
        pix = G.nodes[node_id]["pixels"]
        bf = bf_cache[node_id]
        coords = np.array(np.unravel_index(pix, shape))
        ext = coords.max(axis=1) - coords.min(axis=1) + 1
        boundary_count = int(round(bf * len(pix)))
        debug_records.append(
            {
                "node": int(node_id),
                "reason": "boundary_artifact_premerge",
                "target": int(sim_neigh),
                "angle_rad": float(angle),
                "angle_deg": float(np.degrees(angle)),
                "node_npix": int(G.nodes[node_id]["npix"]),
                "target_npix": int(G.nodes[sim_neigh]["npix"]),
                "boundary_pixels": boundary_count,
                "boundary_fraction": float(bf),
                "neighbor_count": int(G.degree[node_id]),
                "bbox_fill_fraction": float(len(pix) / float(np.prod(ext))),
                "npix_limit": int(npix_limit),
                "will_merge": True,
            }
        )

        version[node_id] = version.get(node_id, 0) + 1  # invalidate the removed node
        bf_cache.pop(node_id, None)
        merge_nodes(G, node_id, sim_neigh)

        # Merging node_id -> sim_neigh changes the boundary fraction of the TARGET
        # only (a neighbour with a different label stays a different label), and
        # changes sim-neighbour angles for the target and all its neighbours
        # (the target's mean orientation moved).
        if (sim_neigh in G.nodes and int(G.nodes[sim_neigh]["npix"]) <= npix_limit
                and G.degree[sim_neigh] > 0):
            bf_cache[sim_neigh] = node_bf(sim_neigh)
        else:
            bf_cache.pop(sim_neigh, None)

        affected = {sim_neigh}
        affected.update(G.adj[sim_neigh])
        for a in affected:
            version[a] = version.get(a, 0) + 1  # invalidate stale entries first
        push(affected)

    return debug_records


# ----------------------------------------------------------------------
# 5. Pipeline, summary, and export
# ----------------------------------------------------------------------
def _run_segmentation_pipeline(
    data,
    max_angle_deg=3.0,
    connectivity=6,
    symmetry=Oh,
    gs_min=10.0,
    include_reference=True,
    progress=True,
    verbose=True,
):
    """Run BFS -> RAG -> boundary pre-merge on selected snapshots."""
    _require_segmentation_dependencies()
    if symmetry is None:
        symmetry = Oh

    periodic = bool(data.get("RVE_continuity", False))
    tolerance = np.deg2rad(max_angle_deg)
    snapshots = data["microstructure"]
    plan, skipped = _snapshot_segmentation_plan(
        snapshots, include_reference=include_reference
    )

    results = []
    for item in _iter_with_progress(plan, enabled=progress):
        s_idx = item["snapshot"]
        snap = item["snap"]
        valid, shape, ori_flat = snapshot_to_grid(snap, symmetry=symmetry)
        rot = ori_flat.data

        bfs_labels, n_bfs = segment_snapshot_bfs(
            valid,
            shape,
            ori_flat,
            tolerance,
            connectivity=connectivity,
            periodic=periodic,
        )

        G = build_grain_rag_3d(
            bfs_labels.copy(),
            rot,
            symmetry,
            connectivity=connectivity,
            periodic=periodic,
        )
        records = merge_boundary_artifact_nodes_3d(
            G, gs_min, connectivity=connectivity, periodic=periodic
        )

        labels = G.graph["label_map"]
        old = old_labels_grid(snap, shape)
        mask = valid
        ari = (
            adjusted_rand_score(old[mask].ravel(), labels[mask].ravel())
            if adjusted_rand_score is not None
            else float("nan")
        )

        results.append(
            {
                "snapshot": s_idx,
                "status": item["status"],
                "write_labels": item["write_labels"],
                "role": item["role"],
                "shape": shape,
                "labels": labels,
                "n_grains_bfs": int(n_bfs),
                "n_grains_after_boundary": int(len(G.nodes)),
                "n_boundary_merges": int(len(records)),
                "n_grains_old": int(len(np.unique(old[mask]))),
                "ARI_vs_old": ari,
            }
        )

    return results, skipped


def print_snapshot_summary(results):
    """Print the per-snapshot segmentation summary table."""
    print(
        f"\n{'snap':>4} {'status':>10} {'mode':>8} {'grid':>12} "
        f"{'BFS':>6} {'final':>6} {'merges':>7} {'old':>5} {'ARI':>7}"
    )
    for result in results:
        nx_, ny_, nz_ = result["shape"]
        mode = "write" if result["write_labels"] else "ARI-only"
        print(
            f"{result['snapshot']:>4} {result['status']:>10} {mode:>8} "
            f"{f'{nx_}x{ny_}x{nz_}':>12} "
            f"{result['n_grains_bfs']:>6} "
            f"{result['n_grains_after_boundary']:>6} "
            f"{result['n_boundary_merges']:>7} "
            f"{result['n_grains_old']:>5} "
            f"{result['ARI_vs_old']:>7.3f}"
        )


def _bunge_to_quat(eul):
    """Convert (N, 3) Bunge Euler angles in radians to unit quaternions."""
    phi1, Phi, phi2 = eul[:, 0], eul[:, 1], eul[:, 2]
    c, s = np.cos(Phi / 2.0), np.sin(Phi / 2.0)
    sigma, delta = (phi1 + phi2) / 2.0, (phi1 - phi2) / 2.0
    return np.stack(
        [
            c * np.cos(sigma),
            s * np.cos(delta),
            s * np.sin(delta),
            c * np.sin(sigma),
        ],
        axis=1,
    )


def _representative_orientation(eul):
    """Pick the member Euler triple closest to the volume-mean quaternion."""
    q = _bunge_to_quat(eul)
    q[(q @ q[0]) < 0] *= -1.0
    mean = q.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0.0:
        return eul[0]
    idx = int(np.argmax(np.abs(q @ (mean / norm))))
    return eul[idx]


def _default_output_path(input_path):
    """Return the default copy path for a segmented MiMeDO file."""
    input_path = Path(input_path)
    if input_path.name == "MiMeDO.json":
        return input_path.with_name("MiMeDO_Updated.json")
    return input_path.with_name(f"{input_path.stem}_segmented{input_path.suffix}")


def write_segmented_mimedo_json(
    results,
    src_path,
    output_path=None,
    write_mode="copy",
):
    """
    Write segmented grain ids into a MiMeDO-format JSON.

    Parameters
    ----------
    results : list
        Result dictionaries from the segmentation pipeline.
    src_path : str or Path
        Source MiMeDO JSON.
    output_path : str or Path or None
        Destination path when write_mode is "copy". If omitted, a default copy
        path is chosen beside the source.
    write_mode : {"copy", "inplace"}
        "copy" leaves src_path untouched. "inplace" writes back to src_path.
    """
    src_path = Path(src_path)
    if write_mode not in {"copy", "inplace"}:
        raise ValueError('write_mode must be "copy" or "inplace"')

    if write_mode == "inplace":
        if output_path is not None and Path(output_path).resolve() != src_path.resolve():
            raise ValueError("output_path must be omitted or equal to src_path in inplace mode")
        target_path = src_path
    else:
        target_path = Path(output_path) if output_path else _default_output_path(src_path)

    with src_path.open() as f:
        data = json.load(f)

    snapshots = data["microstructure"]
    by_snap = {result["snapshot"]: result for result in results}

    for s_idx, snap in enumerate(snapshots):
        if s_idx not in by_snap:
            continue

        result = by_snap[s_idx]
        should_write = result.get(
            "write_labels",
            s_idx != 0 and snapshot_grid_status(snap) == "deformed",
        )
        if not should_write:
            continue

        labels = result["labels"]
        members = {}
        for voxel in snap["voxels"]:
            i, j, k = voxel["voxel_index"]
            label = int(labels[i - 1, j - 1, k - 1])
            voxel["grain_id"] = label
            members.setdefault(label, []).append(voxel)

        new_grains = []
        for label in sorted(members):
            mem = members[label]
            volume = float(sum(v.get("voxel_volume", 0.0) for v in mem))
            eul = np.array([v["orientation"] for v in mem], dtype=float)
            ori = _representative_orientation(eul)
            new_grains.append(
                {
                    "grain_id": int(label),
                    "phase_id": 0,
                    "grain_volume": volume,
                    "orientation": [float(x) for x in ori],
                }
            )
        snap["grains"] = new_grains

    with target_path.open("w") as f:
        json.dump(data, f)

    return str(target_path)


class SegmentationResult(dict):
    """
    Return value of :func:`segment_microstructure`.

    Behaves exactly like the previous plain dict (``out["results"]``,
    ``out["output_path"]``, ...), but its ``repr`` is a short one-liner so a
    notebook cell that ends on the call does not echo the full per-voxel label
    arrays. Use ``out["results"]`` to access the arrays as before.
    """

    def __repr__(self):
        return (
            f"SegmentationResult(snapshots={len(self.get('results', []))}, "
            f"write_targets={len(self.get('write_targets', []))}, "
            f"output_path={self.get('output_path')!r})"
        )


def segment_microstructure(
    input_path="MiMeDO.json",
    output_path=None,
    write_mode="copy",
    max_angle_deg=3.0,
    connectivity=6,
    gs_min=10.0,
    symmetry=Oh,
    include_reference=True,
    progress=True,
    verbose=True,
):
    """
    Run the full current segmentation workflow and write a MiMeDO result file.

    This is the one function intended to be called from notebooks or scripts.

    Parameters
    ----------
    input_path : str or Path
        MiMeDO JSON to segment.
    output_path : str or Path or None
        Destination when write_mode="copy". If omitted, defaults to
        MiMeDO_Updated.json for MiMeDO.json, otherwise <stem>_segmented.json.
    write_mode : {"copy", "inplace"}
        Choose whether to write a MiMeDO-format copy or overwrite input_path.
    max_angle_deg : float
        BFS misorientation tolerance in degrees.
    connectivity : int
        3D connectivity, one of 6, 18, or 26.
    gs_min : float
        Minimum grain size parameter for boundary-artifact pre-merge.
    symmetry : orix symmetry
        Crystal symmetry. Default is cubic Oh.
    include_reference : bool
        Segment snapshot 0 as ARI-only reference, never written.
    progress : bool
        Show progress bar/text if possible.
    verbose : bool
        Print target counts, per-snapshot summary, and output path.

    Returns
    -------
    dict
        {"results", "skipped", "output_path", "write_targets"}.
    """
    _require_segmentation_dependencies()
    if symmetry is None:
        symmetry = Oh

    input_path = Path(input_path)
    with input_path.open() as f:
        data = json.load(f)

    results, skipped = _run_segmentation_pipeline(
        data,
        max_angle_deg=max_angle_deg,
        connectivity=connectivity,
        symmetry=symmetry,
        gs_min=gs_min,
        include_reference=include_reference,
        progress=progress,
        verbose=verbose,
    )

    saved_path = write_segmented_mimedo_json(
        results,
        src_path=input_path,
        output_path=output_path,
        write_mode=write_mode,
    )

    if verbose:
        print_snapshot_summary(results)
        print(f"\nwrote {saved_path}")
        print(
            "write targets:",
            [result["snapshot"] for result in results if result["write_labels"]],
        )

    return SegmentationResult(
        {
            "results": results,
            "skipped": skipped,
            "output_path": saved_path,
            "write_targets": [
                result["snapshot"] for result in results if result["write_labels"]
            ],
        }
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment MiMeDO microstructure snapshots with 3D BFS + RAG cleanup."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="MiMeDO.json",
        help="MiMeDO JSON file to segment.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path when --write-mode copy is used.",
    )
    parser.add_argument(
        "--write-mode",
        choices=("copy", "inplace"),
        default="copy",
        help="Write a copy or write back to the same MiMeDO JSON.",
    )
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=3.0,
        help="Misorientation tolerance in degrees.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(6, 18, 26),
        default=6,
        help="3D neighbor connectivity.",
    )
    parser.add_argument(
        "--gs-min",
        type=float,
        default=10.0,
        help="Minimum grain size parameter for boundary-artifact pre-merge.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Do not segment snapshot 0 as ARI-only reference.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output.",
    )
    return parser.parse_args(argv)


def _main(argv=None):
    args = _parse_args(argv)
    segment_microstructure(
        input_path=args.input_path,
        output_path=args.output,
        write_mode=args.write_mode,
        max_angle_deg=args.max_angle_deg,
        connectivity=args.connectivity,
        gs_min=args.gs_min,
        include_reference=not args.no_reference,
        progress=not args.no_progress,
        verbose=True,
    )


if __name__ == "__main__":
    _main()
