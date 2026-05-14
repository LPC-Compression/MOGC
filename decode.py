#!/usr/bin/env python3
"""
decode.py — Optimised Octree Entropy Decoder
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchac
from pathlib import Path

# ---------------------------------------------------------------------------
#  Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)  # pipelines/ → project root
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

_pybind_dir = os.path.join(_ROOT_DIR, "utils", "pybind")
if _pybind_dir not in sys.path:
    sys.path.append(_pybind_dir)

from models.prob_model.self_prob_model import SelfOrderProbModel
from models.prob_model.self_prob_model import sort_use_indices_3dim
from utils.octree_preproc_uitls import cylin2cart
from src.feature_processor import _process_features
from src.model_loader import load_model
from data.io import save_point_cloud

# ======================================================================
#  Tiny helpers
# ======================================================================

def dec2binAry(dec: np.ndarray, N: int) -> np.ndarray:
    """dec → (M, N) bit array (MSB first), e.g. occupancy codes → 8 bits."""
    dec = np.asarray(dec, dtype=np.int64)
    dec = np.atleast_1d(dec)
    out = np.zeros((len(dec), N), dtype=np.int32)
    for b in range(N):
        out[:, N - 1 - b] = (dec >> b) & 1
    return out


def _is_node_pruned(center_x: np.ndarray, depth: int, max_r: float, max_depth: int) -> np.ndarray:
    """Cylin pruning check matching C++ build_octree logic (octreeutils.cpp:222-228)."""
    if max_r <= 0:
        return np.zeros(len(center_x), dtype=bool)
    inner = (center_x <= max_r / 4) & (depth >= max_depth - 2)
    middle = (center_x > max_r / 4) & (center_x <= max_r / 2) & (depth >= max_depth - 1)
    return inner | middle


# ======================================================================
#  Octree data structure  (fully array‑based — no Python objects)
# ======================================================================
class LevelNodeData:
    """Flat storage for nodes of a single octree level."""
    __slots__ = ("centers", "octants", "is_leaf", "parent_ids",
                 "extents", "z_extents", "curind")

    def __init__(self, N: int = 0):
        self.centers    = np.empty((N, 3), dtype=np.float64)
        self.octants    = np.zeros(N, dtype=np.uint8)
        self.is_leaf    = np.zeros(N, dtype=bool)
        self.parent_ids = np.full(N, -1, dtype=np.int64)
        self.extents    = np.zeros(N, dtype=np.float64)
        self.z_extents  = np.zeros(N, dtype=np.float64)
        self.curind     = np.zeros(N, dtype=np.int32)


# offsets for the 8 child octants in (x, y, z) order
_OCTANT_OFFSETS = np.array([
    [-1, -1, -1],  # 0: x-, y-, z-
    [ 1, -1, -1],  # 1: x+, y-, z-
    [-1,  1, -1],  # 2: x-, y+, z-
    [ 1,  1, -1],  # 3: x+, y+, z-
    [-1, -1,  1],  # 4: x-, y-, z+
    [ 1, -1,  1],  # 5: x+, y-, z+
    [-1,  1,  1],  # 6: x-, y+, z+
    [ 1,  1,  1],  # 7: x+, y+, z+
], dtype=np.float64)


def _build_base_levels(
    metadata: dict,
    node_numlist: list,
) -> list:
    """
    Build L0 … Lmax base octree from metadata (root center, extent, z_extent).
    Returns List[LevelNodeData] with all nodes initialised,
    octants default to 0.
    """
    oct_center = np.array(metadata.get("oct_center", [0.0, 0.0, 0.0]),
                          dtype=np.float64)
    quant_size = float(metadata.get("quant_size", 50.0))
    z_rate = float(metadata.get("z_rate", 1.0))
    lidar_level = int(metadata["lidar_level"])
    root_occupancy = int(metadata["root_occupancy"])  # raw oct value

    total_levels = lidar_level + 1  # L0 … Lmax
    decoded_octree = [LevelNodeData(0) for _ in range(total_levels)]

    # --- L0: root node -----------------------------------------------
    root_cx = oct_center[0]
    root_cy = oct_center[1]
    root_cz = oct_center[2]
    root_ext = quant_size
    root_z_ext = quant_size if z_rate == 1.0 else quant_size * z_rate

    decoded_octree[0] = LevelNodeData(1)
    decoded_octree[0].centers[0]   = [root_cx, root_cy, root_cz]
    decoded_octree[0].octants[0]   = root_occupancy
    decoded_octree[0].is_leaf[0]   = (root_occupancy == 0)
    decoded_octree[0].parent_ids[0]= -1
    decoded_octree[0].extents[0]   = root_ext
    decoded_octree[0].z_extents[0] = root_z_ext
    decoded_octree[0].curind[0]    = 0

    node_numlist[0] = 1

    return decoded_octree


# ======================================================================
#  Vectorised expand_children
# ======================================================================

def _expand_children(decoded_octree: list, level: int, node_numlist: list):
    """
    Vectorised: given level L nodes with ``octants`` set,
    create children at L+1 in one numpy operation.
    """
    parent = decoded_octree[level]
    Np = len(parent.octants)

    if Np == 0:
        return

    # --- Which parents have non‑zero occupancy -------------------------
    occ = parent.octants.astype(np.int32)
    bin_occ = dec2binAry(occ, 8)  # (Np, 8)  0 = empty, 1 = occupied
    child_mask = bin_occ > 0       # (Np, 8)

    N_child = int(child_mask.sum())
    if N_child == 0:
        return

    # --- Build child level ---------------------------------------------
    child = LevelNodeData(N_child)

    parent_ids_flat, octant_ids = np.where(child_mask)
    # parent_ids_flat, octant_ids are both (N_child,)

    child.parent_ids = parent_ids_flat.astype(np.int64)
    child.curind = octant_ids.astype(np.int32)
    child.octants = np.zeros(N_child, dtype=np.uint8)  # unknown occupancy until decoded

    # centres = parent_centre + offset * (child_half_extent)
    p_ext = parent.extents[parent_ids_flat]        # (N_child,)
    p_z_ext = parent.z_extents[parent_ids_flat]    # (N_child,)
    child_ext = p_ext * 0.5
    child_z_ext = p_z_ext * 0.5

    off = _OCTANT_OFFSETS[octant_ids]              # (N_child, 3)
    off[:, 0] *= child_ext
    off[:, 1] *= child_ext
    off[:, 2] *= child_z_ext

    child.centers = parent.centers[parent_ids_flat] + off
    child.extents = child_ext
    child.z_extents = child_z_ext
    # leaf only if occupancy == 0 — children start unset
    child.is_leaf[:] = False

    decoded_octree[level + 1] = child
    node_numlist[level + 1] = N_child


# ======================================================================
#  Vectorised gen_K_parent_seq  (agrees with encoder)
# ======================================================================

def _gen_parent_seq_fast(decoded_octree: list, K: int, node_numlist: list,
                          include_leaves: bool = False):
    """Vectorised per‑node context feature construction.

    When ``include_leaves=False`` (default), leaf nodes are excluded from the output
    sequence, matching the encoder's non-leaf context. When ``include_leaves=True``,
    all nodes are included (used during prediction extension).

    Returns
    -------
    DataStruct : dict with keys "Seq", "Level", "Extents", "Pos"
    """
    total_levels = len(decoded_octree)
    if total_levels == 0:
        return {"Seq": np.empty((0, K), dtype=np.int32),
                "Level": np.empty((0, K, 2), dtype=np.int32),
                "Extents": np.empty((0, K, 2), dtype=np.float32),
                "Pos": np.empty((0, K, 3), dtype=np.float32)}

    # Global index offsets per level (all nodes)
    global_offsets = [0]
    for L in range(total_levels):
        global_offsets.append(global_offsets[-1] + node_numlist[L])

    if not include_leaves:
        is_non_leaf = [~lvl.is_leaf for lvl in decoded_octree]
        n_out = int(sum(nl.sum() for nl in is_non_leaf))

        out_offsets = [0]
        for L in range(total_levels):
            out_offsets.append(out_offsets[-1] + int(is_non_leaf[L].sum()))

        rank_lookup = []
        for L in range(total_levels):
            nl = is_non_leaf[L]
            rank = np.full(node_numlist[L], -1, dtype=np.int32)
            rank[nl] = np.arange(nl.sum(), dtype=np.int32)
            rank_lookup.append(rank)
    else:
        n_out = global_offsets[-1]
        out_offsets = global_offsets

    Seq     = np.zeros((n_out, K), dtype=np.int32)
    Level   = np.zeros((n_out, K, 2), dtype=np.int32)
    Extents = np.zeros((n_out, K, 2), dtype=np.float32)
    Pos     = np.zeros((n_out, K, 3), dtype=np.float64)

    for L in range(total_levels):
        lvl = decoded_octree[L]
        start = out_offsets[L]
        end = out_offsets[L + 1]
        out_idx = np.arange(start, end)

        if not include_leaves:
            nl_mask = is_non_leaf[L]
            Seq   [out_idx, K - 1] = lvl.octants[nl_mask].astype(np.int32)
            Level [out_idx, K - 1, 0] = L + 1
            Level [out_idx, K - 1, 1] = lvl.curind[nl_mask]
            Extents[out_idx, K - 1, 0] = lvl.extents[nl_mask].astype(np.float32)
            Extents[out_idx, K - 1, 1] = lvl.z_extents[nl_mask].astype(np.float32)
            Pos   [out_idx, K - 1, :] = lvl.centers[nl_mask]
        else:
            Seq   [out_idx, K - 1] = lvl.octants.astype(np.int32)
            Level [out_idx, K - 1, 0] = L + 1
            Level [out_idx, K - 1, 1] = lvl.curind
            Extents[out_idx, K - 1, 0] = lvl.extents.astype(np.float32)
            Extents[out_idx, K - 1, 1] = lvl.z_extents.astype(np.float32)
            Pos   [out_idx, K - 1, :] = lvl.centers

        if L == 0:
            for k in range(K - 1):
                Seq   [out_idx, k]     = 256
                Level [out_idx, k, :]  = 0
                Pos   [out_idx, k, :]  = 0
                Extents[out_idx, k, :] = 0
        else:
            parent_lvl = L - 1
            if not include_leaves:
                p_local = lvl.parent_ids[nl_mask]
                p_out = out_offsets[parent_lvl] + rank_lookup[parent_lvl][p_local]
            else:
                p_local = lvl.parent_ids
                p_out = global_offsets[parent_lvl] + p_local
            for k in range(K - 1):
                Seq   [out_idx, k]     = Seq   [p_out, k + 1]
                Level [out_idx, k, :]  = Level [p_out, k + 1, :]
                Pos   [out_idx, k, :]  = Pos   [p_out, k + 1, :]
                Extents[out_idx, k, :] = Extents[p_out, k + 1, :]

    return {
        "Seq":     Seq.astype(np.int32),
        "Level":   Level.astype(np.int32),
        "Extents": Extents.astype(np.float32),
        "Pos":     Pos.astype(np.float32),
    }


# ======================================================================
#  Model helpers
# ======================================================================

# ======================================================================
#  IO
# ======================================================================

def load_octree_file(path: str):
    """
    Read a compressed .octree file (pickle metadata format from encode.py).

    Binary layout:
      magic "OCTR" (4 bytes)
      metadata_len (uint32, 4 bytes)
      metadata pickle bytes (variable)
      num_levels (uint32, 4 bytes)
      directory: (uint32 level_id, uint64 offset, uint64 length) * num_levels
      body: concatenated byte streams

    Returns (metadata dict, level_to_bytes dict).
    """
    import pickle as _pickle

    with open(path, "rb") as f:
        buf = f.read()

    off = 0
    # Magic
    magic = buf[off:off + 4].decode("ascii")
    off += 4
    assert magic == "OCTR", f"Bad magic: {magic}"

    # Metadata (pickle)
    meta_len = struct.unpack_from("<I", buf, off)[0]
    off += 4
    metadata = _pickle.loads(buf[off:off + meta_len])
    off += meta_len

    # Directory
    num_levels = struct.unpack_from("<I", buf, off)[0]
    off += 4

    dir_entries = []
    body_start = off + num_levels * (4 + 8 + 8)  # uint32 + uint64 + uint64 = 20 bytes
    for _ in range(num_levels):
        lvl = struct.unpack_from("<I", buf, off)[0]
        byte_offset = struct.unpack_from("<q", buf, off + 4)[0]
        byte_len = struct.unpack_from("<q", buf, off + 12)[0]
        off += 20
        dir_entries.append((lvl, byte_offset, byte_len))

    # Body
    level_to_bytes = {}
    for lvl, byte_offset, byte_len in dir_entries:
        level_to_bytes[lvl] = buf[body_start + byte_offset:body_start + byte_offset + byte_len]

    return metadata, level_to_bytes


# ======================================================================
#  Reconstruct point cloud
# ======================================================================

def reconstruct_point_cloud(
    decoded_octree: list,   # List[LevelNodeData]
    node_numlist: list,
    metadata: dict,
) -> np.ndarray:
    """Extract leaf‑node centres → inverse cylindrical → Cartesian."""
    qs = metadata.get("qs", [1.0, 1.0, 1.0])
    if isinstance(qs, list):
        qs = np.array(qs, dtype=np.float32)
    recon_params = metadata.get("recon_params", None)
    m = float(metadata.get("m", 1.0))
    n = float(metadata.get("n", 0.015))
    resize = metadata.get("resize", "exp")

    leaf_list = []
    for level in range(len(decoded_octree)):
        lvl = decoded_octree[level]
        # leaf = octant == 0 (no children)
        leaf_pts = lvl.centers[lvl.octants == 0]
        leaf_list.append(leaf_pts)

    leaf_points = np.concatenate(leaf_list, axis=0).astype(np.float32)
    print(f"[Decode] Found {len(leaf_points)} leaf nodes")

    if leaf_points.shape[0] == 0:
        raise RuntimeError("No leaf nodes found in decoded octree")

    out_points = leaf_points * qs
    out_points = cylin2cart(out_points, resize, m, n)
    return out_points


# ======================================================================
#  Level feature extraction  (for model)
# ======================================================================

def _get_level_features_for_decoding(
    decoded_octree: list,
    node_numlist: list,
    level: int,
    K_context: int,
    non_leaf_offsets: list,
):
    """
    Extract normalised features for non-leaf nodes at ``level``.
    Uses ``_gen_parent_seq_fast`` (non-leaf only), matching encoder's ctx_nonleaf.

    Returns (padding_mask, decoded, extents_norm, pos_norm, V_L)
    where V_L is the number of non-leaf nodes at this level.
    """
    datastruct = _gen_parent_seq_fast(decoded_octree, K_context, node_numlist)

    start = non_leaf_offsets[level]
    end   = non_leaf_offsets[level + 1]
    V_L = end - start

    if V_L == 0:
        return None, None, None, None, 0

    feat_seq     = datastruct["Seq"][start:end]       # (V_L, K)
    feat_level   = datastruct["Level"][start:end]     # (V_L, K, 2)
    feat_extents = datastruct["Extents"][start:end]   # (V_L, K, 2)
    feat_pos     = datastruct["Pos"][start:end]       # (V_L, K, 3)

    # data: (V_L, K, 3) — matches encoder: channel 0=level depth, channel 1=curind(octant), channel 2=occupancy
    data_arr = np.concatenate(
        [feat_level.astype(np.int64),
         feat_seq[:, :, np.newaxis].astype(np.int64)], axis=-1)  # (V_L, K, 3)  int64 → torch.long

    # Normalise positions & extents — encoder uses Pos[:, -1, :] (last context slot only)
    pos = feat_pos[:, -1, :].T.copy()             # (3, V_L)
    ext = feat_extents[:, -1, :]                  # (V_L, 2)
    pos_norm, ext_norm = _process_features(pos, ext)

    padding_mask = np.ones(V_L, dtype=np.int32)
    pos_t = torch.from_numpy(pos_norm).unsqueeze(0)                # (1, 3, V_L)
    ext_t = torch.from_numpy(ext_norm).unsqueeze(0)                # (1, V_L, 2)
    dec_t = torch.from_numpy(data_arr).unsqueeze(0)                # (1, V_L, K, 3)
    pm_t = torch.from_numpy(padding_mask).unsqueeze(0).bool()      # (1, V_L)

    return pm_t, dec_t, ext_t, pos_t, V_L


# ======================================================================
#  Autoregressive decode
# ======================================================================

@torch.no_grad()
def decode_octree_autoregressive(
    model: SelfOrderProbModel,
    metadata: dict,
    level_to_bytes: dict,
    device: torch.device,
    context_size: int = 8192,
    num_predict_layer: int = 0,
    threshold: float = 0.9,
):
    """
    Main autoregressive loop.
    1) Build base levels L0…Lmax from metadata.
    2) Bootstrap non‑encoded levels (depth < min_encode_level): set to 255, expand.
    3) For each encoded level: identify leaf nodes (pruned / max_depth) → occupancy=0;
       for non-leaf nodes → build features → model → entropy decode → set octants → expand.
    4) Optional prediction extension.
    """
    lidar_level = int(metadata["lidar_level"])
    num_encode_level = int(metadata["num_encode_level"])
    min_encode_level = int(metadata.get("min_encode_level", 2))
    K_context = int(metadata.get("K_context", 4))
    max_r = float(metadata.get("max_r", 0.0))
    total_levels = lidar_level + 1

    node_numlist = [0] * total_levels

    # --- 1) base octree skeleton ---------------------------------------
    decoded_octree = _build_base_levels(metadata, node_numlist)
    # L0 (root) is already marked non‑leaf or leaf based on root_occupancy
    decoded_octree[0].is_leaf[0] = (decoded_octree[0].octants[0] == 0)
    print(f"[Decode] L0 root oc={decoded_octree[0].octants[0]} leaf={decoded_octree[0].is_leaf[0]}")

    # --- 2) Bootstrap non‑encoded levels ---------------------------------
    _expand_children(decoded_octree, 0, node_numlist)

    for lvl in range(1, min_encode_level - 1):
        if lvl >= total_levels:
            break
        if node_numlist[lvl] > 0:
            decoded_octree[lvl].octants[:] = 255
            decoded_octree[lvl].is_leaf[:] = False
        _expand_children(decoded_octree, lvl, node_numlist)

    print(f"[Decode] Bootstrapped levels 0..{min_encode_level - 2}: "
          f"{node_numlist[:min_encode_level - 1]} nodes")

    # --- 3) Decode loop ------------------------------------------------
    # Encoder encodes depths [min_encode_level, num_encode_level) = depths 2..max_depth-1.
    # Decoder level indices: lvl = depth - 1.
    for lvl in range(min_encode_level - 1, num_encode_level - 1):
        N_L = node_numlist[lvl]
        depth = lvl + 1  # 1-indexed depth
        print(f"\n[Decode] Level {lvl} (depth {depth}): {N_L} nodes")

        if N_L == 0:
            continue

        # ── Identify leaf nodes (pruned or at max_depth) ──
        centers = decoded_octree[lvl].centers
        at_max_depth = np.full(N_L, depth >= lidar_level, dtype=bool)
        pruned = _is_node_pruned(centers[:, 0], depth, max_r, lidar_level)
        is_leaf = at_max_depth | pruned
        non_leaf_mask = ~is_leaf
        V_L = int(non_leaf_mask.sum())

        # Mark leaf nodes
        decoded_octree[lvl].octants[is_leaf] = 0
        decoded_octree[lvl].is_leaf[is_leaf] = True

        print(f"[Decode]   leaf={int(is_leaf.sum())} (pruned={int(pruned.sum())} max_depth={int(at_max_depth.sum())}) "
              f"non_leaf={V_L}")

        if V_L == 0:
            continue

        # ── Parse even/odd byte streams ──
        if depth not in level_to_bytes:
            print(f"[Decode]   No bytes for depth {depth}, skipping")
            continue
        level_bytes = level_to_bytes[depth]

        # ── Compute non-leaf offsets for _gen_parent_seq_fast ──
        non_leaf_offsets = [0]
        for Lv in range(total_levels):
            nl_count = int((~decoded_octree[Lv].is_leaf).sum()) if node_numlist[Lv] > 0 else 0
            non_leaf_offsets.append(non_leaf_offsets[-1] + nl_count)

        # ── Build features for non-leaf nodes at this level ──
        pm_L, dec_L, ext_L, pos_L, V_L_check = _get_level_features_for_decoding(
            decoded_octree, node_numlist, lvl, K_context, non_leaf_offsets)
        assert pm_L is not None and dec_L is not None and ext_L is not None and pos_L is not None
        assert V_L == V_L_check, f"V_L mismatch: {V_L} vs {V_L_check}"

        # ── Read 4-byte even-length prefix ──
        even_len = struct.unpack("<I", level_bytes[:4])[0]
        even_bytes = level_bytes[4:4 + even_len]
        odd_bytes = level_bytes[4 + even_len:]
        print(f"[Decode]   even_bytes={len(even_bytes)}, odd_bytes={len(odd_bytes)}")

        # ── Forward pass: even/odd split in SORTED order ──
        # prob_sorted[:, ::2] = prob1 (even branch, no pre_occ needed)
        all_even_sorted_probs = []
        all_odd_sorted_probs = []
        all_even_valid_masks = []
        all_odd_valid_masks = []
        all_restore_indices = []
        all_saved = []  # (fa1, fa2, codes, ri, pm_ch, V_chunk)

        for chunk_start in range(0, V_L, context_size):
            chunk_end = min(chunk_start + context_size, V_L)
            V_chunk = chunk_end - chunk_start

            pm_ch = pm_L[:, chunk_start:chunk_end]
            dec_ch = dec_L[:, chunk_start:chunk_end]
            ext_ch = ext_L[:, chunk_start:chunk_end]
            pos_ch = pos_L[:, :, chunk_start:chunk_end]

            if V_chunk < context_size:
                pad = context_size - V_chunk
                pm_ch = F.pad(pm_ch, (0, pad), value=False)
                dec_ch = F.pad(dec_ch, (0, 0, 0, 0, 0, pad), value=0)
                ext_ch = F.pad(ext_ch, (0, 0, 0, pad), value=0)
                pos_ch = F.pad(pos_ch, (0, pad), value=0)

            pm_ch = pm_ch.to(device)
            dec_ch = dec_ch.to(device)
            ext_ch = ext_ch.to(device)
            pos_ch = pos_ch.to(device)

            feat_a1, feat_a2, pre_occ_embed, pre_occ_raw, codes, restore_indices = \
                model.forward_features(dec_ch, ext_ch, pos_ch, pm_ch)

            prob_sorted = model.predicthead(feat_a1, feat_a2, pre_occ_embed, codes)
            probs_sorted = F.softmax(prob_sorted, dim=-1)  # (1, CS, 256) in SORTED order

            # Even/odd in SORTED order
            even_sorted = probs_sorted[:, ::2, :]
            odd_sorted = probs_sorted[:, 1::2, :]
            even_valid_mask = pm_ch[:, ::2]   # (1, CS/2)
            odd_valid_mask = pm_ch[:, 1::2]

            valid_even = even_sorted[0][even_valid_mask[0]]
            valid_odd = odd_sorted[0][odd_valid_mask[0]]

            all_even_sorted_probs.append(valid_even)
            all_odd_sorted_probs.append(valid_odd)
            all_even_valid_masks.append(even_valid_mask)
            all_odd_valid_masks.append(odd_valid_mask)
            all_restore_indices.append(restore_indices)
            all_saved.append((feat_a1, feat_a2, pre_occ_raw, codes, restore_indices, pm_ch, V_chunk))

        # Decode evens (prob1, no pre_occ dependency)
        # Build CDF on CPU — matches encoder behavior (encoder uses .cpu() on probs)
        combined_even_cpu = torch.cat([p.cpu() for p in all_even_sorted_probs], dim=0)
        even_cdf = torch.cumsum(combined_even_cpu, dim=-1)
        even_cdf[:, -1] = 1.0
        even_cdf_wz = torch.cat([torch.zeros(combined_even_cpu.shape[0], 1), even_cdf.clamp(max=1.0)], dim=-1)
        even_sorted_symbols = torchac.decode_float_cdf(even_cdf_wz, even_bytes).to(torch.long)

        # ── Stage 2: update pre_occ with decoded evens → re-run PredictHead for odds ──
        all_odd_sorted_probs_corrected = []
        for chunk_idx, (chunk_start, (fa1, fa2, pre_occ_raw_zero, codes, ri, pm_ch, V_chunk)) in \
                enumerate(zip(range(0, V_L, context_size), all_saved)):
            chunk_end = min(chunk_start + context_size, V_L)

            # Build chunk_occs in original order with decoded evens (odds still 0)
            chunk_even_start = sum(int(mask.sum()) for mask in all_even_valid_masks[:chunk_idx])
            chunk_even_end = chunk_even_start + int(all_even_valid_masks[chunk_idx].sum())
            chunk_odd_start = sum(int(mask.sum()) for mask in all_odd_valid_masks[:chunk_idx])
            chunk_odd_end = chunk_odd_start + int(all_odd_valid_masks[chunk_idx].sum())

            # Interleave in sorted order (valid positions only)
            occs_chunk_sorted = torch.zeros(context_size, dtype=torch.long)
            ev_mask = all_even_valid_masks[chunk_idx][0].cpu()
            od_mask = all_odd_valid_masks[chunk_idx][0].cpu()
            even_pos = torch.where(ev_mask)[0] * 2
            odd_pos = torch.where(od_mask)[0] * 2 + 1
            occs_chunk_sorted[even_pos] = even_sorted_symbols[chunk_even_start:chunk_even_end]
            occs_chunk_sorted[odd_pos] = 0  # odds not decoded yet

            # Extract new pre_occ from sorted order: even positions
            new_pre_occ = occs_chunk_sorted[::2].to(torch.int32)

            new_pre_occ_embed = model.geo_feat_generator.embed_occ(new_pre_occ.unsqueeze(0).to(device))

            # Re-run PredictHead with correct pre_occ
            prob_sorted2 = model.predicthead(fa1, fa2, new_pre_occ_embed, codes)
            probs_sorted2 = F.softmax(prob_sorted2, dim=-1)
            odd_sorted2 = probs_sorted2[:, 1::2, :]
            valid_odd2 = odd_sorted2[0][all_odd_valid_masks[chunk_idx][0]]
            all_odd_sorted_probs_corrected.append(valid_odd2)

        # Decode odds with corrected prob2 (CDF on CPU, matching encoder)
        combined_odd_cpu = torch.cat([p.cpu() for p in all_odd_sorted_probs_corrected], dim=0)
        odd_cdf = torch.cumsum(combined_odd_cpu, dim=-1)
        odd_cdf[:, -1] = 1.0
        odd_cdf_wz = torch.cat([torch.zeros(combined_odd_cpu.shape[0], 1), odd_cdf.clamp(max=1.0)], dim=-1)
        odd_sorted_symbols = torchac.decode_float_cdf(odd_cdf_wz, odd_bytes).to(torch.long)

        print(f"[Decode]   decoded {len(even_sorted_symbols)} even + {len(odd_sorted_symbols)} odd symbols "
              f"(total {len(even_sorted_symbols) + len(odd_sorted_symbols)}, expected {V_L})")

        # ── Restore from sorted order to original order ──
        occs_chunks = []
        even_off, odd_off = 0, 0
        for chunk_idx, ri in enumerate(all_restore_indices):
            V_chunk = min(context_size, V_L - chunk_idx * context_size)
            ev_mask = all_even_valid_masks[chunk_idx][0].cpu()
            od_mask = all_odd_valid_masks[chunk_idx][0].cpu()
            n_even_valid = int(ev_mask.sum())
            n_odd_valid = int(od_mask.sum())

            # Interleave in sorted order, filling only valid positions
            occs_sorted = torch.zeros(context_size, dtype=torch.long)
            even_positions = torch.where(ev_mask)[0] * 2   # even positions in sorted order
            odd_positions = torch.where(od_mask)[0] * 2 + 1  # odd positions in sorted order
            occs_sorted[even_positions] = even_sorted_symbols[even_off:even_off + n_even_valid]
            occs_sorted[odd_positions] = odd_sorted_symbols[odd_off:odd_off + n_odd_valid]
            even_off += n_even_valid
            odd_off += n_odd_valid

            # Restore to original order
            occs_original = occs_sorted[ri[0].cpu()].to(torch.long)
            occs_chunks.append(occs_original[:V_chunk])

        occs = torch.cat(occs_chunks, dim=0).to(device)

        occs_np = occs.cpu().numpy().astype(np.uint8)

        # ── Update octants for non-leaf nodes ──
        decoded_octree[lvl].octants[non_leaf_mask] = occs_np
        decoded_octree[lvl].is_leaf[non_leaf_mask] = (occs_np == 0)

        # ── Expand children from non-leaf nodes ──
        _expand_children(decoded_octree, lvl, node_numlist)

    # --- 4) Prediction extension ---------------------------------------
    if num_predict_layer > 0:
        print(f"[Decode] Prediction extension: {num_predict_layer} layer(s), "
              f"threshold={threshold}")
        for pred_iter in range(num_predict_layer):
            print(f"[Decode]   Predict layer {pred_iter + 1}/{num_predict_layer} ...")
            _predict_leaf_nodes_expansion(
                model, decoded_octree, node_numlist,
                K_context, context_size, device, threshold,
            )
            total_nodes = sum(node_numlist)
            print(f"[Decode]     -> {total_nodes} total nodes")

    return decoded_octree, node_numlist


# ======================================================================
#  Prediction extension  (one‑layer expansion)
# ======================================================================

@torch.no_grad()
def _predict_leaf_nodes_expansion(
    model: SelfOrderProbModel,
    decoded_octree: list,
    node_numlist: list,
    K_context: int,
    context_size: int,
    device: torch.device,
    threshold: float,
):
    """
    Identify leaf nodes, run model to predict their occupancy,
    greedily expand those above ``threshold``.
    """
    total_levels = len(decoded_octree)
    # Collect leaf nodes across all levels
    leaf_global = []   # (global_idx, level)

    cum_off = 0
    for L in range(total_levels):
        n_nodes = node_numlist[L]
        leaf_mask = decoded_octree[L].is_leaf | (decoded_octree[L].octants == 0)
        leaf_idx = np.where(leaf_mask)[0]
        leaf_global.append((L, leaf_idx + cum_off))
        cum_off += n_nodes

    if not leaf_global:
        return

    # Build full sequence (all nodes, including leaves)
    datastruct = _gen_parent_seq_fast(decoded_octree, K_context,
                                       node_numlist, include_leaves=True)

    # Process leaves in batches
    all_leaf_levels = []
    all_leaf_gids = []
    for L, gids in leaf_global:
        all_leaf_levels.append(np.full(len(gids), L, dtype=np.int32))
        all_leaf_gids.append(gids)

    if len(all_leaf_gids) == 0:
        return

    all_leaf_levels = np.concatenate(all_leaf_levels)
    all_leaf_gids   = np.concatenate(all_leaf_gids)

    # Sort by descending level: process deep leaves first so _expand_children
    # (which modifies node_numlist at lvl+1) doesn't invalidate global indices
    # computed from the original node_numlist.
    sort_idx = np.argsort(-all_leaf_levels)  # descending
    all_leaf_levels = all_leaf_levels[sort_idx]
    all_leaf_gids = all_leaf_gids[sort_idx]

    N_leaves = len(all_leaf_gids)
    print(f"[Decode]     Predicting occupancy for {N_leaves} leaves")

    for batch_start in range(0, N_leaves, context_size):
        batch_end = min(batch_start + context_size, N_leaves)
        batch_gids = all_leaf_gids[batch_start:batch_end]
        batch_lvls = all_leaf_levels[batch_start:batch_end]
        B = len(batch_gids)

        feat_seq     = datastruct["Seq"][batch_gids]       # (B, K)
        feat_level   = datastruct["Level"][batch_gids]     # (B, K, 2)
        feat_extents = datastruct["Extents"][batch_gids]   # (B, K, 2)
        feat_pos     = datastruct["Pos"][batch_gids]       # (B, K, 3)

        # data: concat(Level, Seq[:,:,np.newaxis]) → matches encoder
        data_arr = np.concatenate(
            [feat_level.astype(np.int64),
             feat_seq[:, :, np.newaxis].astype(np.int64)], axis=-1)  # (B, K, 3)  int64 → torch.long

        pos = feat_pos[:, -1, :].T.copy()              # (3, B)
        ext = feat_extents[:, -1, :]                   # (B, 2)
        pos_norm, ext_norm = _process_features(pos, ext)

        pm_t = torch.ones(1, B, dtype=torch.bool, device=device)
        dec_t = torch.from_numpy(data_arr).unsqueeze(0).to(device)
        ext_t = torch.from_numpy(ext_norm).unsqueeze(0).to(device)
        pos_t = torch.from_numpy(pos_norm).unsqueeze(0).to(device)

        # Pad
        if B < context_size:
            pad = context_size - B
            pm_t = F.pad(pm_t, (0, pad), value=False)
            dec_t = F.pad(dec_t, (0, 0, 0, 0, 0, pad), value=0)
            ext_t = F.pad(ext_t, (0, 0, 0, pad), value=0)
            pos_t = F.pad(pos_t, (0, pad), value=0)

        feat_a1, feat_a2, pre_occ_embed, pre_occ_raw, codes, restore_indices = \
            model.forward_features(dec_t, ext_t, pos_t, pm_t)

        prob_sorted = model.predicthead(feat_a1, feat_a2, pre_occ_embed, codes)  # (1, CS, 256)
        prob = sort_use_indices_3dim(prob_sorted, restore_indices)               # (1, CS, 256) 原始顺序
        probs = F.softmax(prob, dim=-1)                                          # (1, CS, 256)

        # Take even‑position probabilities for the B valid leaf nodes
        pred_oc = torch.argmax(probs[0, :B, :], dim=-1)  # (B,)
        pred_conf = probs[0, :B, :].max(dim=-1)[0]        # (B,)

        # Only expand if confidence >= threshold
        expand_mask = (pred_conf >= threshold).cpu().numpy()

        for i in range(B):
            if expand_mask[i]:
                lvl = int(batch_lvls[i])
                gid = int(batch_gids[i])
                # local index at level lvl
                local_idx = gid - sum(node_numlist[:lvl])

                occ_val = int(pred_oc[i].item())
                decoded_octree[lvl].octants[local_idx] = occ_val
                decoded_octree[lvl].is_leaf[local_idx] = (occ_val == 0)

                # If has children, we need to expand
                if occ_val != 0 and lvl + 1 < total_levels:
                    _expand_children(decoded_octree, lvl, node_numlist)


# ======================================================================
#  Single‑file decode
# ======================================================================

def decode_single_file(
    octree_path: str,
    output_ply: str,
    model: SelfOrderProbModel,
    device: torch.device,
    context_size: int = 8192,
    num_predict_layer: int = 0,
    threshold: float = 0.9,
    timing: dict | None = None,
) -> int:
    print(f"\n[Decode] {Path(octree_path).name}")
    t_start = time.time()

    t0 = time.time()
    metadata, level_to_bytes = load_octree_file(octree_path)
    t_load = time.time() - t0

    t0 = time.time()
    decoded_octree, node_numlist = decode_octree_autoregressive(
        model, metadata, level_to_bytes, device, context_size,
        num_predict_layer=num_predict_layer,
        threshold=threshold,
    )
    t_decode = time.time() - t0

    t0 = time.time()
    points = reconstruct_point_cloud(decoded_octree, node_numlist, metadata)
    t_reconstruct = time.time() - t0

    t0 = time.time()
    save_point_cloud(points, output_ply)
    t_save = time.time() - t0

    print(f"[Decode]   → {output_ply}  ({points.shape[0]:,d} points)")

    if timing is not None:
        timing["load_file"] += t_load
        timing["decode"] += t_decode
        timing["reconstruct"] += t_reconstruct
        timing["save_ply"] += t_save
        timing["total"] += time.time() - t_start

    return points.shape[0]


# ======================================================================
#  main
# ======================================================================

def main():

    parser = argparse.ArgumentParser(description="Octree Entropy Decoder (optimised)")
    parser.add_argument("--input_dir", default="./encoded/",
                        help="Directory with .octree files")
    parser.add_argument("--output_dir", default="decoded/",
                        help="Output directory for .ply files")
    parser.add_argument("--ckpt", default="./weights/kitti/model.ckpt")
    parser.add_argument("--config", default="configs/model/self.yaml")
    parser.add_argument("--context_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_predict_layer", type=int, default=1,
                        help="Number of additional layers to predict beyond encoded levels")
    parser.add_argument("--threshold", type=float, default=0.0001,
                        help="Confidence threshold [0,1] for prediction expansion")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Decode] device = {device}")

    # Load model
    print("[Decode] Loading model …")
    model = load_model(args.ckpt, args.config, device)

    # Scan input
    inp = Path(args.input_dir)
    octree_files = sorted(inp.glob("*.octree"))
    if not octree_files:
        raise FileNotFoundError(f"No .octree files in {inp}")
    print(f"[Decode] {len(octree_files)} .octree file(s) found")

    # Output
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_points = 0
    timing = {"load_file": 0.0, "decode": 0.0, "reconstruct": 0.0,
              "save_ply": 0.0, "total": 0.0}
    n_files = len(octree_files)

    for oct_path in octree_files:
        ply_out = out / f"{oct_path.stem}.ply"
        n_pts = decode_single_file(
            str(oct_path), str(ply_out), model, device,
            context_size=args.context_size,
            num_predict_layer=args.num_predict_layer,
            threshold=args.threshold,
            timing=timing,
        )
        total_points += n_pts

    print(f"\n[Decode] All done.  Total reconstructed points: {total_points:,d}")
    print(f"[Decode] Timing (avg per file, {n_files} files):")
    print(f"  Load file    : {timing['load_file']/n_files:.2f} s")
    print(f"  Decode       : {timing['decode']/n_files:.2f} s")
    print(f"  Reconstruct  : {timing['reconstruct']/n_files:.2f} s")
    print(f"  Save PLY     : {timing['save_ply']/n_files:.2f} s")
    print(f"  Total        : {timing['total']/n_files:.2f} s")


if __name__ == "__main__":
    main()