#!/usr/bin/env python3
"""
src/octree_builder.py
"""

import math

import numpy as np
import torch

from utils.octree_preproc_uitls import cart2cylin
from utils.c_octree_utils.octree_utils import build_octree_py, build_K_context_py, get_node_numlist
from utils.octree_context import OctreeContext
from src.constants import KITTI_OCT_CENTER


def build_octree_features(
    points_xyz: np.ndarray,
    lidar_level: int,
    quant_size: float,
    cylin: bool = True,
    resize: str = "exp",
    m: float = 1.0,
    n: float = 0.015,
    z_rate: float = 1.0,
    oct_center: np.ndarray = KITTI_OCT_CENTER,
    K_context: int = 4,
):
    points = points_xyz.copy()

    points = cart2cylin(points, resize=resize, m=m, n=n)
    recon_params = {'resize': resize, 'm': m, 'n': n}
    bin_num = int(np.round(points[:, 0].max())) + 1
    qs = np.array([1.0,
                   2 * math.pi / max(bin_num - 1, 1),
                   math.pi / max(bin_num - 1, 1)], dtype=np.float32)

    P_quant = (points / qs).astype(np.float32)
    extent = quant_size
    if extent == -1:
        extent = np.max(points[:, 0])

    max_r = float(max(P_quant[:, 0].max(), P_quant[:, 1].max()))

    # ── new C++ octree builder ──────────────────────────────
    tree = build_octree_py(
        points=P_quant,
        max_depth=lidar_level,
        center=oct_center,
        extent=extent,
        z_extent=extent * z_rate,
        cylin=True,
    )

    # ── node counts per level (all nodes) ───────────────────
    node_numlist = get_node_numlist(tree)

    # ── new C++ K-parent context (include all nodes) ────────
    ctx_dict = build_K_context_py(tree, K=K_context, include_leaves=True)
    ctx = OctreeContext.from_build_K_context(ctx_dict, include_leaves=True)

    # ── split non-leaf / leaf ───────────────────────────────
    ctx_nonleaf = ctx.filter_non_leaf()
    ctx_leaf = ctx.filter_leaf()

    # ── convert to torch tensors ────────────────────────────
    def _ctx_to_tensors(ctx_sub):
        """(N, K) fields → (1, N, K, 3) data / (1, N, 2) ext / (1, 3, N) pos / (1, N) label."""
        data = np.stack(
            [ctx_sub.level, ctx_sub.cur_child_idx, ctx_sub.occupancy], axis=-1
        ).astype(np.int64)
        data = torch.from_numpy(data).unsqueeze(0)
        extents = torch.from_numpy(ctx_sub.last_extents).unsqueeze(0)
        pos = torch.from_numpy(ctx_sub.last_pos.T.copy()).unsqueeze(0).float()
        seq_labels = torch.from_numpy(ctx_sub.last_occupancy.copy()).unsqueeze(0).long()
        return data, extents, pos, seq_labels

    data, extents, pos, seq_labels = _ctx_to_tensors(ctx_nonleaf)
    leaf_data, leaf_extents, leaf_pos, leaf_seq_labels = _ctx_to_tensors(ctx_leaf)

    features = dict(data=data, extents=extents, pos=pos,
                    seq_labels=seq_labels, node_numlist=node_numlist)
    leaf_features = dict(data=leaf_data, extents=leaf_extents, pos=leaf_pos,
                         seq_labels=leaf_seq_labels, node_numlist=node_numlist)
    return features, leaf_features, qs, recon_params, tree, None, node_numlist, max_r



