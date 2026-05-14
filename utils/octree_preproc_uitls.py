import math
import os
import sys

import numpy as np

from utils.octree_context import OctreeContext
from utils.c_octree_utils.octree_utils import build_octree_py, build_K_context_py, octree2points_py
from src.constants import KITTI_OCT_CENTER



def cart2cylin(points, resize='exp', m=1.0, n=0.015):
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError("")
    
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]

    rho = np.sqrt(x**2 + y**2)                         # (..., N)
    phi = np.arctan2(y, x + 1e-9)
    phi[phi < 0] += 2 * math.pi                       
    # theta = np.arccos(z / np.where(rho == 0, 1.0, rho))
    theta = np.arccos(np.clip(z / np.where(rho == 0, 1.0, rho), -1.0, 1.0))

    if resize is not None:
        if resize == 'exp':
            rho_t = m * (1 - np.exp(-n * rho))
        else:
            raise ValueError(f"{resize!r}")

        orig_max   = rho.max()                           # scalar
        scaled_max = rho_t.max()                         # scalar

        rho = rho_t * (orig_max / (scaled_max + 1e-9))

    return np.stack([rho, phi, theta], axis=1)      # (N,3)



def cylin2cart(points, resize:str='exp', m:float=1.0, n:float=0.015):
    pts = np.asarray(points)
    if pts.ndim not in (2, 3) or pts.shape[-1] != 3:
        raise ValueError("")

    rho_norm = pts[..., 0]
    phi       = pts[..., 1]
    theta     = pts[..., 2]

    if resize is not None:
        eps = 1e-9
        if pts.ndim == 3:
            # (B, N)
            orig_max   = rho_norm.max(axis=1, keepdims=True)    # (B,1)
        else:
            orig_max   = rho_norm.max()                         # scalar

        def calc_scaled_max(rho_m):
            if resize == 'exp':
                return m * (1 - np.exp(-n * rho_m))
            else:
                raise ValueError(f"{resize!r}")

        scaled_max = calc_scaled_max(orig_max)

        rho_t = rho_norm * (scaled_max / (orig_max + eps))

        if resize == 'exp':
            rho = -np.log(1 - rho_t / m + eps) / n
        else:
            raise ValueError(f"{resize!r}")
    else:
        rho = rho_norm

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = rho * np.cos(theta)

    if pts.ndim == 2:
        return np.stack([x, y, z], axis=1)     # (N,3)
    else:
        return np.stack([x, y, z], axis=2)     # (B,N,3)

def new_glsproc_pc(
    pc: np.ndarray,
    out_dir: str,
    out_name: str,
    oct_center: np.ndarray = KITTI_OCT_CENTER,
    lidar_level: int = 12,
    quant_size: float = 50,
    cylin: bool = True,
    resize: str = 'exp',
    m: float = 1,
    n: float = 0.004,
    z_rate: float = 1,
    context_size: int = 4,
    include_leaves: bool = False,
) -> dict:
    """Build octree and extract K-parent context using the new C++ pipeline.

    Compared to the old ``glsproc_pc``, this function:

    - Uses ``build_octree_py`` (flat BFS, numpy input) instead of
      ``fastutils.GenOctree`` (per-level, list-of-lists input).
    - Uses ``build_K_context_py`` (C++ memcpy propagation) instead of
      the Python-loop ``gen_K_parent_seq``.
    - Returns an :class:`OctreeContext` dataclass with named fields
      instead of a raw (N, K, 8) concatenated array.
    - Saves named arrays to ``.npz`` for self-documenting storage.
    - Supports leaf / non-leaf split via *include_leaves*.

    Parameters
    ----------
    pc : (N, 3) ndarray
        Raw point cloud in Cartesian coordinates.
    out_dir : str
        Output directory (created if missing).
    out_name : str
        Base name for the output file.
    oct_center : (3,) ndarray
        Root node center.
    lidar_level : int
        Maximum octree depth.
    quant_size : float
        XY-plane half-side length of the root node.  -1 means use max(rho).
    cylin : bool
        Whether to use cylindrical coordinate transform.
    resize : str
        Rho rescaling mode (``'exp'``, etc.).
    m, n : float
        Rescaling parameters for ``cart2cylin``.
    z_rate : float
        Z-extent multiplier: ``z_extent = quant_size * z_rate``.
    context_size : int
        Number of ancestor slots in the K-parent context window.
    include_leaves : bool
        If True, ``IsNonLeaf`` mask is included in the context.

    Returns
    -------
    dict
        ``context``      — :class:`OctreeContext` dataclass
        ``out_file``     — saved file path (without ``.npz`` extension)
        ``out_points``   — reconstructed point cloud (Cartesian)
        ``original_pc``  — processed input copy
        ``node_num``     — total number of octree nodes
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    points = pc.copy()
    qs = 1

    if cylin:
        points = cart2cylin(points, resize, m, n)
        bin_num = np.round(points[:, 0].max()) + 1
        qs = np.array(
            [1, 2 * math.pi / (bin_num - 1), math.pi / (bin_num - 1)],
            dtype=np.float32,
        )
    else:
        raise NotImplementedError("")

    P_quant = points / qs

    extent = quant_size
    if extent == -1:
        extent = np.max(points[:, 0])

    # ── new C++ octree builder ──────────────────────────────
    tree = build_octree_py(
        points=P_quant,
        max_depth=lidar_level,
        center=oct_center,
        extent=extent,
        z_extent=extent * z_rate,
        cylin=cylin,
    )

    # ── new C++ K-parent context extractor ──────────────────
    ctx_dict = build_K_context_py(tree, K=context_size, include_leaves=include_leaves)

    # ── wrap in structured dataclass ────────────────────────
    context = OctreeContext.from_build_K_context(ctx_dict, include_leaves=include_leaves)

    # ── save named arrays ───────────────────────────────────
    out_file = os.path.join(out_dir, out_name + "_" + str(context.num_nodes))
    context.save(out_file + ".npz")

    # ── reconstruct point cloud ─────────────────────────────
    out_points = octree2points_py(tree)
    if cylin:
        out_points = out_points * qs
        out_points = cylin2cart(out_points, resize, m, n)

    return {
        "context": context,
        "out_file": out_file,
        "out_points": out_points,
        "original_pc": pc,
        "node_num": context.num_nodes,
    }