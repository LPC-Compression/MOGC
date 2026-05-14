#!/usr/bin/env python3
"""
Octree Entropy Encoder for Point Cloud Compression

Pipeline:
    1. Point Cloud Reading  ->  read .ply files from input directory
    2. Coordinate Transform  ->  Cartesian to Cylindrical (cart2cylin)
    3. Octree Building       ->  fastutils.GenOctree
    4. Node Feature Processing -> gen_K_parent_seq
    5. Model Prediction      ->  SelfOrderProbModel inference + PredictHead two‑branch
    6. Entropy Coding        ->  torchac arithmetic coding (even/odd split)

Output: one .octree file per input .ply point cloud.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data.io import read_point_cloud

from src.model_loader import load_model
from src.octree_builder import build_octree_features
from src.chunk_builder import build_level_chunks
from src.entropy_coding import encode_chunks_entropy, save_encoded_file
from src.constants import get_constants, get_encode_params


def print_gpu_memory(device):
    if not device.type == 'cuda':
        return
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserve = torch.cuda.memory_reserved(device) / 1024**3
    peak_a = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_r = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"[GPU Memory] allocated: {alloc:.2f} GB, reserved: {reserve:.2f} GB | "
          f"peak allocated: {peak_a:.2f} GB, peak reserved: {peak_r:.2f} GB")




# ═══════════════════════════════════════════════════════════════
#  Encode a single point cloud file
# ═══════════════════════════════════════════════════════════════

def encode_single_file(
    ply_path: str,
    output_path: str,
    model,
    device: torch.device,
    lidar_level: int = 15,
    quant_size: float = 50.0,
    m: float = 1.0,
    n: float = 0.015,
    z_rate: float = 1.0,
    context_size: int = 8192,
    cylin: bool = True,
    K_context: int = 4,
    oct_center: np.ndarray = np.array([50, 50, 45], dtype=np.float32),
    timing: dict | None = None,
):
    """
    Encode a single PLY point cloud file → one .octree file.

    Returns
    -------
    bpp : float   bits per point
    """
    t_start = time.time()

    # 1. Read point cloud
    points = read_point_cloud(ply_path)
    points = points.astype(np.float32)
    print(f"  Points: {points.shape[0]:,d}")

    # 2. Build octree + features
    t0 = time.time()
    features, _leaf_features, qs, _recon_params, _tree, _root, node_numlist, max_r = \
        build_octree_features(
            points_xyz=points,
            lidar_level=lidar_level,
            quant_size=quant_size,
            cylin=cylin,
            resize="exp",
            m=m, n=n, z_rate=z_rate,
            oct_center=oct_center,
            K_context=K_context,
        )
    t_octree = time.time() - t0
    print(f"  Octree: {len(node_numlist)} levels, {sum(node_numlist):,d} nodes, max_r={max_r:.4f}")

    # 3. Split by level & chunk  (narrow dict value types for the type checker)
    t0 = time.time()
    f_data = features["data"]
    f_extents = features["extents"]
    f_pos = features["pos"]
    f_labels = features["seq_labels"]
    assert isinstance(f_data, torch.Tensor)
    assert isinstance(f_extents, torch.Tensor)
    assert isinstance(f_pos, torch.Tensor)
    assert isinstance(f_labels, torch.Tensor)
    chunk_result = build_level_chunks(
        f_data, f_extents, f_pos, f_labels,
        context_size, node_numlist=node_numlist,
    )
    chunk_data_list, chunk_ext_list, chunk_pos_list, chunk_lbl_list, \
        st_list, ed_list, level_list, pad_mask_list = chunk_result
    t_chunks = time.time() - t0
    print(f"  Chunks: {len(chunk_data_list)} across {len(set(level_list))} levels")

    # 4. Encode
    t0 = time.time()
    level_to_bytes, total_estimated_bits = encode_chunks_entropy(
        model, chunk_data_list, chunk_ext_list, chunk_pos_list,
        chunk_lbl_list, st_list, ed_list, level_list, pad_mask_list,
        device,
    )
    t_encode = time.time() - t0

    root_occupancy = int(f_labels[0, 0].item())

    # 5. Save
    t0 = time.time()
    save_encoded_file(
        output_path, level_to_bytes, node_numlist, max_r, oct_center,
        qs, _recon_params, m, n, z_rate,
        lidar_level, quant_size, cylin, "exp", K_context,
        num_points=points.shape[0],
        root_occupancy=root_occupancy,
        estimated_bits=total_estimated_bits,
    )
    t_save = time.time() - t0

    # BPP
    total_bits = os.path.getsize(output_path) * 8
    bpp = total_bits / max(points.shape[0], 1)
    print(f"  Bits: {total_bits:,d}  BPP: {bpp:.4f}")

    if timing is not None:
        timing["build_octree"] += t_octree
        timing["build_chunks"] += t_chunks
        timing["entropy_encode"] += t_encode
        timing["save_file"] += t_save
        timing["total"] += time.time() - t_start

    return bpp, points.shape[0]


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Octree Entropy Encoder")
    parser.add_argument("--input_dir", help="Directory with .ply point clouds",
                        default="./test_PCs/SemanticKITTI/")
    parser.add_argument("--output_dir", help="Output directory for .octree files",
                        default="./encoded_14/")
    parser.add_argument("--ckpt", default="./weights/kitti/model.ckpt")
    parser.add_argument("--config", default="configs/model/self.yaml")
    parser.add_argument("--lidar_level", type=int, default=14)
    parser.add_argument("--context_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datatype", default="kitti", choices=["kitti", "ford"], help="Dataset type")
    parser.add_argument("--K_context", type=int, default=4, help="K-context window size")
    args = parser.parse_args()

    if args.datatype == "kitti":
        dataset_name = "SemanticKITTI"
        dataset_key = "SemanticKITTIDataset"
    elif args.datatype == "ford":
        dataset_name = "Ford"
        dataset_key = "FordDataset"
    else:
        raise ValueError(f"Unsupported datatype: {args.datatype}")

    _pos_min, _pos_max, _extent_min, _extent_max, _eps, oct_center = get_constants(dataset_name)
    enc_params = get_encode_params(dataset_key, lidar_level=args.lidar_level)
    quant_size = enc_params["quant_size"]
    m = enc_params["m"]
    n = enc_params["n"]
    z_rate = enc_params["z_rate"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Encode] device = {device}")

    if device.type == 'cuda':
        torch.tensor([0.], device=device)
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Load model (once)
    print("[Encode] Loading model …")
    model = load_model(args.ckpt, args.config, device)

    # 2. Scan input directory for .ply files
    inp = Path(args.input_dir)
    ply_files = sorted(inp.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No .ply files in {inp}")
    print(f"[Encode] {len(ply_files)} .ply file(s) found in {inp}")

    # 3. Create output directory
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 4. Encode each file
    cylin = True
    total_points = 0
    total_bits = 0
    timing = {"build_octree": 0.0, "build_chunks": 0.0, "entropy_encode": 0.0,
              "save_file": 0.0, "total": 0.0}
    n_files = len(ply_files)

    for ply_path in ply_files:
        print(f"\n[Encode] Processing: {ply_path.name}")
        output_path = out / f"{ply_path.stem}.octree"

        bpp, n_pts = encode_single_file(
            ply_path=str(ply_path),
            output_path=str(output_path),
            model=model,
            device=device,
            lidar_level=args.lidar_level,
            quant_size=quant_size,
            m=m, n=n, z_rate=z_rate,
            context_size=args.context_size,
            cylin=cylin,
            oct_center=oct_center,
            K_context=args.K_context,
            timing=timing,
        )

        total_bits += os.path.getsize(output_path) * 8
        total_points += n_pts

    print(f"\n[Encode] All done.  Total points: {total_points:,d}  "
          f"Total bits: {total_bits:,d}  "
          f"Overall BPP: {total_bits / max(total_points, 1):.4f}")
    print(f"[Encode] Timing (avg per file, {n_files} files):")
    print(f"  Build octree : {timing['build_octree']/n_files:.2f} s")
    print(f"  Build chunks : {timing['build_chunks']/n_files:.2f} s")
    print(f"  Entropy encode: {timing['entropy_encode']/n_files:.2f} s")
    print(f"  Save file    : {timing['save_file']/n_files:.2f} s")
    print(f"  Total        : {timing['total']/n_files:.2f} s")
    print_gpu_memory(device)


if __name__ == "__main__":
    main()