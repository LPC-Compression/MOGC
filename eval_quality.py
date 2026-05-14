#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)  # eval/ → project root

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from utils.pc_error import pcerror, get_psnr, distChamfer
from src.constants import DATASET_PARAMS


def read_ply_xyz(path: str) -> np.ndarray:
    from data.io import read_point_cloud
    pts = read_point_cloud(path)
    if pts.ndim == 2 and pts.shape[1] >= 3:
        return pts[:, :3].astype(np.float64)
    return pts.astype(np.float64)


def _resolve_dataset_params(dataset_key: str):
    params = DATASET_PARAMS.get(dataset_key)
    if params is None:
        print(f"[Eval] WARNING: dataset '{dataset_key}' not in DATASET_PARAMS, "
              f"using default peak=1.0 scale=1.0")
        return {"peak": 1.0, "scale": 1.0}
    return params


def evaluate(
    original_ply: str,
    decoded_ply: str,
    dataset: str = "SemanticKITTIDataset",
) -> dict:
    ds = _resolve_dataset_params(dataset)
    peak = ds["peak"]
    scale = ds.get("scale", 1.0)

    orig = read_ply_xyz(original_ply)
    dec = read_ply_xyz(decoded_ply)

    result = {
        "file": Path(original_ply).name,
        "num_points_orig": orig.shape[0],
        "num_points_dec":  dec.shape[0],
    }

    cd = distChamfer(orig, dec, scale=scale)
    result["chamfer_dist"] = cd
    import uuid
    os.makedirs("temp", exist_ok=True)
    tmp_result = f"temp/pcerror_eval_{uuid.uuid4()}.txt"
    try:
        pcerror(orig, dec, True, "-r " + str(peak), tmp_result)
        d1, d2 = get_psnr(tmp_result)
    except Exception as e:
        print(f"[Eval] pc_error 失败: {e}")
        d1, d2 = None, None
        if os.path.exists(tmp_result):
            os.remove(tmp_result)

    result["d1_psnr"] = d1
    result["d2_psnr"] = d2
    return result


def print_result(result: dict):
    print(f"  File           : {result['file']}")
    print(f"  Orig points    : {result['num_points_orig']:,d}")
    print(f"  Dec  points    : {result['num_points_dec']:,d}")
    print(f"  Chamfer Dist   : {result['chamfer_dist']:.6f}")
    d1 = result.get("d1_psnr")
    d2 = result.get("d2_psnr")
    print(f"  D1 PSNR        : {d1 if d1 is not None else 'N/A'}")
    print(f"  D2 PSNR        : {d2 if d2 is not None else 'N/A'}")


def format_result_line(result: dict, prefix: str = "") -> str:
    d1 = result.get("d1_psnr")
    d2 = result.get("d2_psnr")
    return (
        f"{prefix}"
        f"orig_pts={result['num_points_orig']:,d}  "
        f"dec_pts={result['num_points_dec']:,d}  "
        f"CD={result['chamfer_dist']:.6f}  "
        f"D1={d1 if d1 is not None else 'N/A'}  "
        f"D2={d2 if d2 is not None else 'N/A'}"
    )


def _enumerate_pairs(orig_path: str, dec_path: str):
    orig_p = Path(orig_path)
    dec_p = Path(dec_path)

    if orig_p.is_file() and dec_p.is_file():
        return [(str(orig_p), str(dec_p), orig_p.stem)]

    if orig_p.is_dir() and dec_p.is_dir():
        orig_files = sorted(orig_p.glob("*.ply"))
        pairs = []
        for of in orig_files:
            df = dec_p / of.name
            if df.is_file():
                pairs.append((str(of), str(df), of.stem))
        return pairs

    raise FileNotFoundError(
        f"original={orig_path} / decoded={dec_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="./test_PCs/SemanticKITTI/")
    parser.add_argument("--decoded", default="decoded/")
    parser.add_argument("--dataset", default="SemanticKITTIDataset")
    args = parser.parse_args()

    pairs = _enumerate_pairs(args.original, args.decoded)
    if not pairs:
        print("[Eval]")
        return

    ds = _resolve_dataset_params(args.dataset)
    print(f"[Eval] dataset={args.dataset}  peak={ds['peak']}  scale={ds.get('scale', 1.0)}")

    results = []
    for orig_f, dec_f, stem in pairs:
        print(f"[Eval] {stem} ...", end=" ", flush=True)
        result = evaluate(orig_f, dec_f, args.dataset)
        results.append(result)
        d1 = result.get("d1_psnr")
        d2 = result.get("d2_psnr")
        print(f"CD={result['chamfer_dist']:.4f}  "
              f"D1={d1 if d1 is not None else 'N/A'}  "
              f"D2={d2 if d2 is not None else 'N/A'}")

    # 汇总表
    if len(results) > 1:
        print("\n" + "=" * 80)
        print(f"{'File':>8s}  {'Orig':>9s}  {'Dec':>8s}  {'CD':>10s}  {'D1(dB)':>8s}  {'D2(dB)':>8s}")
        print("-" * 80)
        for r in results:
            d1 = r.get("d1_psnr")
            d2 = r.get("d2_psnr")
            d1_s = f"{d1:.4f}" if d1 is not None else "N/A"
            d2_s = f"{d2:.4f}" if d2 is not None else "N/A"
            print(f"{Path(r['file']).stem:>8s}  "
                  f"{r['num_points_orig']:>9,d}  "
                  f"{r['num_points_dec']:>8,d}  "
                  f"{r['chamfer_dist']:>10.4f}  "
                  f"{d1_s:>8s}  "
                  f"{d2_s:>8s}")
        print("-" * 80)
    else:
        print("\n" + "=" * 55)
        print("  Quality Evaluation Result")
        print("=" * 55)
        print_result(results[0])
        print("=" * 55)


if __name__ == "__main__":
    main()