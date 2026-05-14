#!/usr/bin/env python3
"""
src/entropy_coding.py 
"""

import os
import struct
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchac

from models.prob_model.self_prob_model import sort_use_indices_3dim


# ---------- bit utilities (decode_edit) ----------

def bin_swap(x: np.ndarray) -> np.ndarray:
    """Reverse bit order per byte (for occupancy de-packing)."""
    map_ = np.array([int('{:08b}'.format(b)[::-1], 2) for b in range(256)],
                    dtype=np.uint8)
    return map_[x.astype(np.uint8)]


def dec2bin_ary(dec: np.ndarray, N: int) -> np.ndarray:
    """dec → (M, N) bit array (MSB first), e.g. occupancy codes → 8 bits."""
    dec = np.atleast_1d(np.asarray(dec, dtype=np.int64))
    out = np.zeros((len(dec), N), dtype=np.int32)
    for b in range(N):
        out[:, N - 1 - b] = (dec >> b) & 1
    return out


# ---------- arithmetic encode (encode.py) ----------
@torch.no_grad()
def encode_chunks_entropy(
    model,
    chunk_data_list, chunk_ext_list, chunk_pos_list, chunk_lbl_list,
    st_list, ed_list, level_list, pad_mask_list,
    device: torch.device,
):
    """TorchAC arithmetic coding on chunked features (even/odd split).

    All even-CDF entries are concatenated first, then all odd-CDF entries.
    A single ``encode_float_cdf`` call per level ensures the byte stream is
    self-consistent (independent encode calls cannot be concatenated).

    Returns (level_to_bytes: Dict[int, bytes], total_estimated_bits: float)
    """
    raw_model = model
    # Accumulate CDFs and symbols per level: separate even/odd lists
    level_even_cdfs: Dict[int, List[torch.Tensor]] = {}
    level_even_syms: Dict[int, List[torch.Tensor]] = {}
    level_odd_cdfs: Dict[int, List[torch.Tensor]] = {}
    level_odd_syms: Dict[int, List[torch.Tensor]] = {}
    total_estimated_bits = 0.0

    def _compute_cdf_and_est(probs_slice, labels_slice, valid_mask):
        valid_probs = probs_slice[0][valid_mask[0].cpu()]
        valid_symbols = labels_slice[0][valid_mask[0]].cpu()
        est = -torch.log2(
            valid_probs.gather(1, valid_symbols.unsqueeze(-1))
            .squeeze(-1).clamp(min=1e-9)
        ).sum().item()
        cdf = torch.cumsum(valid_probs, dim=-1)
        cdf[:, -1] = 1.0
        cdf_wz = torch.cat([torch.zeros(cdf.shape[0], 1), cdf.clamp(max=1.0)], dim=-1)
        return cdf_wz, valid_symbols.to(torch.int16), est

    for i in range(len(chunk_data_list)):
        lvl = level_list[i]
        d = chunk_data_list[i].to(device)
        e = chunk_ext_list[i].to(device)
        p = chunk_pos_list[i].to(device)
        lbl = chunk_lbl_list[i].to(device)
        pm = pad_mask_list[i].to(device).bool()
        st = st_list[i]
        ed = ed_list[i]
        encode_mask = torch.zeros_like(lbl, dtype=torch.bool)
        encode_mask[:, st:ed] = True

        feat_a1, feat_a2, pre_occ_embed, pre_occ, codes, restore_indices = \
            raw_model.forward_features(d, e, p, pm)

        prob_sorted = raw_model.predicthead(feat_a1, feat_a2, pre_occ_embed, codes)
        probs_sorted = F.softmax(prob_sorted, dim=-1)  # (1, CS, 256) in SORTED order

        # Sort labels to match prob_sorted (Hilbert-sorted) order
        sorted_indices = restore_indices.argsort(dim=1)  # (1, CS)
        lbl_sorted = torch.gather(lbl, 1, sorted_indices)
        encode_mask_sorted = torch.gather(encode_mask, 1, sorted_indices)

        # even/odd in SORTED order (prob_sorted[:, ::2] = prob1, prob_sorted[:, 1::2] = prob2)
        even_probs = probs_sorted[:, ::2, :].cpu()
        even_labels = lbl_sorted[:, ::2]
        even_valid = encode_mask_sorted[:, ::2]
        cdf1_wz, even_syms, even_est = _compute_cdf_and_est(even_probs, even_labels, even_valid)

        odd_probs = probs_sorted[:, 1::2, :].cpu()
        odd_labels = lbl_sorted[:, 1::2]
        odd_valid = encode_mask_sorted[:, 1::2]
        cdf2_wz, odd_syms, odd_est = _compute_cdf_and_est(odd_probs, odd_labels, odd_valid)

        total_estimated_bits += even_est + odd_est

        level_even_cdfs.setdefault(lvl, []).append(cdf1_wz)
        level_even_syms.setdefault(lvl, []).append(even_syms)
        level_odd_cdfs.setdefault(lvl, []).append(cdf2_wz)
        level_odd_syms.setdefault(lvl, []).append(odd_syms)

    # Separate even/odd encoding per level with 4-byte even-length prefix
    level_to_bytes = {}
    for lvl in level_even_cdfs:
        even_cdf = torch.cat(level_even_cdfs[lvl], dim=0)
        even_sym = torch.cat(level_even_syms[lvl], dim=0)
        odd_cdf = torch.cat(level_odd_cdfs[lvl], dim=0)
        odd_sym = torch.cat(level_odd_syms[lvl], dim=0)
        even_all = torchac.encode_float_cdf(even_cdf, even_sym)
        odd_all = torchac.encode_float_cdf(odd_cdf, odd_sym)
        level_to_bytes[lvl] = struct.pack("<I", len(even_all)) + even_all + odd_all
    return level_to_bytes, total_estimated_bits


# ---------- serialisation helpers ----------

def _serialize_recon_params(params):
    if params is None:
        return None
    if isinstance(params, dict):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in params.items()}
    return params


def save_encoded_file(
    output_path: str,
    level_to_bytes: Dict[int, bytes],
    node_numlist: List[int],
    max_r: float,
    oct_center: np.ndarray,
    qs: np.ndarray,
    recon_params,
    m: float, n: float, z_rate: float,
    lidar_level: int, quant_size: float,
    cylin: bool, resize: str, K_context: int,
    num_points: int,
    root_occupancy: int,
    min_encode_level: int = 2,
    estimated_bits: float = 0.0,
):
    """Write .OCTR file with magic header + pickle metadata + directory + body."""
    meta = dict(
        root_occupancy=root_occupancy,
        node_numlist=node_numlist,
        max_r=max_r,
        oct_center=oct_center.tolist() if isinstance(oct_center, np.ndarray) else oct_center,
        qs=qs.tolist() if isinstance(qs, np.ndarray) else qs,
        recon_params=_serialize_recon_params(recon_params),
        m=m, n=n, z_rate=z_rate,
        lidar_level=lidar_level, quant_size=quant_size,
        cylin=cylin, resize=resize, K_context=K_context,
        num_points=num_points,
        num_encode_level=len(node_numlist),
        min_encode_level=min_encode_level,
    )
    meta_bytes = pickle.dumps(meta)
    meta_len = len(meta_bytes)

    dir_entries: List[Tuple[int, int, int]] = []
    body_parts: List[bytes] = []
    offset = 0
    for lvl in sorted(level_to_bytes.keys()):
        data = level_to_bytes[lvl]
        dir_entries.append((int(lvl), int(offset), int(len(data))))
        body_parts.append(data)
        offset += len(data)

    body = b"".join(body_parts)

    with open(output_path, "wb") as f:
        f.write(b"OCTR")
        f.write(struct.pack("<I", meta_len))
        f.write(meta_bytes)
        f.write(struct.pack("<I", len(dir_entries)))
        for lvl_id, off, length in dir_entries:
            f.write(struct.pack("<I q q", lvl_id, off, length))
        f.write(body)

    total_bytes = 4 + 4 + meta_len + 4 + 20 * len(dir_entries) + len(body)
    body_bytes = len(body)
    est_bpp = estimated_bits / max(num_points, 1)

    print(f"  Saved → {output_path}")
    print(f"    Levels encoded: {sorted(level_to_bytes.keys())}")
    print(f"    Total nodes : {sum(node_numlist)}")
    print(f"    File size   : {os.path.getsize(output_path):,d} B  ({total_bytes:,d} B computed)")
    print(f"    body size   : {body_bytes:,d} B")
    print(f"    Points      : {num_points:,d}")
    print(f"    body bpp    : {body_bytes * 8 / max(num_points, 1):.4f}")
    print(f"    Estimated entropy bits: {estimated_bits:.2f}  Est. BPP: {est_bpp:.4f}")
    print(f"    max_r       : {max_r:.4f}")