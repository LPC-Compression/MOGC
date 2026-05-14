#!/usr/bin/env python3
"""
src/chunk_builder.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.feature_processor import _process_features


def build_level_chunks(
    data_all: torch.Tensor,         # (1, N, K, 3)
    extents_all: torch.Tensor,      # (1, N, 2)
    pos_all: torch.Tensor,          # (1, 3, N)
    labels_all: torch.Tensor,       # (1, N)
    context_size: int,
    min_level: int = 2,
    node_numlist=None,             
):
    """
    tuple[list[torch.Tensor], ...]
        (chunk_data_list, chunk_ext_list, chunk_pos_list, chunk_lbl_list,
         st_list, ed_list, level_list, pad_mask_list)
    """
    N = data_all.shape[1]
    levels = data_all[0, :, -1, 0].cpu().numpy()       # (N,)

    chunk_data_list, chunk_ext_list, chunk_pos_list, chunk_lbl_list = [], [], [], []
    st_list, ed_list, level_list, pad_mask_list = [], [], [], []

    first_valid_global_idx = -1
    max_loop_level = len(node_numlist) if node_numlist is not None else int(max(levels)) + 1

    for lvl in range(min_level, max_loop_level):
        idx = np.where(levels == lvl)[0]
        if len(idx) == 0:
            if node_numlist is not None:
                raise Exception(f"No nodes found for level {lvl}. Check node_numlist and levels.")
            continue

        if first_valid_global_idx == -1:
            first_valid_global_idx = idx[0]

        total = len(idx)
        for start in range(0, total, context_size):
            global_start = idx[start]
            global_end = idx[min(start + context_size, total) - 1] + 1
            actual_len = global_end - global_start
            pad_len = context_size - actual_len

            actual_global_start = global_start
            front_pad_len = 0

            chunk_d = data_all[:, actual_global_start:global_end]
            chunk_e = extents_all[:, actual_global_start:global_end]
            chunk_p = pos_all[:, :, actual_global_start:global_end]
            chunk_l = labels_all[:, actual_global_start:global_end]

            p_np = chunk_p[0].cpu().numpy()
            e_np = chunk_e[0].cpu().numpy()
            p_np, e_np = _process_features(p_np, e_np)
            chunk_p = torch.from_numpy(p_np).unsqueeze(0)
            chunk_e = torch.from_numpy(e_np).unsqueeze(0)

            padding_mask = torch.ones(1, chunk_d.shape[1], dtype=torch.int32)
            remain_pad = context_size - chunk_d.shape[1]
            if remain_pad > 0:
                chunk_d = F.pad(chunk_d, (0, 0, 0, 0, 0, remain_pad), value=0)
                chunk_e = F.pad(chunk_e, (0, 0, 0, remain_pad), value=0)
                chunk_p = F.pad(chunk_p, (0, remain_pad), value=0)
                chunk_l = F.pad(chunk_l, (0, remain_pad), value=-1)
                padding_mask = F.pad(padding_mask, (0, remain_pad), value=0)

            chunk_data_list.append(chunk_d)
            chunk_ext_list.append(chunk_e)
            chunk_pos_list.append(chunk_p)
            chunk_lbl_list.append(chunk_l)
            st_list.append(front_pad_len)
            ed_list.append(front_pad_len + actual_len)
            level_list.append(lvl)
            pad_mask_list.append(padding_mask)

    return (chunk_data_list, chunk_ext_list, chunk_pos_list, chunk_lbl_list,
            st_list, ed_list, level_list, pad_mask_list)