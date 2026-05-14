#!/usr/bin/env python3
"""
src/feature_processor.py -- Feature normalization
==================================================
Provides the ``_process_features`` function that normalizes positions and
extents to [-1, 1]. The logic is identical across all scripts; only the
dataset-specific constants differ.

Originally from: kitti_test_predict.py, ford_test_predict.py,
                 ford_test_predict_optimized.py, decode_edit.py
"""

import numpy as np

from src.constants import POS_MIN, POS_MAX, EXT_MIN, EXT_MAX, EPS


def _process_features(pos: np.ndarray, extents: np.ndarray,
                      pos_min: np.ndarray = POS_MIN,
                      pos_max: np.ndarray = POS_MAX,
                      ext_min: np.ndarray = EXT_MIN,
                      ext_max: np.ndarray = EXT_MAX):
    """
    Normalize positions and extents to [-1, 1] -- consistent with
    encode.py / TestGLSDataset.

    Parameters
    ----------
    pos : np.ndarray
        Position, (3, N) or (N, 3), float32.
    extents : np.ndarray
        Extents, (N, 2), float32.
    pos_min, pos_max : np.ndarray
        Position boundaries, (3,).
    ext_min, ext_max : np.ndarray
        Extent boundaries, (2,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (pos_norm (3, N), extents_norm (N, 2)), dtype float32.
    """
    if pos.shape[-1] == 3 and pos.ndim == 2:
        pos = pos.T
    pos = 2.0 * (pos - pos_min.reshape(3, 1)) / (pos_max.reshape(3, 1) - pos_min.reshape(3, 1) + EPS) - 1.0
    pos = pos.astype(np.float32)
    extents = 2.0 * (extents - ext_min) / (ext_max - ext_min + EPS) - 1.0
    extents = extents.astype(np.float32)
    return pos, extents