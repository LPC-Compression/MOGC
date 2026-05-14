#!/usr/bin/env python3
"""
src/constants.py -- Centralized project constants
==================================================
All global constants (boundaries, scaling factors, etc.) are defined here
and indexed by dataset name.

Usage:
    from src.constants import POS_MIN, POS_MAX, EXT_MIN, EXT_MAX, EPS, OCT_CENTER, get_constants
"""

import numpy as np

# --- Numerical precision --------------------------------------------------
EPS: float = 1e-8

# --- SemanticKITTI coordinate boundaries ----------------------------------
KITTI_POS_MIN = np.array([0.0, 0.0, -5.0], dtype=np.float32)
KITTI_POS_MAX = np.array([85.0, 85.0, 80.0], dtype=np.float32)
KITTI_EXT_MIN = np.array([0.0, 0.0], dtype=np.float32)
KITTI_EXT_MAX = np.array([50.0, 50.0], dtype=np.float32)
KITTI_OCT_CENTER = np.array([50.0, 50.0, 45.0], dtype=np.float32)

# --- Ford dataset coordinate boundaries -----------------------------------
FORD_POS_MIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
FORD_POS_MAX = np.array([2**17, 2**17, 2**17], dtype=np.float32)   # 131072.0
FORD_EXT_MIN = np.array([0.0, 0.0], dtype=np.float32)
FORD_EXT_MAX = np.array([2**18, 2**18], dtype=np.float32)          # 262144.0
FORD_OCT_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# --- Dataset PSNR / octree related parameters ------------------------------
KITTI_PEAK: str = "59.70"
KITTI_QUANT_SIZE: float = 50.0
KITTI_SCALE: float = 1.0

FORD_PEAK: str = "30000"
FORD_SCALE: float = 1.0
# Ford quant_size is dynamic: 2**(18 - lidar_level), not set here

# --- cart2cylin parameters -------------------------------------------------
KITTI_M: float = 1.0
KITTI_N: float = 0.015
KITTI_Z_RATE: float = 1.0

FORD_M: float = 1.0
FORD_N: float = 1e-5
FORD_Z_RATE: float = 1.0

# --- Indexed by dataset name (backward-compatible) -------------------------
_POS_MIN = {
    "SemanticKITTI": KITTI_POS_MIN,
    "Ford": FORD_POS_MIN,
}
_POS_MAX = {
    "SemanticKITTI": KITTI_POS_MAX,
    "Ford": FORD_POS_MAX,
}
_EXT_MIN = {
    "SemanticKITTI": KITTI_EXT_MIN,
    "Ford": FORD_EXT_MIN,
}
_EXT_MAX = {
    "SemanticKITTI": KITTI_EXT_MAX,
    "Ford": FORD_EXT_MAX,
}
_OCT_CENTER = {
    "SemanticKITTI": KITTI_OCT_CENTER,
    "Ford": FORD_OCT_CENTER,
}

# --- Backward compatibility: module-level variables, default KITTI ---------
POS_MIN = KITTI_POS_MIN
POS_MAX = KITTI_POS_MAX
EXT_MIN = KITTI_EXT_MIN
EXT_MAX = KITTI_EXT_MAX
OCT_CENTER = KITTI_OCT_CENTER

# --- OctreeDataBuildTool parameter table -----------------------------------
# Keyed by get_dataset_name() return value from data/ modules
DATASET_PARAMS = {
    "SemanticKITTIDataset": {
        "peak": KITTI_PEAK,
        "oct_center": KITTI_OCT_CENTER,
        "quant_size": KITTI_QUANT_SIZE,
        "scale": KITTI_SCALE,
        "m": KITTI_M,
        "n": KITTI_N,
        "z_rate": KITTI_Z_RATE,
    },
    "FordDataset": {
        "peak": FORD_PEAK,
        "oct_center": FORD_OCT_CENTER,
        "quant_size": None,          # dynamic: 2**(18 - lidar_level)
        "scale": FORD_SCALE,
        "m": FORD_M,
        "n": FORD_N,
        "z_rate": FORD_Z_RATE,
    },
}


def get_constants(dataset_name: str):
    """
    Return the full constant tuple given a dataset name.

    Parameters
    ----------
    dataset_name : str
        ``"SemanticKITTI"`` or ``"Ford"``

    Returns
    -------
    tuple[np.ndarray, ...]
        (pos_min, pos_max, ext_min, ext_max, eps, oct_center)
    """
    if dataset_name not in _POS_MIN:
        raise KeyError(
            f"Unknown dataset: {dataset_name}, supported: {list(_POS_MIN.keys())}"
        )
    return (
        _POS_MIN[dataset_name],
        _POS_MAX[dataset_name],
        _EXT_MIN[dataset_name],
        _EXT_MAX[dataset_name],
        EPS,
        _OCT_CENTER[dataset_name],
    )


def get_encode_params(dataset_name: str, lidar_level: int | None = None) -> dict:
    """Return full parameter dict for encoding, with dynamic Ford quant_size.

    Parameters
    ----------
    dataset_name : str
        Dataset key from ``DATASET_PARAMS`` (e.g. ``"SemanticKITTIDataset"``).
    lidar_level : int or None
        Octree depth; only needed for Ford to compute dynamic quant_size.

    Returns
    -------
    dict
        Keys: peak, oct_center, quant_size, scale, m, n, z_rate
    """
    params = dict(DATASET_PARAMS[dataset_name])
    if params["quant_size"] is None:
        if lidar_level is None:
            raise ValueError(
                f"lidar_level is required for {dataset_name} (dynamic quant_size)"
            )
        params["quant_size"] = float(2 ** (18 - lidar_level))
    return params


# --- Default dataset paths -------------------------------------------------
DEFAULT_KITTI_DATA_ROOT: str = "./SemanticKITTIDataset/"

FORD_PATH: dict[str, list[str]] = {
    "train": ["./Ford_full/train/Ford_01_q_1mm/"],
    "test": ["./Ford_full/test/Ford_02_q_1mm/",
              "./Ford_full/test/Ford_03_q_1mm/"],
}

# --- Utility / temp file paths ---------------------------------------------
PCERROR_BIN_PATH: str = "utils/pc_error"
TEMP_DIR: str = "temp"