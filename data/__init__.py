"""Dataset definitions.

The ``data`` module defines loading logic for the SemanticKITTI and Ford
datasets, as well as loading logic for feature data after octree construction.
"""

from typing import Literal, Tuple

from .semantic_kitti import SemanticKITTIDataset
from .ford import FordDataset
from src.constants import DEFAULT_KITTI_DATA_ROOT


def get_semantic_kitti_dataset(
    data_root: str = DEFAULT_KITTI_DATA_ROOT,
    split: Literal["train", "val", "test"] = "train",
    sample_ratio: float = 1,
) -> SemanticKITTIDataset:
    """Create a SemanticKITTI dataset instance.

    Args:
        data_root: Local root directory of the SemanticKITTI dataset.
        split: Which partition of point clouds to load. Defaults to "train".
        sample_ratio: Sub-sampling ratio for interval-based subset. Defaults to 1.

    Returns:
        A SemanticKITTIDataset instance.
    """
    return SemanticKITTIDataset(
        data_root=data_root, split=split, sample_ratio=sample_ratio
    )


def get_ford_dataset(
    split: Literal["train", "test"] = "train",
) -> FordDataset:
    return FordDataset(split=split)