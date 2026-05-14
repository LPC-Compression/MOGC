"""Structured container for K-parent octree context features.

Replaces raw (N, K, 8) concatenated arrays with named fields, convenience
accessors, and clean save/load — eliminating fragile index-based slicing
like `[:, -1, 5:8]` throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OctreeContext:
    """K-parent context features for all octree nodes.

    Each field is indexed as (node, context_slot), where context_slot K-1 is
    the current node and slots 0..K-2 are ancestor context shifted from parent.

    Attributes
    ----------
    occupancy : (N, K) int32
        8-bit occupancy mask per context slot. Pad value = 256.
    level : (N, K) int32
        Tree depth per context slot (root = 1).
    cur_child_idx : (N, K) int32
        Child index (0-7) within parent per context slot.
    pos : (N, K, 3) float32
        Center (x, y, z) per context slot.
    extents : (N, K, 2) float32
        (extent, z_extent) per context slot.
    is_non_leaf : (N,) bool or None
        Leaf / non-leaf mask. None when leaf information is unavailable.
    """

    occupancy: np.ndarray
    level: np.ndarray
    cur_child_idx: np.ndarray
    pos: np.ndarray
    extents: np.ndarray
    is_non_leaf: np.ndarray | None = None

    # ── shape / sizing ──────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def context_size(self) -> int:
        return int(self.occupancy.shape[1])

    # ── last-slot accessors (current node) ──────────────────────

    @property
    def last_occupancy(self) -> np.ndarray:
        """(N,) int32 — occupancy of the current (last-slot) node."""
        return self.occupancy[:, -1]

    @property
    def last_level(self) -> np.ndarray:
        """(N,) int32 — depth of the current node."""
        return self.level[:, -1]

    @property
    def last_cur_child_idx(self) -> np.ndarray:
        """(N,) int32 — child index of the current node."""
        return self.cur_child_idx[:, -1]

    @property
    def last_pos(self) -> np.ndarray:
        """(N, 3) float32 — center xyz of the current node."""
        return self.pos[:, -1, :]

    @property
    def last_extents(self) -> np.ndarray:
        """(N, 2) float32 — (extent, z_extent) of the current node."""
        return self.extents[:, -1, :]

    # ── derived masks ───────────────────────────────────────────

    @property
    def non_leaf_mask(self) -> np.ndarray:
        """(N,) bool — True for non-leaf nodes."""
        if self.is_non_leaf is not None:
            return self.is_non_leaf
        return self.occupancy[:, -1] != 0

    @property
    def leaf_mask(self) -> np.ndarray:
        """(N,) bool — True for leaf nodes."""
        return ~self.non_leaf_mask

    # ── channel-stacked view (for training compatibility) ────────

    @property
    def channels(self) -> np.ndarray:
        """(N, K, 3) int64 — stacked (occupancy, level, cur_child_idx).

        Matches the `data = arr[:, :, :3]` slice that training code expects.
        """
        return np.stack(
            [self.occupancy, self.level, self.cur_child_idx], axis=-1
        ).astype(np.int64)

    # ── I/O ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save to .npz with named arrays.

        Parameters
        ----------
        path : str
            File path. ``.npz`` is appended if missing.
        """
        if not path.endswith(".npz"):
            path += ".npz"
        payload = {
            "occupancy": self.occupancy,
            "level": self.level,
            "cur_child_idx": self.cur_child_idx,
            "pos": self.pos,
            "extents": self.extents,
        }
        if self.is_non_leaf is not None:
            payload["is_non_leaf"] = self.is_non_leaf
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> OctreeContext:
        """Load from a .npz file written by :meth:`save`.

        Also accepts legacy files if they contain the expected named arrays.
        """
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)
        return cls(
            occupancy=data["occupancy"].astype(np.int32),
            level=data["level"].astype(np.int32),
            cur_child_idx=data["cur_child_idx"].astype(np.int32),
            pos=data["pos"].astype(np.float32),
            extents=data["extents"].astype(np.float32),
            is_non_leaf=data["is_non_leaf"].astype(bool)
            if "is_non_leaf" in data
            else None,
        )

    @classmethod
    def from_build_K_context(
        cls, ctx: dict, include_leaves: bool = False
    ) -> OctreeContext:
        """Construct directly from :func:`build_K_context_py` output dict.

        Parameters
        ----------
        ctx : dict
            Output of ``build_K_context_py`` with keys
            ``Occupancy``, ``Level``, ``CurChildIdx``, ``Pos``, ``Extents``,
            and optionally ``IsNonLeaf``.
        include_leaves : bool
            Must match the value passed to ``build_K_context_py``.
        """
        return cls(
            occupancy=ctx["Occupancy"].astype(np.int32),
            level=ctx["Level"].astype(np.int32),
            cur_child_idx=ctx["CurChildIdx"].astype(np.int32),
            pos=ctx["Pos"].astype(np.float32),
            extents=ctx["Extents"].astype(np.float32),
            is_non_leaf=ctx["IsNonLeaf"].astype(bool)
            if include_leaves
            else None,
        )

    # ── legacy bridge ────────────────────────────────────────────

    def to_legacy_array(self) -> np.ndarray:
        """Return (N, K, 8) concatenated array in the new-format layout.

        Channel layout (last dim):
            [0]     Occupancy
            [1]     Level
            [2]     CurChildIdx
            [3:5]   Extents (extent, z_extent)
            [5:8]   Pos (x, y, z)

        This matches the legacy ``(N, K, 8)`` channel layout and is
        compatible with the training dataloader slicing in
        ``data/gls_dataset.py``.
        """
        return np.concatenate(
            [
                self.occupancy[:, :, None],       # (N, K, 1)
                self.level[:, :, None],           # (N, K, 1)
                self.cur_child_idx[:, :, None],   # (N, K, 1)
                self.extents,                     # (N, K, 2)
                self.pos,                         # (N, K, 3)
            ],
            axis=2,
        )  # (N, K, 8)

    # ── slicing helpers ─────────────────────────────────────────

    def filter(self, mask: np.ndarray) -> OctreeContext:
        """Return a new instance with only the nodes where *mask* is True."""
        mask = np.asarray(mask, dtype=bool)
        return OctreeContext(
            occupancy=self.occupancy[mask],
            level=self.level[mask],
            cur_child_idx=self.cur_child_idx[mask],
            pos=self.pos[mask],
            extents=self.extents[mask],
            is_non_leaf=self.is_non_leaf[mask]
            if self.is_non_leaf is not None
            else None,
        )

    def filter_non_leaf(self) -> OctreeContext:
        """Return subset with only non-leaf nodes."""
        return self.filter(self.non_leaf_mask)

    def filter_leaf(self) -> OctreeContext:
        """Return subset with only leaf nodes."""
        return self.filter(self.leaf_mask)
