import glob
from typing import List, Union
import re
import os
import random
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

from lightning.pytorch import LightningDataModule
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader
import torch

from src.constants import get_constants
from src.feature_processor import _process_features
from utils.octree_context import OctreeContext


# ======================================================================
# Shared module-level functions
# ======================================================================
def _setup_boundaries(dataset_name):
    """Resolve physical boundaries from dataset_name, eliminating duplicated code."""
    constants_key = dataset_name.replace("Dataset", "")
    pos_min, pos_max, ext_min, ext_max, epsilon, _ = get_constants(constants_key)
    return pos_min, pos_max, ext_min, ext_max, epsilon


def _build_chunks_numpy(data, extents, pos, label, level_list,
                        context_size, max_depth=None):
    """
    Group nodes by depth and split into fixed-length context chunks.

    Parameters
    ----------
    data : np.ndarray, shape (N, 4, 3)
        Raw node sequence.
    extents : np.ndarray, shape (N, 2)
        Normalized extent features.
    pos : np.ndarray, shape (3, N)
        Normalized position features.
    label : np.ndarray, shape (N,)
        Node labels.
    level_list : np.ndarray, shape (N,)
        Depth of each node.
    context_size : int
        Maximum chunk length.
    max_depth : int or None
        Maximum processing depth; None means unlimited.

    Returns
    -------
    list[tuple]
        Each element is (chunk_d, chunk_e, chunk_p, chunk_l, level, padding_mask, st, ed).
    """
    change_indices = np.flatnonzero(level_list[:-1] != level_list[1:]) + 1
    boundaries = np.concatenate(([0], change_indices, [len(level_list)]))
    first_valid_global_idx = -1
    chunks = []

    for idx in range(len(boundaries) - 1):
        st_idx = boundaries[idx]
        ed_idx = boundaries[idx + 1]
        level = level_list[st_idx]

        if level < 2:
            continue
        if max_depth is not None and level > max_depth:
            break

        if first_valid_global_idx == -1:
            first_valid_global_idx = st_idx

        total_len = ed_idx - st_idx
        if total_len == 0:
            raise ValueError(f"Level {level} has no data points, which is unexpected.")

        for i in range(0, total_len, context_size):
            global_start = st_idx + i
            global_end = min(st_idx + i + context_size, ed_idx)
            content_len = global_end - global_start
            pad_needed = context_size - content_len

            # Prefer padding from previously processed nodes;
            # if that is insufficient, zero-pad at the tail
            front_pad_len = 0
            if pad_needed > 0:
                front_pad_len = min(pad_needed, global_start - first_valid_global_idx)

            chunk_start = global_start - front_pad_len
            tail_pad_len = pad_needed - front_pad_len

            chunk_d = data[chunk_start:global_end]
            chunk_e = extents[chunk_start:global_end]
            chunk_p = pos[:, chunk_start:global_end]
            chunk_l = label[chunk_start:global_end]

            padding_mask = np.ones(context_size, dtype=np.int64)
            if tail_pad_len > 0:
                chunk_d = np.pad(chunk_d, ((0, tail_pad_len), (0, 0), (0, 0)),
                                 mode='constant', constant_values=0)
                chunk_e = np.pad(chunk_e, ((0, tail_pad_len), (0, 0)),
                                 mode='constant', constant_values=0)
                chunk_p = np.pad(chunk_p, ((0, 0), (0, tail_pad_len)),
                                 mode='constant', constant_values=0)
                chunk_l = np.pad(chunk_l, (0, tail_pad_len),
                                 mode='constant', constant_values=-1)
                padding_mask[-tail_pad_len:] = 0

            chunks.append((chunk_d, chunk_e, chunk_p, chunk_l, level, padding_mask,
                           front_pad_len, front_pad_len + content_len))

    return chunks


# ======================================================================
# Helper: unpack list[tuple] from _build_chunks_numpy into separate lists
# (used by TestGLSDataset.__getitem__)
# ======================================================================
def _unpack_chunks(chunks):
    """Unpack list[tuple] into 8 independent lists.

    _build_chunks_numpy tuple order: (d, e, p, l, level, mask, st, ed)
    Output order: (d, e, p, l, st, ed, level, mask)
    """
    if not chunks:
        return [], [], [], [], [], [], [], []
    cols = list(zip(*chunks))
    return (list(cols[0]), list(cols[1]), list(cols[2]), list(cols[3]),
            list(cols[6]), list(cols[7]), list(cols[4]), list(cols[5]))


def _process_file_data(current_file_data, context_size, pos_min, pos_max,
                       ext_min, ext_max, max_depth=None):
    """Shared file-data processing: feature extraction, normalization,
    channel reordering, and chunking."""
    pos = current_file_data[:, -1, 5:8]
    extents = current_file_data[:, -1, 3:5]
    pos, extents = _process_features(pos, extents, pos_min, pos_max, ext_min, ext_max)
    data = current_file_data[:, :, :3].astype(np.int64)
    data = np.roll(data, shift=-1, axis=2)
    label = data[:, -1, 2]
    level_list = data[:, -1, 0]
    return _build_chunks_numpy(data, extents, pos, label, level_list,
                               context_size, max_depth=max_depth)


def _process_octree_context(ctx, context_size, pos_min, pos_max,
                            ext_min, ext_max, max_depth=None):
    """Process OctreeContext into chunks using named fields.

    Semantically identical to :func:`_process_file_data` but uses
    :class:`OctreeContext` named accessors instead of raw-array index slicing.
    """
    # keep only non-leaf nodes (matching old pipeline behaviour)
    if ctx.is_non_leaf is not None:
        ctx = ctx.filter_non_leaf()

    pos = ctx.last_pos
    extents = ctx.last_extents
    pos, extents = _process_features(pos, extents, pos_min, pos_max, ext_min, ext_max)

    # channel order: [occupancy, depth, child_idx] -- same as old arr[:, :, :3]
    data = np.stack([ctx.occupancy, ctx.level, ctx.cur_child_idx], axis=-1).astype(np.int64)
    data = np.roll(data, shift=-1, axis=2)       # -> [depth, child_idx, occupancy]
    label = data[:, -1, 2]                        # occupancy of current node
    level_list = data[:, -1, 0]                   # depth of current node
    return _build_chunks_numpy(data, extents, pos, label, level_list,
                               context_size, max_depth=max_depth)


class OptimizedIterableTrainGLSDataset(IterableDataset):
    def __init__(self, root, context_size, dataset_name, max_depth=20,
                 mix_batch_size=6, min_pool_size=50, prefetch_files=6,
                 io_threads=4):
        self.dataset_name = dataset_name
        self.context_size = context_size
        # 1. Load file list
        self.file_names = sorted(glob.glob(os.path.join(root, "*.npz")))
        assert self.file_names, f'No files found in the specified root: {root}!'
        # 2. Physical boundary config (imported uniformly from constants.py)
        (self.pos_min, self.pos_max, self.extent_min, self.extent_max, self.epsilon) = \
            _setup_boundaries(self.dataset_name)
        self.max_depth = max_depth
        print(f"max Depth set to {max_depth}")

        # ---------- Multi-file interleaving buffer parameters ----------
        self.mix_batch_size = mix_batch_size     # trigger pool shuffle + drain after every N files
        self.min_pool_size = min_pool_size       # min chunks to keep in pool after shuffle
        self.prefetch_files = prefetch_files     # max concurrently background-loaded files
        io_threads = min(io_threads, os.cpu_count() or 1)

        # Create a thread pool in __init__, reused across epochs to avoid repeated overhead
        self._executor = ThreadPoolExecutor(max_workers=io_threads)

    def __del__(self):
        """Ensure thread pool is shut down with the object."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)

    def _process_single_file(self, file_path):
        """Load a new file, split by level, pad, and shuffle all chunks."""
        ctx = OctreeContext.load(file_path)
        chunks = _process_octree_context(ctx, self.context_size,
                                         self.pos_min, self.pos_max,
                                         self.extent_min, self.extent_max,
                                         max_depth=self.max_depth)
        random.shuffle(chunks)
        return chunks

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # 0. Randomness settings
        base_seed = torch.initial_seed()
        seed = base_seed + worker_info.id if worker_info is not None else base_seed
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

        # 1. Copy file list for this epoch shuffling
        worker_files = list(self.file_names)
        # 2. When multi-processing, split files evenly across workers to avoid duplication
        if worker_info is not None:
            per_worker = int(np.ceil(len(worker_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_files = worker_files[worker_id * per_worker: (worker_id + 1) * per_worker]
        random.shuffle(worker_files)

        n_files = len(worker_files)
        if n_files == 0:
            return

        # ==========================================
        # 3. Multi-file interleaving buffer (Mix Pool)
        # ==========================================
        chunk_pool = []           # global chunk buffer pool
        files_completed = 0       # number of files fully processed
        file_iter = iter(worker_files)
        pending = {}              # {Future: None}

        # Pre-fill: submit the first prefetch_files files
        for _ in range(min(self.prefetch_files, n_files)):
            fut = self._executor.submit(self._process_single_file, next(file_iter))
            pending[fut] = None

        while pending:
            done, _ = futures.wait(
                pending.keys(),
                return_when=futures.FIRST_COMPLETED
            )

            for fut in done:
                del pending[fut]
                chunks = fut.result()
                chunk_pool.extend(chunks)
                files_completed += 1

                # Replenish the next file to keep the pipeline full
                try:
                    next_fut = self._executor.submit(
                        self._process_single_file, next(file_iter)
                    )
                    pending[next_fut] = None
                except StopIteration:
                    pass

                # After every mix_batch_size files (or the last file),
                # shuffle the pool and yield a batch, controlling memory and
                # improving data mixing
                if (files_completed % self.mix_batch_size == 0
                        or files_completed == n_files):
                    random.shuffle(chunk_pool)
                    while len(chunk_pool) > self.min_pool_size:
                        yield chunk_pool.pop()

        # 4. Drain remaining chunks
        if chunk_pool:
            random.shuffle(chunk_pool)
            for chunk in chunk_pool:
                yield chunk


class TestGLSDataset(Dataset):
    def __init__(self, roots: Union[str, List[str]], context_size: int, dataset_name: str):
        self.dataset_name = dataset_name
        self.context_size = context_size

        if isinstance(roots, str):
            roots = [roots]

        # 1. Load file list
        self.file_items = self._build_interleaved_file_list(roots)
        assert self.file_items, f'No files found in the specified roots: {roots}!'
        # 2. Physical boundary config (imported uniformly from constants.py)
        (self.pos_min, self.pos_max, self.extent_min, self.extent_max, self.epsilon) = \
            _setup_boundaries(self.dataset_name)

    def _build_interleaved_file_list(self, roots: List[str]):
        """Parse depth and merge files from different folders in round-robin order."""
        file_groups = []
        for root_dir in roots:
            match = re.search(r'level_(\d+)', root_dir)
            if not match:
                raise ValueError(f"Cannot parse tree depth from directory name: {root_dir}")
            depth = int(match.group(1))
            files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
            if files:
                file_groups.append([(f, depth) for f in files])

        if not file_groups:
            return []

        interleaved_items = []
        max_len = max(len(g) for g in file_groups)
        for i in range(max_len):
            for g in file_groups:
                if i < len(g):
                    interleaved_items.append(g[i])

        return interleaved_items

    def _load_new_file(self, file_idx):
        """Load a new file and return the corresponding tree depth."""
        file_path, tree_depth = self.file_items[file_idx]
        ctx = OctreeContext.load(file_path)
        _, point_num, _ = file_path.split('/')[-1].split('_')
        point_num = int(point_num)
        return ctx, point_num, tree_depth

    def __getitem__(self, index):
        ctx, point_num, tree_depth = self._load_new_file(index)
        chunks = _process_octree_context(ctx, self.context_size,
                                         self.pos_min, self.pos_max,
                                         self.extent_min, self.extent_max,
                                         max_depth=None)
        (chunk_d_list, chunk_e_list, chunk_p_list, chunk_l_list,
         st_list, ed_list, batch_level_list, padding_mask_list) = _unpack_chunks(chunks)
        return (chunk_d_list, chunk_e_list, chunk_p_list, chunk_l_list,
                st_list, ed_list, point_num, batch_level_list, padding_mask_list, tree_depth)

    def __len__(self):
        return len(self.file_items)


def test_collate_fn(batch):
    # batch_size=1, directly take the first sample from the list
    sample = batch[0]
    chunk_d_list, chunk_e_list, chunk_p_list, chunk_l_list, \
        st_list, ed_list, point_num, batch_level_list, padding_mask_list, tree_depth = sample
    data_list_out = torch.from_numpy(np.stack(chunk_d_list))
    extents_list_out = torch.from_numpy(np.stack(chunk_e_list)).float()
    pos_list_out = torch.from_numpy(np.stack(chunk_p_list)).float()
    label_list_out = torch.from_numpy(np.stack(chunk_l_list))
    st_tensor = torch.tensor(st_list, dtype=torch.long)
    ed_tensor = torch.tensor(ed_list, dtype=torch.long)
    level_tensor = torch.tensor(batch_level_list, dtype=torch.long)
    padding_mask_tensor = torch.from_numpy(np.stack(padding_mask_list))
    return (data_list_out, extents_list_out, pos_list_out, label_list_out,
            st_tensor, ed_tensor, point_num, level_tensor, padding_mask_tensor, tree_depth)


class GLSLoader(LightningDataModule):
    def __init__(self, train_root, test_root, context_size, dataset_name,
                 batch_size, num_workers, pin_memory, drop_last,
                 max_depth=20,
                 trainloader_name="",
                 val_loader_name=""):
        super().__init__()
        self.train_root = train_root
        self.test_root = test_root
        self.context_size = context_size
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.trainloader_name = trainloader_name

        if trainloader_name == "IterableTrainGLSDataset":
            print("###########  Using IterableTrainGLSDataset #################")
            self.train_set = OptimizedIterableTrainGLSDataset(
                self.train_root, self.context_size, self.dataset_name, max_depth=max_depth)
        else:
            raise NotImplementedError()

        if val_loader_name == "TestGLSDataset":
            print("###########  Using TestGLSDataset #################")
            self.val_set = TestGLSDataset(self.test_root, self.context_size, self.dataset_name)
        else:
            raise NotImplementedError()

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        loader_train = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers, persistent_workers=True,
                                      pin_memory=self.pin_memory, drop_last=True,
                                      prefetch_factor=2)
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.val_set, batch_size=1, shuffle=False,
                                num_workers=self.num_workers, persistent_workers=True,
                                pin_memory=self.pin_memory, drop_last=False,
                                collate_fn=test_collate_fn)
        return loader_val