import math
import os
from typing import Literal, TypedDict

import numpy as np
from torch.utils.data import Dataset

from src.constants import DEFAULT_KITTI_DATA_ROOT

class KITTIPointCloudItem(TypedDict):
    coord: np.ndarray          # shape: [N, 3]
    reflectance: np.ndarray    # shape: [N]
    data_path: str

class SemanticKITTIDataset(Dataset):
    def __init__(self, data_root:str=DEFAULT_KITTI_DATA_ROOT,
                    split:Literal["train", "val", "test"]="train", 
                    sample_ratio:float=1):
        super(SemanticKITTIDataset, self).__init__()
        self.split = split
        self.data_root = data_root
        self.sample_ratio = sample_ratio
        assert 0 < sample_ratio <= 1
        self.data_list = self.get_data_list()
    
    def get_dataset_name(self) -> str:
        return "SemanticKITTIDataset"
    
    def get_data_list(self) -> list:
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        else:
            raise NotImplementedError
        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            if self.sample_ratio < 1:
                total_nums = len(seq_files)
                sampled_count = max(2, math.ceil(total_nums * self.sample_ratio))
                new_seq_files = [seq_files[0], seq_files[total_nums-1]]
                step = (total_nums - 1) / (sampled_count - 1)
                for i in range(1, sampled_count-1):
                    new_seq_files.insert(-1, seq_files[math.ceil(i * step)])
                seq_files = new_seq_files
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx) -> KITTIPointCloudItem:
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        reflectance = scan[:, 3]
        return {"coord": coord.astype(np.float32), 
                "reflectance": reflectance.astype(np.float32),
                "data_path": data_path}