from typing import Literal, TypedDict

from glob import glob
import numpy as np
from torch.utils.data import Dataset

from .io import read_point_cloud
from src.constants import FORD_PATH


class FordPointCloudItem(TypedDict):
    coord: np.ndarray          # shape: [N, 3]
    data_path: str

class FordDataset(Dataset):
    def __init__(self, split:Literal["train", "test"]="train", sample_ratio=1.0):
        self.split = split
        self.sample_ratio = sample_ratio
        self.data_list = self.get_data_list()
    
    def get_dataset_name(self) -> str:
        return "FordDataset"
    
    def get_data_list(self) -> list[str]:
        data_list = []
        for path in FORD_PATH.get(self.split, []):
            data_list += glob(path + "*.ply")
        if self.sample_ratio < 1.0:
            data_list = data_list[::int(1/self.sample_ratio)]
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx) -> FordPointCloudItem:
        data_path = self.data_list[idx]
        coord = read_point_cloud(self.data_list[idx])
        return {"coord": coord.astype("float32"),
                "data_path": data_path}