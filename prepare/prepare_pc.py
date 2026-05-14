import os

import numpy as np

from data.io import read_point_cloud
from data import SemanticKITTIDataset, FordDataset
from utils.pc_error import write_ply_with_normals


from tqdm import tqdm

if __name__ == "__main__":
    base_dir = "./test_PCs/"
    os.makedirs(base_dir, exist_ok=True)
    kitti_save_path = os.path.join(base_dir, "SemanticKITTI/")
    os.makedirs(kitti_save_path, exist_ok=True)
    kitti_test = SemanticKITTIDataset(split="test", sample_ratio=0.02)
    for i in tqdm(list(range(len(kitti_test)))):
        pc = kitti_test[i]["coord"].astype(np.float32)
        write_ply_with_normals(pc , os.path.join(kitti_save_path, f"{i}.ply"))


    ford_save_path = os.path.join(base_dir, "Ford/")
    os.makedirs(ford_save_path, exist_ok=True)
    ford_test = FordDataset(split="test")
    for i in tqdm(list(range(0, len(ford_test), 6))):
        pc = ford_test[i]["coord"].astype(np.float32)
        write_ply_with_normals(pc, os.path.join(ford_save_path, f"{i}.ply"))