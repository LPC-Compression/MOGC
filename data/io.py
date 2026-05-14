import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import torch
import os

# Utility functions for reading/writing point cloud data using PyntCloud
def read_point_cloud(filepath):
    pc = PyntCloud.from_file(filepath)
    pc = np.array(pc.points, dtype=np.float32)[:, :3]
    return pc

def save_point_cloud(pc, path):
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    df = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(path)

# Utility function to read file size
def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8