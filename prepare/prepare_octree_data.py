import concurrent.futures
import json
import os
import shutil
import uuid
from typing import Union

import numpy as np
from tqdm import tqdm

from data import SemanticKITTIDataset, FordDataset 
from utils.octree_preproc_uitls import new_glsproc_pc
from utils.pc_error import pcerror, get_psnr, distChamfer
from src.constants import DATASET_PARAMS, TEMP_DIR

def augment_point_cloud(points: np.ndarray, 
                        rot_z_range=(-np.pi, np.pi), 
                        scale_range=(0.95, 1.05), 
                        translation_std=0.5, 
                        dropout_max_ratio=0.05) -> np.ndarray:
    assert points.shape[1] == 3, f"Expected input shape (N, 3), but got {points.shape}"
    
    aug_points = points.copy()
    if rot_z_range is not None:
        angle = np.random.uniform(rot_z_range[0], rot_z_range[1])
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        rot_matrix = np.array([
            [cos_val, -sin_val, 0],
            [sin_val,  cos_val, 0],
            [0,        0,       1]
        ])
        aug_points = np.dot(aug_points, rot_matrix.T)
        
    if scale_range is not None:
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        aug_points = aug_points * scale_factor
        
    if translation_std > 0:
        translation = np.random.normal(0, translation_std, size=(1, 3))
        translation = np.clip(translation, -3 * translation_std, 3 * translation_std)
        aug_points = aug_points + translation
    if dropout_max_ratio > 0:
        dropout_ratio = np.random.uniform(0, dropout_max_ratio)
        num_points = aug_points.shape[0]
        keep_mask = np.random.random(num_points) > dropout_ratio
        aug_points = aug_points[keep_mask]
        
    return aug_points

def _worker_process_task(origin_pc, idx, config):
    result = new_glsproc_pc(
        pc = origin_pc,
        out_dir = config['save_dir'],
        out_name = str(idx) + "_" + str(origin_pc.shape[0]),
        oct_center = config['oct_center'],
        lidar_level = config['lidar_level'],
        quant_size = config['quant_size'],
        cylin = config['cylin'],
        resize = config['resize'],
        m = config['m'],
        n = config['n'],
        z_rate = config['z_rate'],
        include_leaves = False,
    )
    out_file = result['out_file']
    reconstruct_pc = result['out_points']
    origin_pc_processed = result['original_pc']
    nodenum = result['node_num']
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    tmp_test_file = f"{TEMP_DIR}/pcerror_results_" + str(uuid.uuid4()) + ".txt"
    
    pcerror(origin_pc_processed, reconstruct_pc, True, "-r " + config['peak'], tmp_test_file)
    psnrd1, psnrd2 = get_psnr(tmp_test_file)
    cd = distChamfer(origin_pc_processed, reconstruct_pc, config['scale'])
    
    if os.path.exists(tmp_test_file):
        os.remove(tmp_test_file)
        
    return len(origin_pc_processed), cd, psnrd1, psnrd2, nodenum

class OctreeDataBuildTool():
    def __init__(self, dataset:Union[SemanticKITTIDataset, FordDataset], lidar_level=12, quant_size=50, \
                m=1.0, n=0.015, z_rate=1, num_workers = 4, \
                 save_dir = 'DATA/', cylin=True, resize='exp'):
        self.dataset = dataset
        self.save_dir = save_dir
        self.lidar_level = lidar_level
        self.cylin = cylin
        self.resize = resize

        self.m = m
        self.n = n
        self.z_rate=z_rate
        self.num_workers = num_workers

        dataset_name = dataset.get_dataset_name()
        params = DATASET_PARAMS.get(dataset_name)
        if params is None:
            raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")
        self.peak = params["peak"]
        self.oct_center = params["oct_center"].copy()
        self.scale = params["scale"]
        if params["quant_size"] is not None:
            self.quant_size = params["quant_size"]
        else:
            self.quant_size = 2**(18 - self.lidar_level) 
    
    def begin_build(self):
        os.makedirs(self.save_dir, exist_ok=True)
        total_num = len(self.dataset)
        all_results = {}
        
        task_config = {
            'save_dir': self.save_dir,
            'oct_center': self.oct_center,
            'lidar_level': self.lidar_level,
            'quant_size': self.quant_size,
            'cylin': self.cylin,
            'resize': self.resize,
            'm': self.m,
            'n': self.n,
            'z_rate': self.z_rate,
            'peak': self.peak,
            'scale': self.scale
        }

        pipeline_size = self.num_workers * 4

        
        with tqdm(total=total_num, desc="Processing Point Clouds") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                pending: dict = {}
                next_idx = 0  
                while len(pending) < pipeline_size and next_idx < total_num:
                    origin_pc = self.dataset[next_idx]["coord"]
                    future = executor.submit(_worker_process_task, origin_pc, next_idx, task_config)
                    pending[future] = next_idx
                    next_idx += 1
                
                while pending:
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        idx = pending.pop(future)
                        try:
                            result = future.result()
                            num_points, cd, psnrd1, psnrd2, nodenum = result
                            all_results[idx] = {
                                "num_points": num_points,
                                "cd": cd,
                                "psnrd1": psnrd1,
                                "psnrd2": psnrd2,
                                "nodenum": nodenum
                            }
                        except Exception as exc:
                            print(f"{exc}")
                        finally:
                            pbar.update(1)
                            if next_idx < total_num:
                                origin_pc = self.dataset[next_idx]["coord"]
                                future = executor.submit(_worker_process_task, origin_pc, next_idx, task_config)
                                pending[future] = next_idx
                                next_idx += 1
        return all_results


def main(save_dir='DATA/SemanticKITTI/test/'):
    tree_layer_indexs = [10, 11, 12, 13, 14, 15]
    kitti_test_dataset = SemanticKITTIDataset(split="test", sample_ratio=0.01)
# def main(save_dir='DATA/SemanticKITTI/train/'):
#     tree_layer_indexs = [15]
#     kitti_test_dataset = SemanticKITTIDataset(split="train", sample_ratio=1)
    Depths = []
    CDs = []
    D1 = []
    D2 = []
    node_nums = []

    for depth in tree_layer_indexs:
        build_tool = OctreeDataBuildTool(
            dataset=kitti_test_dataset,
            lidar_level=depth,
            num_workers=24,
            save_dir=save_dir + f"level_{depth}/",
            resize="exp",
            m = 1,
            n = 0.015,
            quant_size = 50
        )
        results = build_tool.begin_build()
        mean_cd = np.mean([res["cd"] for res in results.values()])
        mean_psnrd1 = np.mean([res["psnrd1"] for res in results.values()])
        mean_psnrd2 = np.mean([res["psnrd2"] for res in results.values()])
        mean_nodenum = np.mean([res["nodenum"] for res in results.values()])
        node_nums.append(mean_nodenum)
        print(f"Depth {depth} - Mean CD: {mean_cd:.6f}, Mean PSNR-D1: {mean_psnrd1:.2f}, Mean PSNR-D2: {mean_psnrd2:.2f}, Mean NodeNum: {mean_nodenum:.1f}")
        Depths.append(depth)
        CDs.append(mean_cd)
        D1.append(mean_psnrd1)
        D2.append(mean_psnrd2)

        with open(save_dir + f"level_{depth}/" + f'results_{depth}.json', 'w') as f:
            json.dump(results, f, indent=4)

    print("Depths:", [depth for depth in Depths])
    print("Node Numbers:", node_nums)
    print("CDs:", CDs)
    print("D1:", D1)
    print("D2:", D2)


if __name__ == "__main__":
    main()
