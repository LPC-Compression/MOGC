import os
import uuid

import numpy as np
import subprocess
from scipy.spatial import KDTree

from src.constants import PCERROR_BIN_PATH, TEMP_DIR


def estimate_normals_vectorized(points, k=30):
    n_points = points.shape[0]
    
    tree = KDTree(points)
    
    _, indices = tree.query(points, k=k)
    
    normals = np.zeros((n_points, 3))
    
    for i in range(n_points):
        neighbors = points[indices[i]]
        
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        cov = np.dot(centered.T, centered) / (k - 1)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        normals[i] = eigenvectors[:, 0]
    
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    centroid = np.mean(points, axis=0)
    directions = points - centroid
    dot_products = np.sum(normals * directions, axis=1)
    normals[dot_products < 0] *= -1
    
    return normals

def write_ply_with_normals(points,  filename):
    n_points = points.shape[0]

    normals = estimate_normals_vectorized(points)
    
    # 创建PLY文件头
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "end_header"
    ]
    
    with open(filename, 'w') as f:
        
        f.write('\n'.join(header) + '\n')
        
        for i in range(n_points):
            x, y, z = points[i]
            nx, ny, nz = normals[i]
            f.write(f"{x} {y} {z} {nx} {ny} {nz}\n")

def write_ply_data(filename, points, attributeName=[], attriType=[]):
    """
    write data to ply file.
    e.g pt.write_ply_data('ScanNet_{:5d}.ply'.format(idx), np.hstack((point,np.expand_dims(label,1) )) , attributeName=['intensity'], attriType =['uint16'])
    """
    # if os.path.exists(filename):
    #   os.system('rm '+filename)
    if type(points) is list:
        points = np.array(points)

    attrNum = len(attributeName)
    assert points.shape[1] >= (attrNum + 3)

    if os.path.dirname(filename) != "" and not os.path.exists(
        os.path.dirname(filename)
    ):
        os.makedirs(os.path.dirname(filename))

    plyheader = (
        ["ply\n", "format ascii 1.0\n"]
        + ["element vertex " + str(points.shape[0]) + "\n"]
        + ["property float x\n", "property float y\n", "property float z\n"]
    )
    for aN, attrName in enumerate(attributeName):
        plyheader.extend(["property " + attriType[aN] + " " + attrName + "\n"])
    plyheader.append("end_header")
    typeList = {"uint16": "%d", "float": "float", "uchar": "%d"}

    np.savetxt(
        filename,
        points,
        newline="\n",
        fmt=["%f", "%f", "%f"] + [typeList[t] for t in attriType],
        header="".join(plyheader),
        comments="",
    )

    return

def distChamfer(f1, f2, scale=1.0):
    f1 = f1 / scale
    f2 = f2 / scale
    tree = KDTree(f1, compact_nodes=False)
    d1, _ = tree.query(f2, k=1, workers=-1, eps=0)
    tree = KDTree(f2, compact_nodes=False)
    d2, _ = tree.query(f1, k=1, workers=-1, eps=0)
    return max(d1.mean(), d2.mean())

def get_psnr(tmp_test_file):
    with open(tmp_test_file) as f:
        c = f.readlines()
    f.close()
    for i in range(len(c)):
        if c[i].startswith('3.'):
            d1 = float(c[i+2].split(' ')[-1])
            try:
                d2 = float(c[i+4].split(' ')[-1])
            except Exception as e:
                d2 = 0.
            break
    os.remove(tmp_test_file)
    return d1, d2

def pcerror(
    pcRefer,
    pc,
    pcReferNorm_flag,
    pcerror_cfg_params,
    pcerror_result,
    pcerror_path=PCERROR_BIN_PATH,
):
    """
    Options:
            --help=0            This help text
      -a,   --fileA=""          Input file 1, original version
      -b,   --fileB=""          Input file 2, processed version
      -n,   --inputNorm=""      File name to import the normals of original point
                                cloud, if different from original file 1n
      -s,   --singlePass=0      Force running a single pass, where the loop is
                                over the original point cloud
      -d,   --hausdorff=0       Send the Haursdorff metric as well
      -c,   --color=0           Check color distortion as well
      -l,   --lidar=0           Check lidar reflectance as well
      -r,   --resolution=0      Specify the intrinsic resolution
            --dropdups=2        0(detect), 1(drop), 2(average) subsequent points
                                with same coordinates
            --neighborsProc=1   0(undefined), 1(average), 2(weighted average),
                                3(min), 4(max) neighbors with same geometric
                                distance
            --averageNormals=1  0(undefined), 1(average normal based on neighbors
                                with same geometric distance)
            --mseSpace=1        Colour space used for PSNR calculation
                                0: none (identity) 1: ITU-R BT.709 8: YCgCo-R
            --nbThreads=1       Number of threads used for parallel processing
    """
    if pcerror_result is not None:
        pcLabel = os.path.basename(pcerror_result).split(".")[0]
    else:
        pcLabel = "pt0"
    a = ''
    b = ''
    pcReferNorm = None
    if type(pc) is not str:
        a = str(uuid.uuid4()) + '.ply'
        write_ply_data(f"{TEMP_DIR}/{pcLabel}{a}", pc)
        pc = f"{TEMP_DIR}/{pcLabel}{a}"
    if type(pcRefer) is not str:
        temp_b = str(uuid.uuid4())
        b =  temp_b + '.ply'

        if pcReferNorm_flag:
            pcReferNorm = f"{TEMP_DIR}/{pcLabel}{temp_b}_n.ply"
            write_ply_with_normals(pcRefer, pcReferNorm)
        write_ply_data(f"{TEMP_DIR}/{pcLabel}{b}", pcRefer)
        pcRefer = f"{TEMP_DIR}/{pcLabel}{b}"

    if pcerror_result is not None:
        f = open(pcerror_result, "w")
    else:
        import sys

        f = sys.stdout
    if type(pcerror_cfg_params) is str:
        pcerror_cfg_params = pcerror_cfg_params.split(" ")
    if not pcReferNorm_flag:
        subprocess.Popen(
            [pcerror_path, "-a", pcRefer, "-b", pc] + pcerror_cfg_params,
            stdout=f,
            stderr=f,
        ).wait()
    else:
        subprocess.Popen(
            [pcerror_path, "-a", pcRefer, "-b", pc, "-n", pcReferNorm]
            + pcerror_cfg_params,
            stdout=f,
            stderr=f,
        ).wait()
    if pcerror_result:
        f.close()
    if a:
        os.remove(pc)
    if b:
        os.remove(pcRefer)
        if pcReferNorm_flag:
            os.remove(pcReferNorm)
