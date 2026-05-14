# MOGC — Multi-Order Interleaved Sequential Context Modeling for LiDAR Point Cloud Geometry Compression

## Features

- probability model with multi-order interleaved sequential context
- Swin Transformer encoder for context aggregation
- Cylindrical coordinate octree representation for LiDAR data
- TorchAC arithmetic coding with even/odd split
- Autoregressive decoding with optional prediction extension
- End-to-end pipeline: training → encoding → decoding → evaluation

---

## Environment Setup

### Requirements

- Python 3.10+
- PyTorch 2.x
- PyTorch Lightning
- Hydra + OmegaConf
- torchac (PyTorch arithmetic coding)
- PyntCloud (PLY I/O)
- wandb (logging)
- scipy, numpy, tqdm

Install dependencies:

```bash
pip install torch pytorch-lightning hydra-core omegaconf torchac pyntcloud wandb scipy tqdm
```

### Compile C++ Octree Extension

The core octree building operations use a pybind11 C++ extension:

```bash
cd utils/c_octree_utils
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) octreeutils.cpp -o octreeutils$(python3-config --extension-suffix)
```

### MPEG PCC Evaluation Binary

Download the MPEG point cloud compression `pc_error` evaluation binary and place it at:

```
utils/pc_error
```

Update `PCERROR_BIN_PATH` in `src/constants.py` if using a different path.

---

## Data Preparation

### SemanticKITTI

Download the [SemanticKITTI dataset](http://www.semantic-kitti.org/) and place it with the following structure:

```
<kitti_root>/sequences/00/velodyne/*.bin
<kitti_root>/sequences/01/velodyne/*.bin
...
<kitti_root>/sequences/21/velodyne/*.bin
```

### Ford

Download the Ford dataset with the following structure:

```
<ford_root>/train/Ford_01_q_1mm/*.ply
<ford_root>/test/Ford_02_q_1mm/*.ply
<ford_root>/test/Ford_03_q_1mm/*.ply
```

---

## Path Configuration

After downloading the datasets, update the following files to point to your local paths:

### `src/constants.py`

```python
# SemanticKITTI dataset root
DEFAULT_KITTI_DATA_ROOT: str = "/your/path/to/SemanticKITTI/"

# Ford dataset paths
FORD_PATH: dict[str, list[str]] = {
    "train": ["/your/path/to/Ford/train/Ford_01_q_1mm/"],
    "test": ["/your/path/to/Ford/test/Ford_02_q_1mm/",
              "/your/path/to/Ford/test/Ford_03_q_1mm/"],
}

# Path to compiled pc_error binary
PCERROR_BIN_PATH: str = "utils/pc_error"
```

### `configs/config.yaml`

```yaml
absolute_path: "/your/path/to/MOGC/"
gpus: [0]          # GPU device IDs
project_name: "MOGC"
```

---

## Data Preprocessing

### Prepare Training Data (Octree Building)

Build octree data for training — converts raw point clouds into pre-processed octree features with quality metrics:

```bash
python prepare/prepare_octree_data.py
```

By default this processes SemanticKITTI test set at octree depths 10–15 and outputs to `DATA/SemanticKITTI/test/level_{depth}/`. To generate training data, modify `main()` in the script:

```python
def main(save_dir='DATA/SemanticKITTI/train/'):
    tree_layer_indexs = [15]
    kitti_test_dataset = SemanticKITTIDataset(split="train", sample_ratio=1)
```

For Ford:

```python
def main(save_dir='Processed_data/Ford/train/'):
    tree_layer_indexs = [17]
    kitti_test_dataset = FordDataset(split="train")
```

### Prepare Test Point Clouds

Export test point clouds as `.ply` files for the encoding pipeline:

```bash
python prepare/prepare_pc.py
```

Outputs to:
- `test_PCs/SemanticKITTI/{0,1,2,...}.ply` (every 50th test frame)
- `test_PCs/Ford/{0,6,12,...}.ply` (every 6th test frame)

---

## Training

Train the model using Hydra configuration:

```bash
# Train on SemanticKITTI (depth 15)
python train.py \
  dataset=kitti \
  model=self \
  run_name=kitti_self \
  gpus=[0] \
  absolute_path=/your/path/to/MOGC/

# Train on Ford (depth 17)
python train.py \
  dataset=ford \
  model=self \
  run_name=ford_self \
  gpus=[0] \
  absolute_path=/your/path/to/MOGC/
```

Key training config overrides (via CLI or `configs/train/default.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.train_data_dir` | `DATA/SemanticKITTI/train/level_15/` | Preprocessed training data |
| `data.val_data_dirs` | Level 10–15 directories | Preprocessed validation data |
| `data.batch_size` | 12 (KITTI), 16 (Ford) | Batch size per GPU |
| `train.optimizer.lr` | 2e-4 | Learning rate |
| `train.lr_scheduler.warmup_steps` | 4000 (KITTI), 5000 (Ford) | Warmup steps |
| `data.max_epochs` | 10 | Max training epochs |

Checkpoints are saved to `results/{run_name}/ckpt/`. To resume a W&B run, set `wandb_id` in `configs/config.yaml`.

---

## Encoding

Compress test point clouds into `.octree` binary files:

```bash
# SemanticKITTI
python encode.py \
  --input_dir ./test_PCs/SemanticKITTI/ \
  --output_dir ./encoded/ \
  --ckpt ./weights/kitti/model.ckpt \
  --config configs/model/self.yaml \
  --lidar_level 14 \
  --datatype kitti \
  --device cuda:0

# Ford
python encode.py \
  --input_dir ./test_PCs/Ford/ \
  --output_dir ./encoded_ford/ \
  --ckpt ./weights/ford/model.ckpt \
  --config configs/model/self.yaml \
  --lidar_level 17 \
  --datatype ford \
  --device cuda:0
```

| Argument | Description |
|----------|-------------|
| `--input_dir` | Directory containing `.ply` point clouds |
| `--output_dir` | Directory for output `.octree` files |
| `--ckpt` | Path to trained model checkpoint |
| `--config` | Model config YAML |
| `--lidar_level` | Octree max depth (14 for KITTI, 17 for Ford) |
| `--datatype` | `kitti` or `ford` |
| `--K_context` | Context window size (default 4) |
| `--device` | CUDA device |

---

## Decoding

Decode `.octree` files back to point clouds:

```bash
# SemanticKITTI
python decode.py \
  --input_dir ./encoded/ \
  --output_dir ./decoded/ \
  --ckpt ./weights/kitti/model.ckpt \
  --config configs/model/self.yaml \
  --device cuda:0 \
  --num_predict_layer 1

# Ford
python decode.py \
  --input_dir ./encoded_ford/ \
  --output_dir ./decoded_ford/ \
  --ckpt ./weights/ford/model.ckpt \
  --config configs/model/self.yaml \
  --device cuda:0 \
  --num_predict_layer 1
```

| Argument | Description |
|----------|-------------|
| `--input_dir` | Directory containing `.octree` files |
| `--output_dir` | Directory for output `.ply` files |
| `--ckpt` | Path to model checkpoint |
| `--config` | Model config YAML |
| `--num_predict_layer` | Additional prediction layers beyond encoded depth |
| `--threshold` | Confidence threshold for prediction expansion (default 0.0001) |
| `--device` | CUDA device |

---

## Evaluation

Compute quality metrics (Chamfer distance, D1 PSNR, D2 PSNR) between original and decoded point clouds:

```bash
# SemanticKITTI
python eval_quality.py \
  --original ./test_PCs/SemanticKITTI/ \
  --decoded ./decoded/ \
  --dataset SemanticKITTIDataset

# Ford
python eval_quality.py \
  --original ./test_PCs/Ford/ \
  --decoded ./decoded_ford/ \
  --dataset FordDataset
```

| Argument | Description |
|----------|-------------|
| `--original` | Directory of original `.ply` files |
| `--decoded` | Directory of decoded `.ply` files |
| `--dataset` | `SemanticKITTIDataset` or `FordDataset` |

Metrics reported per file: original points, decoded points, Chamfer distance (CD), D1 PSNR (dB), D2 PSNR (dB).

---

## Pre-trained Models

| Dataset | Checkpoint |
|---------|-----------|
| SemanticKITTI | `weights/kitti/model.ckpt` |
| Ford | `weights/ford/model.ckpt` |

---

## Project Structure

```
├── configs/
│   ├── config.yaml                    # Root Hydra config
│   ├── dataset/
│   │   ├── kitti.yaml                 # KITTI data + training params
│   │   └── ford.yaml                  # Ford data + training params
│   ├── model/
│   │   └── self.yaml                  # SelfOrderProbModel architecture
│   └── train/
│       └── default.yaml               # Default training hyperparameters
├── data/                              # Dataset classes
│   ├── semantic_kitti.py              # SemanticKITTI Dataset
│   ├── ford.py                        # Ford Dataset
│   ├── gls_dataset.py                 # Preprocessed octree data loader
│   └── io.py                          # PLY read/write utilities
├── models/                            # Model architecture
│   ├── modules/
│   │   ├── dgcnn.py                   # Geometry feature generator
│   │   └── nn.py                      # MLP / residual blocks
│   ├── prob_model/
│   │   ├── self_prob_model.py         # SelfOrderProbModel
│   │   ├── order_prob_model.py        # OrderProbModel
│   │   └── rotate_prob_model.py       # RotateOrderProbModel
│   ├── Attention_module/
│   │   ├── order_swin.py              # Swin encoder (z-order)
│   │   └── rotate_order_swin.py       # Swin encoder (rotated order)
│   └── train_modules/
│       └── base_train_module.py       # Lightning training module
├── prepare/
│   ├── prepare_octree_data.py         # Build preprocessed octree data
│   └── prepare_pc.py                  # Export test point clouds
├── src/
│   ├── constants.py                   # Dataset constants, paths, parameters
│   ├── model_loader.py                # Checkpoint loading
│   ├── octree_builder.py              # Octree + feature construction
│   ├── chunk_builder.py               # Context-chunk splitting
│   ├── entropy_coding.py              # TorchAC encode/decode + serialization
│   └── feature_processor.py           # Feature normalization
├── utils/
│   ├── c_octree_utils/                # C++ octree operations (pybind11)
│   ├── octree_preproc_uitls.py        # Octree preprocessing utilities
│   ├── octree_context.py              # Octree context data structure
│   ├── node_order.py                  # Hilbert/Z-order sorting
│   ├── pc_error.py                    # MPEG PCC evaluation wrapper
│   └── draw_img.py                    # RD curve plotting
├── weights/
│   ├── kitti/model.ckpt               # Pre-trained KITTI model
│   └── ford/model.ckpt                # Pre-trained Ford model
├── encode.py                          # Encoder entry point
├── decode.py                     # Decoder entry point
├── train.py                           # Training entry point
└── eval_quality.py                    # Evaluation entry point
```
