import os
import sys
import math
from typing import Any, Dict

# Allow imports from project root when train.py is run from scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb

import warnings
warnings.filterwarnings("ignore", message=".*has_cuda.*")
warnings.filterwarnings("ignore", message=".*has_cudnn.*")
warnings.filterwarnings("ignore", message=".*has_mps.*")
warnings.filterwarnings("ignore", message=".*has_mkldnn.*")

from models import MODEL_REGISTRY
from data.gls_dataset import GLSLoader

seed = 42
seed_everything(seed)
torch.use_deterministic_algorithms(False)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    absolute_path = cfg.absolute_path
    config_dict: Dict[str, Any] = \
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 

    model_name = cfg.model.train_module_name
    model_class = MODEL_REGISTRY.get(model_name)
    if not model_class:
        raise ValueError(f"{model_name}")
    model = model_class(model_cfg=cfg.model, train_cfg=cfg.train)

    dataloader = GLSLoader(
        train_root=os.path.join(absolute_path, cfg.data.train_data_dir),
        test_root=[os.path.join(absolute_path, val_dir) for val_dir in cfg.data.val_data_dirs],
        context_size=cfg.model.context_size,
        dataset_name=cfg.data.dataset_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        trainloader_name=cfg.data.get("trainloader_name", ""),
        val_loader_name=cfg.data.get("val_loader_name", ""),
        max_depth= cfg.data.get("max_depth", 20),
    )

    run_name = cfg.run_name
    save_path = os.path.join(absolute_path, "results", run_name)
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_path, "ckpt"),
        filename='best_model-{epoch:02d}-{val_bpp:.4f}',
        save_top_k=-1,
        monitor='val_bpp',
        mode='min',
        save_last=True
    )
    
    wandb_id = config_dict.get("wandb_id", None)
    
    wandb_logger = WandbLogger(
        project=cfg.project_name,  
        name=run_name,       
        save_dir=save_path,      
        config=config_dict,
        id=wandb_id,          
        resume="allow"              
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        max_steps=math.ceil(cfg.train.lr_scheduler.max_steps),
        default_root_dir=save_path,
        deterministic=False,
        accelerator='gpu',
        devices=cfg.gpus,
        precision='16-mixed',
        logger=wandb_logger, # type: ignore
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0, 
        log_every_n_steps=cfg.train.log_every_n_steps,
        num_sanity_val_steps=50,
        val_check_interval=20000,  
        accumulate_grad_batches=cfg.data.get("accumulate_grad_batches", 1),
    )
 
    resume_ckpt_path = os.path.join(save_path, "ckpt", "None.ckpt") # ignore
    weights_only = False 
    

    if os.path.exists(resume_ckpt_path):
        if not weights_only:
            trainer.fit(
                model=model, 
                datamodule=dataloader, 
                ckpt_path=resume_ckpt_path,
            )
        else:
            checkpoint = torch.load(resume_ckpt_path, map_location="cpu")
            raw_weights = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("model."):
                    raw_weights[k[len("model."):]] = v
            if hasattr(model.model, "_orig_mod"):
                native_model = model.model._orig_mod # type: ignore
            else:
                native_model = model.model   
            missing, unexpected = native_model.load_state_dict(raw_weights, strict=True) # type: ignore
            trainer.fit(model=model, datamodule=dataloader)
    else:
        trainer.fit(model=model, datamodule=dataloader)
    

if __name__ == "__main__":
    main()
