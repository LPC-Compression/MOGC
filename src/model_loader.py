#!/usr/bin/env python3
"""
src/model_loader.py -- Unified model loading
=============================================
Loads SelfOrderProbModel and variants from a checkpoint, automatically
stripping the torch.compile (``_orig_mod``) prefix.
"""

import torch
from omegaconf import OmegaConf

from models.prob_model import PROB_MODEL_REGISTRY

# Short aliases to registered class names
_MODEL_ALIASES = {
    "self": "SelfOrderProbModel",
    "self_prob": "SelfOrderProbModel",
}


def load_model(ckpt_path: str, config_path: str, device: torch.device,
               model_type: str = "self"):
    """
    Load a pretrained model from a checkpoint, automatically stripping
    the ``_orig_mod`` prefix.

    Parameters
    ----------
    ckpt_path : str
        Path to the ``.ckpt`` file.
    config_path : str
        Path to the OmegaConf YAML configuration file.
    device : torch.device
    model_type : str
        ``"self"`` | ``"order"`` | ``"rotate"`` | ``"geov4"`` | ``"z_order_self"``

    Returns
    -------
    SelfOrderProbModel or subclass, already set to eval mode.
    """
    cls_name = _MODEL_ALIASES.get(model_type, "SelfOrderProbModel")
    if cls_name not in PROB_MODEL_REGISTRY:
        raise ValueError(
            f"Model {cls_name} (alias: {model_type}) not registered; missing module?"
        )
    model_cls = PROB_MODEL_REGISTRY[cls_name]
    cfg = OmegaConf.load(config_path)
    model = model_cls(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    raw_weights = {}
    for k, v in checkpoint["state_dict"].items():
        key = k[len("model."):] if k.startswith("model.") else k
        raw_weights[key] = v

    missing, unexpected = model.load_state_dict(raw_weights, strict=True)
    print(f"Weights loaded. missing keys: {missing}, unexpected keys: {unexpected}")
    model = model.to(device)
    model.eval()
    return model