import math

from matplotlib import pyplot as plt
import lightning.pytorch as pl
import numpy as np
import wandb
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import torch._inductor.config
torch._inductor.config.benchmark_kernel = False

from utils.draw_img import draw_rd_curve

from ..prob_model import PROB_MODEL_REGISTRY

class BppRecorder:
    def __init__(self, max_level):
        self.max_level = max_level
        self.bits_list = np.zeros(max_level, dtype=np.float32)
        self.point_count = np.zeros(max_level, dtype=np.float32)
    
    def record(self, tree_depth:int, bits:float, point_count:float):
        assert 1 <= tree_depth <= self.max_level
        self.bits_list[tree_depth - 1] += bits
        self.point_count[tree_depth - 1] += point_count
    
    def clear(self):
        self.bits_list.fill(0.0)
        self.point_count.fill(0.0)
    
    def get_bpp_list_range(self, start_level, end_level):
        assert end_level <= self.max_level
        return (self.bits_list[start_level-1:end_level] / (self.point_count[start_level-1:end_level] + 1e-9)).tolist()


class LevelMetricsRecorder:
    def __init__(self, num_classes, max_level, ece_bins=10):
        self.num_classes = num_classes
        self.max_level = max_level
        self.ece_bins = ece_bins

    def to(self, device):
        if not hasattr(self, '_device') or self._device != device:
            self._device = device
            self.reset()
        return self

    def reset(self):
        device = self._device
        self.data = {}
        for lvl in range(1, self.max_level + 1):
            self.data[lvl] = {
                'total': torch.tensor(0, device=device, dtype=torch.int64),
                'correct': torch.tensor(0, device=device, dtype=torch.int64),
                'confusion': torch.zeros((self.num_classes, self.num_classes), device=device, dtype=torch.int64),
                'ece_acc': torch.zeros(self.ece_bins, device=device, dtype=torch.float64),
                'ece_conf': torch.zeros(self.ece_bins, device=device, dtype=torch.float64),
                'ece_count': torch.zeros(self.ece_bins, device=device, dtype=torch.int64)
            }

    def update(self, levels, preds, labels, mask):
        self.to(preds.device)
        B = levels.shape[0]

        for lvl in range(1, self.max_level + 1):
            lvl_mask = (levels == lvl)  # (B,)
            if not lvl_mask.any():
                continue

            batch_preds = preds[lvl_mask]    # (B_lvl, L, C)
            batch_labels = labels[lvl_mask]  # (B_lvl, L)
            batch_mask = mask[lvl_mask]      # (B_lvl, L)

            valid_preds = batch_preds[batch_mask]   # (N, C)
            valid_labels = batch_labels[batch_mask] # (N,)

            if valid_labels.numel() == 0:
                continue

            N = valid_labels.numel()
            pred_cls = valid_preds.argmax(dim=-1)  # (N,)
            correct = (pred_cls == valid_labels)   # (N,)

            d = self.data[lvl]
            d['total'] += N
            d['correct'] += correct.sum()


            flat_idx = valid_labels * self.num_classes + pred_cls  # (N,)
            ones = torch.ones(N, device=self._device, dtype=torch.int64)
            flat_counts = torch.zeros(self.num_classes * self.num_classes, device=self._device, dtype=torch.int64)
            flat_counts.scatter_add_(0, flat_idx, ones)
            d['confusion'] += flat_counts.view(self.num_classes, self.num_classes)

            # ECE 
            max_probs = torch.softmax(valid_preds, dim=-1).max(dim=-1).values  # (N,)
            bin_idx = torch.clamp((max_probs * self.ece_bins).long(), 0, self.ece_bins - 1)  # (N,)

            d['ece_count'].scatter_add_(0, bin_idx, torch.ones(N, device=self._device, dtype=torch.int64))
            d['ece_acc'].scatter_add_(0, bin_idx, correct.to(dtype=torch.float64))
            d['ece_conf'].scatter_add_(0, bin_idx, max_probs.to(dtype=torch.float64))



class BaseTrainModule(pl.LightningModule):
    def __init__(self, model_cfg, train_cfg):
        super().__init__()

        model = PROB_MODEL_REGISTRY[model_cfg.model_name](model_cfg)
        self.model = torch.compile(model)

        self.train_cfg = train_cfg
        self.model_cfg = model_cfg

        self.num_classes = self.model_cfg.num_class

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.bpp_recorder = BppRecorder(model_cfg.geo_feat_generator.max_level)
        self.loss_weight_full = getattr(train_cfg, 'loss_weight_full', 1.0)
        self.loss_weight1 = getattr(train_cfg, 'loss_weight1', 1.0)
        self.loss_weight2 = getattr(train_cfg, 'loss_weight2', 1.0)

        self.level_metrics_recorder = LevelMetricsRecorder(
            num_classes=self.num_classes,
            max_level=model_cfg.geo_feat_generator.max_level,
            ece_bins=10
        )
        self.draw_level_start = getattr(train_cfg, 'draw_level_start', 1)
        self.draw_level_end = getattr(train_cfg, 'draw_level_end', model_cfg.geo_feat_generator.max_level)
        self.CD_list = getattr(train_cfg, 'CD_list', [])
        self.D1_list = getattr(train_cfg, 'D1_list', [])
        self.D2_list = getattr(train_cfg, 'D2_list', [])

    def forward(self, data, extents, pos, padding_mask):
        if self.training:
            return self.model(data, extents, pos, padding_mask)
        else:
            raw_model = getattr(self.model, "_orig_mod", self.model)
            return raw_model(data, extents, pos, padding_mask)
    
    def configure_optimizers(self): # type: ignore
        optimizer_cfg = self.train_cfg.optimizer
        if optimizer_cfg.name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), \
                                      lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")
        
        if self.train_cfg.lr_scheduler is None:
            return { "optimizer": optimizer}
        
        lr_scheduler_cfg = self.train_cfg.lr_scheduler
        if lr_scheduler_cfg.name == "CosineWarmup":
            print(f"[CosineWarmup] max_steps={lr_scheduler_cfg.max_steps}, warmup_steps={lr_scheduler_cfg.warmup_steps}")
            max_steps = lr_scheduler_cfg.max_steps
            warmup_steps = lr_scheduler_cfg.warmup_steps
            base_lr = optimizer_cfg.lr
            min_lr = lr_scheduler_cfg.min_lr
            scheduler = LambdaLR(optimizer, self._cosine_warmup_lambda(max_steps, warmup_steps, base_lr, min_lr))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")
    
    @staticmethod
    def _cosine_warmup_lambda(max_steps, warmup_steps, base_lr, min_lr):
        min_lr_ratio = min_lr / base_lr
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        return lr_lambda
     
    def training_step(self, batch, batch_idx):
        data, extents, pos, labels, levels, padding_mask, st, ed = batch
        pred = self(data, extents, pos, padding_mask)
        raw_loss = F.cross_entropy(
            pred.view(-1, self.model_cfg.num_class),
            labels.reshape(-1),
            reduction='none',
            ignore_index=-1
        ) / math.log(2)
        raw_loss = raw_loss.view(labels.shape)  # (B, L)

        valid_mask = (labels != -1)  # (B, L)

        seq_len = labels.size(1)
        indices = torch.arange(seq_len, device=self.device)
        even_mask = (indices % 2 == 0).unsqueeze(0) & valid_mask
        odd_mask = (indices % 2 != 0).unsqueeze(0) & valid_mask

        loss = raw_loss[valid_mask].sum() / (valid_mask.sum() + 1e-9)
        loss1 = raw_loss[even_mask].sum() / (even_mask.sum() + 1e-9)
        loss2 = raw_loss[odd_mask].sum() / (odd_mask.sum() + 1e-9)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("loss_even", loss1, on_step=True, on_epoch=False)
        self.log("loss_odd", loss2, on_step=True, on_epoch=False)
        return self.loss_weight_full * loss + self.loss_weight1 * loss1 + self.loss_weight2 * loss2

    def validation_step(self, batch, batch_idx):
        data_list_out, extents_list_out, pos_list_out, label_list_out, \
           st_tensor, ed_tensor, point_num, level_tensor, padding_mask_tensor, tree_depth = batch
        MAX_BATCH_SIZE = 64
        num_chunks = data_list_out.shape[0]

        bits_count_tensor = torch.tensor(8.0, device=self.device)
        total_metrics = torch.zeros(6, device=self.device) # [total_loss, total_tokens, even_loss, even_tokens, odd_loss, odd_tokens]

        for j in range(0, num_chunks, MAX_BATCH_SIZE):
            end_idx = min(j + MAX_BATCH_SIZE, num_chunks)
            batched_d = data_list_out[j:end_idx]     # (B, N, 4, 3)
            batched_e = extents_list_out[j:end_idx] # (B, N, 4, 2)
            batched_p = pos_list_out[j:end_idx]     # (B, 3, N)
            batched_l = label_list_out[j:end_idx]    # (B, N)
            curr_st = st_tensor[j : end_idx].view(-1, 1) # (B, 1)
            curr_ed = ed_tensor[j : end_idx].view(-1, 1) # (B, 1)
            curr_level = level_tensor[j:end_idx] # (B,)
            batched_padding_mask = padding_mask_tensor[j:end_idx] # (B, N)
            preds = self(batched_d, batched_e, batched_p, batched_padding_mask) # (B, N, 256)

            mask = (batched_l != -1) # B, N
            raw_loss = F.cross_entropy(
                preds.view(-1, self.model_cfg.num_class), 
                batched_l.view(-1), 
                reduction='none', 
                ignore_index=-1
            ) / math.log(2)
            raw_loss = raw_loss.view(batched_l.shape) # (B, L)

            seq_len = batched_l.size(1)
            indices = torch.arange(seq_len, device=self.device) # L,
            even_mask = (indices % 2 == 0).unsqueeze(0) & mask # (B, L)
            odd_mask = (indices % 2 != 0).unsqueeze(0) & mask  # (B, L)

            total_metrics[0] += raw_loss[mask].sum()
            total_metrics[1] += mask.sum()
            
            total_metrics[2] += raw_loss[even_mask].sum()
            total_metrics[3] += even_mask.sum()
            
            total_metrics[4] += raw_loss[odd_mask].sum()
            total_metrics[5] += odd_mask.sum()

            range_mask = (indices.unsqueeze(0) >= curr_st) & (indices.unsqueeze(0) < curr_ed) # (B, L)
            bits_count_tensor += self.batch_compute_bits(preds, batched_l, range_mask)

            self.level_metrics_recorder.update(curr_level, preds, batched_l, range_mask)

        res = total_metrics.cpu().numpy()

        self.log("val_loss", res[0] / max(res[1], 1), on_epoch=True, batch_size=1)
        self.log("val_even_loss", res[2] / max(res[3], 1), on_epoch=True, batch_size=1)
        self.log("val_odd_loss", res[4] / max(res[5], 1), on_epoch=True, batch_size=1)

        self.bpp_recorder.record(
            tree_depth=tree_depth, 
            bits=bits_count_tensor.item() / 1000, 
            point_count=point_num / 1000
        )


    def batch_compute_bits(self, pred, gt, mask):
        probs = F.softmax(pred, dim=-1)
        safe_gt = gt.masked_fill(~mask, 0)
        target_probs = torch.gather(probs, dim=2, index=safe_gt.unsqueeze(-1)).squeeze(-1)
        log_probs = -torch.log2(target_probs.clamp(min=1e-9))
        return (log_probs * mask).sum()

    def on_validation_epoch_end(self):
        bpp_list = self.bpp_recorder.get_bpp_list_range(self.draw_level_start, self.draw_level_end)
        d1_fig, d2_fig, cd_fig = draw_rd_curve(bpp_list, self.CD_list, self.D1_list, self.D2_list)
        log_dict: dict = {}
        log_dict["Validation/D1_Curve_Image"] = wandb.Image(d1_fig)
        log_dict["Validation/D2_Curve_Image"] = wandb.Image(d2_fig)
        log_dict["Validation/CD_Curve_Image"] = wandb.Image(cd_fig)
        
        for i, bpp_val in enumerate(bpp_list):
            log_dict[f"Validation/bpp_Level_{self.draw_level_start + i}"] = bpp_val

        for lvl in sorted(self.level_metrics_recorder.data.keys()):
            d = self.level_metrics_recorder.data[lvl]
            total = d['total'].item()
            if total == 0:
                continue

            acc = d['correct'].item() / total
            log_dict[f"Validation/Level_{lvl}/Accuracy"] = acc


            conf_mat = d['confusion'].cpu().numpy()  # (C, C)
            recalls = []
            for c in range(self.num_classes):
                tp = conf_mat[c, c]
                fn = conf_mat[c, :].sum() - tp
                if tp + fn > 0:
                    recalls.append(tp / (tp + fn))
            if recalls:
                log_dict[f"Validation/Level_{lvl}/Macro_Recall"] = np.mean(recalls)

            # ECE (Expected Calibration Error)
            ece_count = d['ece_count'].cpu().numpy()
            ece_acc   = d['ece_acc'].cpu().numpy()
            ece_conf  = d['ece_conf'].cpu().numpy()
            total_bin = ece_count.sum()
            if total_bin > 0:
                bin_acc  = np.divide(ece_acc, ece_count, out=np.zeros_like(ece_acc), where=ece_count > 0)
                bin_conf = np.divide(ece_conf, ece_count, out=np.zeros_like(ece_conf), where=ece_count > 0)
                ece = np.sum(np.abs(bin_acc - bin_conf) * ece_count) / total_bin
                log_dict[f"Validation/Level_{lvl}/ECE"] = ece

        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.log(log_dict)  # type: ignore[union-attr]

        self.log("val_bpp", bpp_list[-1], on_epoch=True, prog_bar=True)
        plt.close('all')
        self.bpp_recorder.clear()
        self.level_metrics_recorder.reset()   


    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.') and not k.startswith('model._orig_mod.'):
                new_k = k.replace('model.', 'model._orig_mod.', 1)
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model._orig_mod.'):
                new_k = k.replace('model._orig_mod.', 'model.', 1)
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict