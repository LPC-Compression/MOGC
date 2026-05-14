

import torch
from torch import nn
import torch.nn.functional as F


class DeepNestedResidualBlock(nn.Module):
    def __init__(self, channels=256, seq_len=4096, kernel_size=3, num_conv_layers=8):
        super().__init__()

        self.num_groups = num_conv_layers // 2

        self.conv_groups = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size,
                          padding=kernel_size//2, padding_mode='replicate'),
                nn.GELU(),
                nn.LayerNorm([channels, seq_len]),
                nn.Conv1d(channels, channels, kernel_size=kernel_size,
                          padding=kernel_size//2, padding_mode='replicate'),
                nn.LayerNorm([channels, seq_len])
            ) for _ in range(self.num_groups)
        ])
        
        self.activation = nn.GELU()
        
    def forward(self, x):
        original = x  
        for group in self.conv_groups:
            residual = x
            x = group(x)
            x = x + residual
            x = self.activation(x)
        
        return self.activation(x)


def _make_mlp(in_ch, mlp_channels):
        layers = []
        for out_ch in mlp_channels:
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        return nn.Sequential(*layers)

def upsample2len(x, target_len):
    # x: (B, n, C)
    # target_len: int
    assert target_len % x.shape[1] == 0
    if x.shape[1] == target_len:
        return x
    scale = target_len // x.shape[1]
    return x.repeat_interleave(scale, dim=1)

def _concat_states_optimized(hidden_states):
    bsz, target_len, dim = hidden_states[1].shape 
    selected = hidden_states[1:] 
    total_dim = dim * len(selected)
    out = torch.empty(
        (bsz, target_len, total_dim),
        device=hidden_states[1].device,
        dtype=hidden_states[1].dtype,
    )
    offset = 0
    for state in selected:
        state = upsample2len(state, target_len)
        out[:, :, offset:offset + dim] = state
        offset += dim
    return out
