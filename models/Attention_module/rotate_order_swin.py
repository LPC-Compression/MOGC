
from dataclasses import dataclass
from typing import Optional, Tuple, Type
import random

import torch
from torch import nn
from torch.nn import functional as F

from utils.node_order import get_sort_restore_indices, sort_use_indices, downsample_code

OPEN_BIAS = False

import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MonaOp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=OPEN_BIAS)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=OPEN_BIAS)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=OPEN_BIAS)
        self.projector = nn.Conv1d(channels, channels, kernel_size=1, bias=OPEN_BIAS)

    def forward(self, x): # x: B,C,N
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x

class MSCA(nn.Module):
    def __init__(self, channels, hidden_channels=64, dropout_p=0.1):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.project1 = nn.Linear(channels, hidden_channels, bias=OPEN_BIAS)
        self.project2 = nn.Linear(hidden_channels, channels, bias=OPEN_BIAS)
        self.adapter_conv = MonaOp(hidden_channels) 
        self.dropout = nn.Dropout(p=dropout_p)
        self.ls = nn.Parameter(torch.ones(channels) * 1e-5)

    def forward(self, x): # x: B,N,C
        identity = x
        x = self.norm(x)
        project1 = self.project1(x) 
        project1_permuted = project1.permute(0, 2, 1).contiguous()      
        project1_conv = self.adapter_conv(project1_permuted)
        project1_restored = project1_conv.permute(0, 2, 1).contiguous()
        nonlinear = F.gelu(project1_restored)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        return identity + self.ls * project2
    

# 窗口划分操作
def window_partition(x, window_size): 
    B, N, C = x.shape
    x = x.reshape(B, N // window_size, window_size, C)
    windows = x.reshape(-1, window_size, C)
    return windows

def window_reverse(windows, sequence_size):
    Bn, window_size, C = windows.shape
    num_windows = sequence_size // window_size
    x = windows.reshape(Bn//num_windows, num_windows, window_size, windows.shape[-1])
    x = x.reshape(Bn//num_windows, -1, windows.shape[-1])
    return x

class SwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """
    def __init__(self, sequence_size: int, dim: int, norm_layer:Type[nn.Module] = RMSNorm) -> None:
        super().__init__()
        assert sequence_size % 2 == 0
        self.sequence_size = sequence_size
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, dim, bias=OPEN_BIAS)
        self.norm = norm_layer(2 * dim)
    """
    input_feature: B,N,C, input_size: int -> output_feature: B,N/2,C
    """
    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        assert self.sequence_size == input_feature.shape[1]
        batch_size, _, num_channels = input_feature.shape
        input_feature_0 = input_feature[:, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, :]
        input_feature = torch.cat([input_feature_0, input_feature_1], -1)
        input_feature = input_feature.reshape(batch_size, -1, 2 * num_channels)
        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)
        return input_feature


def precompute_freqs_cos_sin(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    
    return freqs_cos, freqs_sin

def apply_rotary_emb_real(xq, xk, freqs_cos, freqs_sin):
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_0, xq_1 = xq_reshaped[..., 0], xq_reshaped[..., 1] # x, y
    xk_0, xk_1 = xk_reshaped[..., 0], xk_reshaped[..., 1] 
    
    freqs_cos = freqs_cos.view(1, 1, freqs_cos.shape[0], freqs_cos.shape[1]).to(xq.device)
    freqs_sin = freqs_sin.view(1, 1, freqs_sin.shape[0], freqs_sin.shape[1]).to(xq.device)
    
    xq_out_0 = xq_0 * freqs_cos - xq_1 * freqs_sin
    xk_out_0 = xk_0 * freqs_cos - xk_1 * freqs_sin
    
    xq_out_1 = xq_0 * freqs_sin + xq_1 * freqs_cos
    xk_out_1 = xk_0 * freqs_sin + xk_1 * freqs_cos
    
    xq_out = torch.stack([xq_out_0, xq_out_1], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_0, xk_out_1], dim=-1).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(
        self, dim, num_heads, window_size,
        qkv_bias: bool = False,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=OPEN_BIAS)
        self.k = nn.Linear(dim, dim, bias=OPEN_BIAS)
        self.v = nn.Linear(dim, dim, bias=OPEN_BIAS)
        self.attention_dropout = attention_dropout

        freqs_cos, freqs_sin = precompute_freqs_cos_sin(self.head_dim, self.window_size)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.reshape(new_x_shape) # B, N, num_heads, head_dim
        return x.permute(0, 2, 1, 3).contiguous() # B, num_heads, N, head_dim

    def forward(
        self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor,
        q_bias:Optional[torch.Tensor]=None, k_bias:Optional[torch.Tensor]=None, v_bias:Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        )-> dict:
        q, k, v = self.q(q), self.k(k), self.v(v)
        q = q + q_bias if q_bias is not None else q
        k = k + k_bias if k_bias is not None else k
        v = v + v_bias if v_bias is not None else v
        B_n, window_size, dims = q.shape
        # B*num_windows, num_heads, window_size, head_dim
        q, k, v = self.transpose_for_scores(q), self.transpose_for_scores(k), self.transpose_for_scores(v) 

        q, k = apply_rotary_emb_real(q, k, self.freqs_cos, self.freqs_sin)
        # ==========================

        attn_mask_processed = None
        if attention_mask is not None:
            num_windows = attention_mask.shape[0]               
            actual_batch_size = B_n // num_windows
            attn_mask_processed = attention_mask.unsqueeze(1).unsqueeze(0)
            attn_mask_processed = attn_mask_processed.expand(actual_batch_size, -1, self.num_heads, -1, -1)
            attn_mask_processed = attn_mask_processed.reshape((B_n, self.num_heads, window_size, window_size))

        
        context = F.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=attn_mask_processed,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False
                ) # (B*num_windows, num_heads, window_size, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().reshape(B_n, window_size, self.dim) # (B*num_windows, window_size, dim)
        return {
            "context": context
        }

class AttentionWithPenalty(nn.Module):
    def __init__(self, dim, num_heads, window_size, is_cross_attention:bool, conv_kernel_size:int=3,
                 qkv_bias: bool = True, attention_dropout: float = 0.0, hidden_dropout_prob:float=0.0):
        super().__init__()
        self.window_size = window_size
        self.is_cross_attention = is_cross_attention
        self.attention = Attention(dim=dim, num_heads=num_heads, window_size=window_size, 
                                   qkv_bias=qkv_bias, attention_dropout=attention_dropout,)

        self.q_conv = nn.Conv1d(dim, dim, 
                                kernel_size=conv_kernel_size, padding='same', groups=dim, bias=OPEN_BIAS)
        self.k_conv = nn.Conv1d(dim, dim, 
                                kernel_size=conv_kernel_size, padding='same', groups=dim, bias=OPEN_BIAS)
        self.v_conv = nn.Conv1d(dim, dim, 
                                kernel_size=conv_kernel_size, padding='same', groups=dim, bias=OPEN_BIAS)
        
        self.output_dense_proj = nn.Linear(dim, dim, bias=OPEN_BIAS)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, 
                attention_mask: Optional[torch.FloatTensor] = None,
                )-> dict:
        k_bias, v_bias = self.k_conv(k.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous(), self.v_conv(v.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if self.is_cross_attention:
            q_bias = None
        else:
            q_bias = self.q_conv(q.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        atten_result = self.attention(q, k, v, q_bias, k_bias, v_bias, attention_mask)
        
        context_out = self.output_dense_proj(atten_result["context"])
        context_out = self.dropout(context_out)

        return {
            "context": context_out
        }
        
    
class SwiGLU(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 8.0/4.0, hidden_dropout_prob: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        out = self.w3(hidden)
        return self.dropout(out)

# Swin 网络层
class SwinLayer(nn.Module):
    def __init__(
            self, dim, num_heads, window_size, is_cross_attention,
            sequence_size, shift_size, 
            hidden_dropout_prob=0.0,
            qkv_bias:bool=True, attention_dropout:float=0.0,
            layer_norm_eps:float=1e-5,
    ):
        super().__init__()
        self.window_size = window_size
        self.sequence_size = sequence_size
        self.shift_size = shift_size
        self.is_cross_attention = is_cross_attention

        assert shift_size < window_size and sequence_size % 2 == 0

        self.attention = AttentionWithPenalty(dim, num_heads, window_size, is_cross_attention,
                                              qkv_bias=qkv_bias, attention_dropout=attention_dropout, 
                                              hidden_dropout_prob=hidden_dropout_prob)
        self.norm1 = RMSNorm(dim, eps=layer_norm_eps)
        self.norm2 = RMSNorm(dim, eps=layer_norm_eps)

        self.mona1 = MSCA(channels=dim)
        self.mona2 = MSCA(channels=dim)

        self.swiglu = SwiGLU(dim, mlp_ratio=8.0/3.0, hidden_dropout_prob=hidden_dropout_prob)

        self.ls1 = nn.Parameter(torch.ones(dim) * 1e-5)
        self.ls2 = nn.Parameter(torch.ones(dim) * 1e-5)

    def window_partition(self, sequence_data:torch.Tensor):
        if self.shift_size > 0:
            sequence_data = torch.roll(sequence_data, shifts=(-self.shift_size,), dims=(1,))
        windows = window_partition(sequence_data, self.window_size) # B*num_windows, window_size, dim
        return windows
    
    def window_reverse(self, window_data:torch.Tensor, sequence_size): # B*num_windows, window_size, dim
        sequence_data = window_reverse(window_data, sequence_size) # B, N, dim
        if self.shift_size > 0:
            sequence_data = torch.roll(sequence_data, shifts=(self.shift_size,), dims=(1,)) # B, N, dim
        return sequence_data

    def get_attn_mask(self, sequence_size, dtype, device):
        if self.shift_size > 0:
            mask = torch.zeros(sequence_size, dtype=dtype, device=device) # N
            mask[-self.window_size : -self.shift_size] = 1
            mask[-self.shift_size : ] = 2 # N
            mask_windows = window_partition(mask.reshape(1, sequence_size, 1), self.window_size).squeeze(-1)  # num_windows, window_size
            attn_mask = (mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)).to(dtype) * -100.0# num_windows, window_size, window_size
        else:
            attn_mask = None
        return attn_mask

    def forward(
        self, 
        query:torch.Tensor, hidden_states:torch.Tensor
    ):
        B, sequence_size, dim = hidden_states.shape

        assert not self.is_cross_attention or query is not None, "交叉注意力下 query不能为None"
        assert sequence_size % self.window_size == 0
        assert sequence_size == self.sequence_size and query.shape[1] == sequence_size

        attn_mask = self.get_attn_mask(sequence_size, hidden_states.dtype, hidden_states.device)
        x = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.is_cross_attention:
            query_windows = self.window_partition(query) # type: ignore
            hidden_states_windows = self.window_partition(hidden_states)
        else:
            hidden_states_windows = self.window_partition(hidden_states)
            query_windows = hidden_states_windows # B*num_windows, window_size, dim
        atten_out = self.attention(
            q=query_windows, k=hidden_states_windows, v=hidden_states_windows,
            attention_mask=attn_mask,)
        
        x = x + self.ls1 * self.window_reverse(atten_out["context"], sequence_size) # B, N, dim
        x = self.mona1(x)
        x = x + self.ls2 * self.swiglu(self.norm2(x))
        x = self.mona2(x)
        return {
            "context": x
        }
    
class SwinStage(nn.Module):
    def __init__(
        self, dim, num_heads, window_size, 
        is_cross_attention, sequence_size,
        depth, 
        order:list,
        attention_dropout: float = 0.0,
        downsample:bool=False,
        shuffle_order = False,
        hidden_dropout_prob:float=0.0
    ):
        super().__init__()
        self.sequence_size = sequence_size
        self.is_cross_attention = is_cross_attention
        self.order = order
        self.shuffle_order = shuffle_order

        self.swin_layers = nn.ModuleList(
            [
                SwinLayer(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    is_cross_attention=is_cross_attention, sequence_size=sequence_size,
                    shift_size=0,
                    attention_dropout=attention_dropout,
                    hidden_dropout_prob=hidden_dropout_prob
                    # shift_size=0 if (i%2==0 and sequence_size > window_size) else window_size//2
                ) for i in range(depth)
            ]
        )
        self.down, self.down_hidden = None, None
        if downsample:
            self.down_hidden = SwinPatchMerging(sequence_size=sequence_size, dim=dim)
            if is_cross_attention:
                self.down_query = SwinPatchMerging(sequence_size=sequence_size, dim=dim)
        
    def forward(
        self, query:torch.Tensor, hidden_states:torch.Tensor,
        hidden_states_codes:dict, query_codes:dict 
    ):
        assert query.shape[1] == self.sequence_size
        assert len(self.swin_layers) > 0, "SwinStage must have at least one layer."
        hidden_states_sort_indexs = {order_name: get_sort_restore_indices(code) \
                                     for order_name,code in hidden_states_codes.items()}
        if self.is_cross_attention:
            query_sort_indexs = {order_name: get_sort_restore_indices(code) \
                                     for order_name,code in query_codes.items()}
        else:
            query_sort_indexs = {}

        layer_output = None
        order = self.order.copy()
        if self.shuffle_order:
            random.shuffle(order)
        for i, layer_module in enumerate(self.swin_layers):
            # 1. reorder
            current_order = order[i]
            hidden_states = sort_use_indices(hidden_states, hidden_states_sort_indexs[current_order][0])
            if not self.is_cross_attention:
                query = hidden_states
            else:
                query = sort_use_indices(query, query_sort_indexs[current_order][0])
            # process
            layer_output = layer_module(query, hidden_states)
            hidden_states = layer_output["context"]

            # restore
            hidden_states = sort_use_indices(hidden_states, hidden_states_sort_indexs[current_order][1])
            if not self.is_cross_attention:
                query = hidden_states
            else:
                query = sort_use_indices(query, query_sort_indexs[current_order][1])
        states_before_down = hidden_states

        if self.down_hidden is not None:
            hidden_states = self.down_hidden(hidden_states)
            hidden_states_codes = {order_name:downsample_code(code) for order_name,code in hidden_states_codes.items()}
            if self.is_cross_attention:
                assert self.down_query is not None
                query = self.down_query(query)
                query_codes = {order_name:downsample_code(code) for order_name,code in query_codes.items()}
            else:
                query = hidden_states
                query_codes = hidden_states_codes
        return {
            "query": query,
            "hidden_states": hidden_states,
            "states_before_down":states_before_down,
            "hidden_states_codes": hidden_states_codes,
            "query_codes": query_codes
        }




class RotateOrderSwinEncoder(nn.Module):
    _RESIDUAL_PROJ_KEYWORDS = ['output_dense_proj', 'w3', 'project2', 'reduction', 'projector']

    @staticmethod
    def _init_weights(module, name, num_layers):
        if any(kw in name for kw in RotateOrderSwinEncoder._RESIDUAL_PROJ_KEYWORDS):
            if hasattr(module, 'weight'):
                scaled_std = 0.02 / math.sqrt(2.0 * num_layers)
                nn.init.trunc_normal_(module.weight, std=scaled_std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(
        self, dim, window_size, 
        num_heads_list:list, depth_list:list, 
        is_cross_attention, sequence_size, order_list:list,
        attention_dropout: float = 0.0,
        shuffle_order:bool=False,
        hidden_dropout_prob:float=0.0
    ):
        super().__init__()
        assert len(depth_list) == len(num_heads_list)
        self.num_layers = len(depth_list)
        self.sequence_size = sequence_size
        self.is_cross_attention = is_cross_attention
        total_depth = sum(depth_list)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                SwinStage(
                    dim=dim, num_heads=num_heads_list[i], depth=depth_list[i],
                    window_size=window_size,
                    is_cross_attention=is_cross_attention, sequence_size=sequence_size,
                    order=order_list[i],
                    downsample=True if i < self.num_layers - 1 else False,
                    shuffle_order=shuffle_order,
                    attention_dropout=attention_dropout
                )
            )
            assert sequence_size % 2 == 0, "Sequence size 必须整除2"
            sequence_size = sequence_size // 2
        
        self.norm = RMSNorm(dim)

        for name, module in self.named_modules():
            RotateOrderSwinEncoder._init_weights(module, name, total_depth)
    
    def forward(
        self, hidden_states:torch.Tensor, sequence_size:int,
        hidden_states_codes:dict, query_codes:dict, 
        query:Optional[torch.Tensor]=None, 
        output_hidden_states_before_downsampling: Optional[bool] = True,
    ):
        assert sequence_size == self.sequence_size
        assert not self.is_cross_attention or query is not None
        if query is None:
            query = hidden_states
        hidden_states_list = [hidden_states]
        for i, layer_module in enumerate(self.layers):
            layer_output = layer_module(
                query, hidden_states, hidden_states_codes, query_codes
            )
            hidden_states_codes, query_codes = layer_output["hidden_states_codes"], layer_output["query_codes"]
            query = layer_output["query"]
            hidden_states = layer_output["hidden_states"]
            if output_hidden_states_before_downsampling:
                hidden_states_list.append(
                    layer_output["states_before_down"]
                )
            else:
                hidden_states_list.append(hidden_states) # type: ignore
        hidden_states = self.norm(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": hidden_states_list
        }



        



    
