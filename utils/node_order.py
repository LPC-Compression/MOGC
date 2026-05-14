import torch
from serialization.default import encode

@torch.no_grad()
def normalized_int_coords(coords:torch.Tensor, depth:int)->torch.LongTensor:
    assert coords.dim() == 3 and coords.shape[2] == 3, f"{coords.shape}。"
    B, N, _ = coords.shape
    device = coords.device
    min_vals, _ = torch.min(coords, dim=1, keepdim=True)  # (B, 1, 3)
    max_vals, _ = torch.max(coords, dim=1, keepdim=True)  # (B, 1, 3)
    scale = max_vals - min_vals
    scale[scale == 0] = 1.0 
    norm_coords = (coords - min_vals) / scale 
    max_coord_val = 2**depth - 1
    int_coords = (norm_coords * max_coord_val).long() 
    return int_coords

@torch.no_grad()
@torch.compiler.disable
def get_code(int_coords:torch.Tensor, order:str, depth:int, padding_mask:torch.Tensor):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    code = encode(int_coords.contiguous(), depth=depth, order=order)
    valid_mask = padding_mask.bool()
    max_code_per_batch = code.max(dim=1, keepdim=True)[0] # B, 1
    noise = torch.arange(int_coords.shape[1], device=int_coords.device).unsqueeze(0) \
                + max_code_per_batch + 1 # (1, N) + (B, 1) = B, N
    code = torch.where(valid_mask, code, noise)
    return code

@torch.no_grad()
def get_sort_restore_indices(code:torch.Tensor):
    B, N = code.shape
    device = code.device
    sorted_code, sorted_indices = torch.sort(code, dim=1)
    restore_indices = torch.empty_like(sorted_indices)
    arange = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    restore_indices.scatter_(1, sorted_indices, arange)
    return sorted_indices, restore_indices

def sort_use_indices(features:torch.Tensor, indices:torch.Tensor):
    assert features.dim() == 3, indices.dim() == 2
    sorted_feat = torch.gather(features, 1, indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    return sorted_feat

@torch.no_grad()
def downsample_code(code:torch.Tensor):
    code_even = code[:, ::2]
    code_odd = code[:, 1::2]
    return (code_even + code_odd)//2
