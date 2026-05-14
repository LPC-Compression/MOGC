
import torch
from torch import nn

from ..modules.nn import DeepNestedResidualBlock, \
                            _make_mlp, _concat_states_optimized
from ..modules.dgcnn import GeoFeatGenerator_v3 
from ..Attention_module.rotate_order_swin import RotateOrderSwinEncoder
from utils.node_order import normalized_int_coords, get_code, get_sort_restore_indices

from . import PROB_MODEL_REGISTRY

def sort_use_indices_2dim(features:torch.Tensor, indices:torch.Tensor):
    assert features.dim() == 2
    assert indices.dim() == 2
    sorted_feat = torch.gather(features, dim=1, index=indices)
    
    return sorted_feat

def sort_use_indices_3dim(features:torch.Tensor, indices:torch.Tensor):
    assert features.dim() == 3, indices.dim() == 2
    sorted_feat = torch.gather(features, 1, indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    return sorted_feat

def sort_use_indices_4dim(features:torch.Tensor, indices:torch.Tensor):
    assert features.dim() == 4
    assert indices.dim() == 2
    expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).expand_as(features)
    
    sorted_feat = torch.gather(features, dim=1, index=expanded_indices)
    
    return sorted_feat

class PredictHead(nn.Module):
    def __init__(self, num_class, hidden_dims, occ_embedd_dim, context_size, cfg, shuffle_order):
        super(PredictHead, self).__init__()
        self.num_class = num_class

        self.prob_pred_mlp1 = _make_mlp(hidden_dims, cfg.prob_pred_mlp1_channels+ [num_class])

        self.context_conv_block_a1 = DeepNestedResidualBlock(channels=hidden_dims, 
                                        seq_len=context_size//2, kernel_size=cfg.res_block.kernel_size, 
                                        num_conv_layers=cfg.res_block.num_conv_layers)
        self.pre_occ_mlp = _make_mlp(occ_embedd_dim, cfg.pre_occ_mlp_channels)
        self.pre_attn_mlp = _make_mlp(hidden_dims, cfg.pre_attn_mlp_channels)
        self.fusion_mlp = _make_mlp(cfg.pre_occ_mlp_channels[-1] + cfg.pre_attn_mlp_channels[-1], [hidden_dims])
        self.fusion_norm = nn.LayerNorm(hidden_dims)

        cross_swin_cfg = cfg.cross_swin_encoder
        self.swin_cross_transformer = RotateOrderSwinEncoder(
            dim = hidden_dims, window_size=cross_swin_cfg.window_size, 
            num_heads_list=cross_swin_cfg.num_heads, depth_list=cross_swin_cfg.depths,
            is_cross_attention=False, sequence_size=context_size,
            order_list = cross_swin_cfg.order_list,
            shuffle_order = shuffle_order,
            hidden_dropout_prob=0.1
        )

        self.prob_pred_mlp2 = _make_mlp(hidden_dims * len(cross_swin_cfg.depths) + hidden_dims, cfg.prob_pred_mlp2_channels+[num_class])
    
    def forward(self, feat_a1, feat_a2, pre_occ_embed, codes):
        # branch 1
        prob1 = self.prob_pred_mlp1(feat_a1) # B, N//2, 256
        # branch 2
        feat_a1_contextual = self.context_conv_block_a1(feat_a1.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        pre_occ_feat = self.pre_occ_mlp(pre_occ_embed) # B, N//2, 16
        pre_attn_feat = self.pre_attn_mlp(feat_a1_contextual) # B, N//2, 240
        combined_feat = torch.concat((pre_occ_feat, pre_attn_feat), dim=2) # B, N//2, 256
        pre_feat = self.fusion_mlp(combined_feat) # B, N//2, 256
        pre_feat = self.fusion_norm(pre_feat)

        B, N_half, dim = pre_feat.shape

        combined_seq = torch.empty((B, N_half * 2, dim), device=pre_feat.device, dtype=pre_feat.dtype)
        combined_seq[:, 0::2, :] = pre_feat 
        combined_seq[:, 1::2, :] = feat_a2  

        cross_output = self.swin_cross_transformer(combined_seq, N_half*2, 
                            hidden_states_codes=codes,
                            query_codes=codes, output_hidden_states_before_downsampling=True)
        all_hidden_states = _concat_states_optimized(cross_output["hidden_states"])

        odd_features = all_hidden_states[:, 1::2, :]
        feat_a2 = torch.concat((odd_features, feat_a2), 2)
        prob2 = self.prob_pred_mlp2(feat_a2) # B, N//2, num_class

        B, N_half, dim = prob1.shape
        probs = torch.stack([prob1, prob2], dim=2).reshape((B, N_half*2, dim))
        return probs

class Preprocessor:
    def __init__(self,):
        self.order_names = ["z", "z-trans", "hilbert", "hilbert-trans"]
        self.depth=15

    def __call__(self, data, extents, pos, padding_mask):
        # data: B, context_size, 4, 2
        # extents: B, context_size, 2
        # pos: B, 3, context_size
        # padding_mask: B, context_size
        point = pos.permute(0, 2, 1).contiguous() # (B, context_size, 3)
        int_coords = normalized_int_coords(point, self.depth)

        codes_unsorted = {
            order_name: get_code(int_coords, order_name, self.depth, padding_mask)
            for order_name in self.order_names
        }

        hilbert_code = codes_unsorted["hilbert"]
        sorted_indices, restore_indices = get_sort_restore_indices(hilbert_code)
        data = sort_use_indices_4dim(data, sorted_indices)
        extents = sort_use_indices_3dim(extents, sorted_indices)
        pos = sort_use_indices_3dim(point, sorted_indices).permute(0, 2, 1).contiguous() # (B, 3, context_size)
        padding_mask = sort_use_indices_2dim(padding_mask, sorted_indices)

        pre_occ = data[:, ::2, -1, -1]  
        codes = {
            name: sort_use_indices_2dim(code, sorted_indices)
            for name, code in codes_unsorted.items()
        }

        return data, extents, pos, padding_mask, restore_indices, pre_occ, codes

@PROB_MODEL_REGISTRY.register("SelfOrderProbModel")
class SelfOrderProbModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.preprocessor = Preprocessor()
        self.model_cfg = model_cfg

        geo_cfg = model_cfg.geo_feat_generator
        self.geo_feat_generator = GeoFeatGenerator_v3(max_level=geo_cfg.max_level, embed_dims=geo_cfg.embed_dims, \
                                                      pos_channels=geo_cfg.pos_channels, mlp_attr2_channels=geo_cfg.mlp_attr2_channels, \
                                                        mlp_attr3_channels=geo_cfg.mlp_attr3_channels, edge_mlp1_channels=geo_cfg.edge_mlp1_channels, \
                                                            edge_mlp2_channels=geo_cfg.edge_mlp2_channels, out_channels=model_cfg.self_swin_encoder.num_channels)
        self_swin_cfg = model_cfg.self_swin_encoder
        self.swin_encoder = RotateOrderSwinEncoder(
            dim = self_swin_cfg.num_channels, window_size=self_swin_cfg.window_size, 
            num_heads_list=self_swin_cfg.num_heads, depth_list=self_swin_cfg.depths, 
            is_cross_attention=False, sequence_size=model_cfg.context_size,
            order_list=self_swin_cfg.order_list,
            shuffle_order = model_cfg.shuffle_order,
            hidden_dropout_prob=0.1
        )
        ancient_mlp_channels = model_cfg.ancient_mlp_channels
        self.ancient_mlp = _make_mlp(len(self_swin_cfg.depths) * self_swin_cfg.num_channels, ancient_mlp_channels)

        predict_head_cfg = model_cfg.predict_head
        self.predicthead = PredictHead(num_class=model_cfg.num_class, hidden_dims=model_cfg.ancient_mlp_channels[-1], \
                                       occ_embedd_dim=geo_cfg.embed_dims.occ, context_size=model_cfg.context_size, \
                                        cfg=predict_head_cfg, shuffle_order = model_cfg.shuffle_order)

    def forward_features(self, data, extents, pos, padding_mask):
        data, extents, pos, padding_mask, restore_indices, pre_occ, codes = self.preprocessor(data, extents, pos, padding_mask) # B, context_size//2
        B, context_size, _, _ = data.shape
        data = data.reshape(B, context_size, -1)[:, :, :-1]
        feat = self.geo_feat_generator(data, extents, pos) # (bsz, num_points, 256)
    
        self_output = self.swin_encoder(feat, self.model_cfg.context_size,\
                                hidden_states_codes=codes, query_codes=codes,
                                output_hidden_states_before_downsampling=True)
        self_output = _concat_states_optimized(self_output["hidden_states"]) # B, context_size, 256 * 5
        feat_a = self.ancient_mlp(self_output) # B, context_size, 256
        feat_a1 = feat_a[:, ::2]
        feat_a2 = feat_a[:, 1::2]
        pre_occ_embed = self.geo_feat_generator.embed_occ(pre_occ)
        return feat_a1, feat_a2, pre_occ_embed, pre_occ, codes, restore_indices

    def forward(self, data, extents, pos, padding_mask):
        feat_a1, feat_a2, pre_occ_embed, pre_occ, codes, restore_indices = self.forward_features(data, extents, pos, padding_mask)
        prob = self.predicthead(feat_a1, feat_a2, pre_occ_embed, codes) # B, context_size, 256
        prob = sort_use_indices_3dim(prob, restore_indices)
        return prob