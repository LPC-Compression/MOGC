import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def get_graph_feature_optimized(x, k=1, idx=None):  # x: B, C, N
    center = torch.mean(x, dim=2, keepdim=True)
    
    relative_feature = x - center
    
    feature = torch.cat((relative_feature, x), dim=1)

    return feature.unsqueeze(-1).contiguous()


class GeoFeatGenerator_v3(nn.Module):
    def __init__(self, num_near_neighbor=10, max_level=17, out_channels:int=256,
                 embed_dims={"occ":16, "level":4, "octant":4, "extent":16},
                 attr_counts={"occ":3, "level":4, "octant":4, "extent":1},
                 pos_channels=[64, 128, 256],
                 mlp_attr2_channels=[80, 64, 64],
                 mlp_attr3_channels=[128, 128, 128],
                 edge_mlp1_channels=[256, 256, 256],
                 edge_mlp2_channels=[256, 256, 128]):
        
        super(GeoFeatGenerator_v3, self).__init__()
        self.k = num_near_neighbor
        self.max_level = max_level

        occ_d, level_d, octant_d, extent_d = embed_dims["occ"], embed_dims["level"], \
                                                embed_dims["octant"], embed_dims["extent"]
        self.occ_enc = nn.Embedding(257, occ_d)
        self.level_enc = nn.Embedding(max_level, level_d)
        self.octant_enc = nn.Embedding(9, octant_d)
        self.extent_enc = nn.Linear(2, extent_d)
        self.attr_dim = (occ_d * attr_counts["occ"] + 
                         level_d * attr_counts["level"] + 
                         octant_d * attr_counts["octant"] + 
                         extent_d * attr_counts["extent"]) 

        self.conv1 = self._make_conv_block(6, pos_channels[0])

        self.conv2 = self._make_conv_block((pos_channels[0] + self.attr_dim) * 2, pos_channels[1])

        self.conv3 = self._make_conv_block((pos_channels[1] + mlp_attr2_channels[-1]) * 2, pos_channels[2])

        self.mlp_attr2 = self._make_mlp(self.attr_dim, mlp_attr2_channels)
        self.mlp_attr3 = self._make_mlp(mlp_attr2_channels[-1], mlp_attr3_channels)

        edge_mlp1_in = sum(pos_channels)
        self.edge_mlp1 = self._make_mlp(edge_mlp1_in, edge_mlp1_channels)
        # edge_mlp2_in: pos3 + edge_mlp1_out
        edge_mlp2_in = pos_channels[-1] + edge_mlp1_channels[-1]
        self.edge_mlp2 = self._make_mlp(edge_mlp2_in, edge_mlp2_channels)

        self.out_mlp = nn.Linear(mlp_attr3_channels[-1] + edge_mlp2_channels[-1], out_channels)
    
    def _make_conv_block(self, in_ch, out_ch): 
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(num_groups=32, num_channels=out_ch),
            nn.LeakyReLU(0.2)
        )

    def _make_mlp(self, in_ch, channels):
        layers = []
        current_in = in_ch
        for i, out_ch in enumerate(channels):
            layers.append(nn.Linear(current_in, out_ch))
            if i < len(channels) - 1:
                layers.append(nn.LeakyReLU())      
            current_in = out_ch 
        return nn.Sequential(*layers)

    def embed_attributes(self, x, extent): 
        occ = x[:, :, 2::3]
        level = x[:, :, ::3]
        octant = x[:, :, 1::3]
        occ_embed = self.occ_enc(occ).reshape(x.shape[0], x.shape[1], -1)
        level_embed = self.level_enc(level).reshape(x.shape[0], x.shape[1], -1)
        octant_embed = self.octant_enc(octant).reshape(x.shape[0], x.shape[1], -1)
        extent_embed = self.extent_enc(extent).reshape(x.shape[0], x.shape[1], -1)
        return torch.concat((occ_embed, level_embed, octant_embed, extent_embed), 2)

    def forward(self, x, extent, pos):# x: (bsz, num_points, 15) extent: (bsz, num_points, 2) pos: (bsz, 3, num_points)
        attr_embed = self.embed_attributes(x, extent) # (bsz, num_points, attr_dim)

        k = min(self.k, pos.shape[2])
        pos_feat = get_graph_feature_optimized(pos, k=k) # (bsz, 6, num_points, k)
        pos_feat = self.conv1(pos_feat) # (bsz, 64, num_points, k)
        pos1 = pos_feat.max(dim=-1)[0] # (bsz, 64, num_points)

        combined_feat2 = torch.concat((pos1, attr_embed.transpose(1, 2)), dim=1) # (bsz, 64+attr_dim, num_points)
        pos_feat2 = self.conv2(get_graph_feature_optimized(combined_feat2.squeeze(-1), k=k)) # (bsz, 128, num_points, k)
        pos2 = pos_feat2.max(dim=-1)[0] # (bsz, 128, num_points)
        attr_embed2 = self.mlp_attr2(attr_embed) # (bsz, num_points, 64)

        combined_feat3 = torch.concat((pos2, attr_embed2.transpose(1, 2)), dim=1) # (bsz, 128+64, num_points)
        pos_feat3 = self.conv3(get_graph_feature_optimized(combined_feat3.squeeze(-1), k=k)) # (bsz, 256, num_points, k)
        pos3 = pos_feat3.max(dim=-1)[0] # (bsz, 256, num_points)
        attr_embed3 = self.mlp_attr3(attr_embed2) # (bsz, num_points, 128)

        pos_combined = torch.concat((pos1, pos2, pos3), dim=1).transpose(1, 2) # (bsz, num_points, 448)
        edge_feat = self.edge_mlp1(pos_combined) # (bsz, num_points, 256)
        edge_feat = self.edge_mlp2(torch.concat((pos3.transpose(1, 2), edge_feat), dim=2)) # (bsz, num_points, 128)
        
        output = self.out_mlp(torch.concat((attr_embed3, edge_feat), dim=2)) # (bsz, num_points, out_channels)

        return output 

    def embed_occ(self, occ):
        return self.occ_enc(occ)
