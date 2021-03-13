# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

# from util.misc import inverse_sigmoid
from mmpose.models.utils.ms_deform_attn import MSDeformAttn
from mmcv.cnn import ConvModule



def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, with_box_refine=True, hidden_dim=256,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 decoder_layer_type="deformable", decoder_use_self_attn=True, decoder_share_query=False,
                 num_joints=17,
                 use_heatmap_loss=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels
        self.decoder_share_query = decoder_share_query
        self.num_joints = num_joints
        self.use_heatmap_loss = use_heatmap_loss

        if self.use_heatmap_loss:
            from mmcv.cnn import build_upsample_layer
            num_layers = 1
            num_kernels=[4]
            num_filters=[256]

            layers = []
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i])

                planes = num_filters[i]
                layers.append(
                    build_upsample_layer(
                        dict(type='deconv'),
                        in_channels=d_model,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False))
                layers.append(nn.BatchNorm2d(planes))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = planes

            self.deconv_layer = nn.Sequential(*layers)
            # import pdb
            # pdb.set_trace()

            self.final_layer = nn.Sequential(
            # ConvModule(
            #     d_model,
            #     d_model,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     norm_cfg=dict(type='BN'),
            #     act_cfg=dict(type='ReLU'),
            #     inplace=True),
                ConvModule(
                    d_model,
                    num_joints,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False)
            )

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        if decoder_layer_type == "deformable":
            if decoder_use_self_attn:
                decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                                  dropout, activation,
                                                                  num_feature_levels, nhead, dec_n_points)
            else:
                decoder_layer = DeformableTransformerDecoderLayer_wo_self_attn(d_model, dim_feedforward,
                                                                  dropout, activation,
                                                                  num_feature_levels, nhead, dec_n_points)
        elif decoder_layer_type == "standard":
            decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points)
        else:
            raise NotImplementedError

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                    with_box_refine, hidden_dim)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if decoder_share_query:
                reference_points = [nn.Linear(d_model, 2)] * self.num_joints
                self.reference_points = nn.Sequential(*reference_points)
            else:
                self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()


    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            if self.decoder_share_query:
                for i in range(self.num_joints):
                    xavier_uniform_(self.reference_points[i].weight.data, gain=1.0)
                    constant_(self.reference_points[i].bias.data, 0.)
            else:
                xavier_uniform_(self.reference_points.weight.data, gain=1.0)
                constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

        if self.use_heatmap_loss:
            from mmcv.cnn import (build_upsample_layer, constant_init, normal_init)
            for _, m in self.deconv_layer.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001, bias=0)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            # mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            # mask_flatten.append(mask)
        # torch.Size([bs, 65, 256])
        src_flatten = torch.cat(src_flatten, 1)
        # torch.Size([bs, 65])
        # mask_flatten = torch.cat(mask_flatten, 1)
        # torch.Size([bs, 65, 256])
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # tensor([[8, 6], [4, 3], [2, 2], [1, 1]], device='cuda:0')
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # tensor([ 0, 48, 60, 64], device='cuda:0')
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # torch.Size([bs, 4, 2])
        # valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = torch.ones([src_flatten.size(0), self.num_feature_levels, 2],
                        dtype=torch.float, device=src_flatten.device)
        # encoder
        # torch.Size([bs, 65, 256])
        memory = self.encoder(src_flatten, spatial_shapes,
                              level_start_index, valid_ratios,
                              lvl_pos_embed_flatten)

        # memory = src_flatten
        # prepare input for decoder
        bs, _, c = memory.shape
        # torch.Size([17, 512]) -> torch.Size([17, 256]) and torch.Size([17, 256])
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        # torch.Size([17, 256]) -> torch.Size([bs, 17, 256])
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        # torch.Size([17, 256]) -> torch.Size([bs, 17, 256])
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        # torch.Size([bs, 17, 256]) -> torch.Size([bs, 17, 2])
        if self.decoder_share_query:
            hs_list, init_reference_out_list, inter_references_out_list = [], [], []
            for i in range(len(self.reference_points)):
                reference_points = self.reference_points[i](query_embed).sigmoid()
                init_reference_out = reference_points
                # hs_list.append(hs)
                init_reference_out_list.append(init_reference_out)
                # inter_references_out_list.append(inter_references)

            init_reference_out = torch.cat(init_reference_out_list, dim=1)
            # import pdb
            # pdb.set_trace()
            tgt = tgt.expand(-1, self.num_joints, -1)
            query_embed = query_embed.expand(-1, self.num_joints, -1)

            hs, inter_references = self.decoder(tgt, init_reference_out, memory,
                                                spatial_shapes, level_start_index, valid_ratios,
                                                query_embed)
            # hs = torch.cat(hs_list, dim=2)
            # inter_references_out = torch.cat(inter_references_out_list, dim=2)

        else:
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points # [1,17,2]

            # decoder
            # hs: [num_dec_layers, bs, 17, 2]
            # inter_references: [num_dec_layers, bs, 17, 2]
            hs, inter_references = self.decoder(tgt, reference_points, memory,
                                                spatial_shapes, level_start_index, valid_ratios,
                                                query_embed)
        inter_references_out = inter_references

        if self.use_heatmap_loss:
            # memory.resize(memory[:, :level_start_index[1], :].size(0), 256, spatial_shapes[0][0], spatial_shapes[0][1]
            # [bs, concat_feats, hidden_dim] -> [bs, hidden_dim, concat_feats] ->

            if self.num_feature_levels == 3:
                h = spatial_shapes[0][0]
                w = spatial_shapes[0][1]
                x = memory.permute(0, 2, 1)[:, :, :level_start_index[1]].contiguous().view(bs, c, h, w)
            elif self.num_feature_levels == 4:
                h = spatial_shapes[1][0]
                w = spatial_shapes[1][1]
                x = memory.permute(0, 2, 1)[:, :, level_start_index[1]:level_start_index[2]].contiguous().view(bs, c, h, w)
            else:
                raise NotImplementedError

            x = self.deconv_layer(x)

            if self.num_feature_levels == 4:
                # h = spatial_shapes[0][0]
                # w = spatial_shapes[0][1]
                # memory_s2 = memory.permute(0, 2, 1)[:, :, :level_start_index[1]].contiguous().view(bs, c, h, w)
                # out_heatmap = self.final_layer(x + memory_s2)
                out_heatmap = self.final_layer(x)
            else:
                out_heatmap = self.final_layer(x)

            return hs, init_reference_out, inter_references_out, out_heatmap

        return hs, init_reference_out, inter_references_out




class OneQueryDeformableTransformer(DeformableTransformer):
    def forward(self, srcs, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):

        for lvl, pos_embed in enumerate(pos_embeds):
            bs, c, h, w = pos_embed.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # src = src.flatten(2).transpose(1, 2)
            # mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            # src_flatten.append(src)
            # mask_flatten.append(mask)
        # torch.Size([bs, 65, 256])
        # src_flatten = torch.cat(src_flatten, 1)
        # torch.Size([bs, 65])
        # mask_flatten = torch.cat(mask_flatten, 1)
        # torch.Size([bs, 65, 256])
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # tensor([[8, 6], [4, 3], [2, 2], [1, 1]], device='cuda:0')
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=pos_embed.device)
        # tensor([ 0, 48, 60, 64], device='cuda:0')
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # torch.Size([bs, 4, 2])
        # import pdb
        # pdb.set_trace()
        # valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = torch.ones([srcs.size(0), self.num_feature_levels, 2],
                        dtype=torch.float, device=pos_embed.device)
        # encoder
        # torch.Size([bs, 65, 256])
        # memory = self.encoder(src_flatten, spatial_shapes,
        #                       level_start_index, valid_ratios,
        #                       lvl_pos_embed_flatten)

        memory = srcs
        # prepare input for decoder
        bs, _, c = memory.shape
        # torch.Size([17, 512]) -> torch.Size([17, 256]) and torch.Size([17, 256])
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        # torch.Size([17, 256]) -> torch.Size([17, 1, 256]) -> torch.Size([17 * bs, 1, 256])
        query_embed = query_embed.unsqueeze(1).expand(bs, -1, -1)
        # torch.Size([17, 256]) -> torch.Size([17 * bs, 1, 256])
        tgt = tgt.unsqueeze(1).expand(bs, -1, -1)
        # torch.Size([bs, 17, 256]) -> torch.Size([bs, 17, 2])
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        # decoder
        # hs: [num_dec_layers, 17*bs, 1, 2]
        # inter_references: [num_dec_layers, bs, 17, 2]
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios,
                                            query_embed)
        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out




class DeformableTransformerDecoderLayer_wo_self_attn(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


    def forward(self, tgt, query_pos, reference_points,
                src, src_spatial_shapes,
                level_start_index, src_padding_mask=None):
        # self attention
        # import pdb
        # pdb.set_trace()
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q.transpose(0, 1),
        #                       k.transpose(0, 1),
        #                       tgt.transpose(0, 1))[0].transpose(0, 1)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        if self.num_layers == 0:
            self.dropout = nn.Dropout(0.1)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        if self.num_layers == 0:
            # import pdb
            # pdb.set_trace()
            return self.dropout(src + pos)

        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points,
                src, src_spatial_shapes,
                level_start_index, src_padding_mask=None):
        # self attention
        # import pdb
        # pdb.set_trace()
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1),
                              k.transpose(0, 1),
                              tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        # import pdb
        # pdb.set_trace()
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, with_box_refine=True, hidden_dim=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        if with_box_refine:
            # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
            self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)
            nn.init.constant_(self.coord_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.coord_embed.layers[-1].bias.data, 0)
            self.coord_embed = _get_clones(self.coord_embed, num_layers)
            nn.init.constant_(self.coord_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            # nn.init.constant_(self.coord_embed.layers[-1].bias.data[2:], -2.0)
            self.coord_embed = None

        self.class_embed = None

    def forward(self, tgt, reference_points,
                src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        # import pdb
        # pdb.set_trace()
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input,
                           src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            # import pdb
            # pdb.set_trace()
            if self.coord_embed is not None:
                tmp = self.coord_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # pdb.set_trace()
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points,
                src, src_spatial_shapes,
                level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1),
                              k.transpose(0, 1),
                              tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                                   src.transpose(0, 1),
                                   src.transpose(0, 1),
                                   attn_mask=None,
                                   key_padding_mask=None)[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class TransformerDecoderLayer2(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer2, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer2, self).__setstate__(state)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # tgt2: torch.Size([17, 1, 512])
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        import pdb
        pdb.set_trace()
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)

if __name__ == "__main__":
    decoder_layer = TransformerDecoderLayer2(d_model=512, nhead=8)
    memory = torch.rand(256, 1, 512)
    tgt = torch.rand(17, 1, 512)
    out = decoder_layer(tgt, memory)
