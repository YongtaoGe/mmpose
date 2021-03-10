import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)
from mmpose.core.post_processing import fliplr_regression
from mmpose.models.builder import build_loss
from mmpose.models.registry import HEADS
from mmpose.models.utils.transformer_utils import PositionEmbeddingSine, PositionEmbeddingLearned
from .deformable_transformer import DeformableTransformer
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


@HEADS.register_module()
class TransHead(nn.Module):
    """regression head with fully connected layers.

    paper ref: Alexander Toshev and Christian Szegedy,
    ``DeepPose: Human Pose Estimation via Deep Neural Networks.''.

    Args:
        in_channels (int): Number of input channels
        num_joints (int): Number of joints
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 hidden_dim=256,
                 out_indices=(0, 1, 2, 3),
                 with_box_refine=True,
                 num_encoder_layers=0,
                 num_decoder_layers=6,
                 decoder_layer_type="deformable",
                 decoder_use_self_attn=True,
                 num_stages=1,
                 neck_type=None,
                 decoder_share_query=False
                 ):
        super().__init__()
        self.backbone_out_channels = {0:256, 1:256, 2:256, 3:256}
        self.out_indices = out_indices
        self.num_levels = len(out_indices)
        self.num_stages = num_stages
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.neck_type=neck_type

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.hidden_dim = hidden_dim

        # if self.num_encoder_layers > 0:
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        # else:
        #     patch_embed_modules = []
        #     feat_size = [[64, 48], [32, 24], [16, 12], [8, 6]]
        #     for i in range(self.num_levels):
        #         patch_embed_modules.append(
        #             PositionEmbeddingSine(hidden_dim // 2, normalize=True))
        #             # PositionEmbeddingLearned(hidden_dim // 2, feat_h=feat_size[i][0], feat_w=feat_size[i][1]))
        #     self.position_embedding = nn.Sequential(*patch_embed_modules)

        if decoder_share_query:
            self.query_embed = nn.Embedding(1, hidden_dim * 2)
        else:
            self.query_embed = nn.Embedding(self.num_joints, hidden_dim * 2)

        transformer_modules = []
        for i in range(num_stages):
            transformer_modules.append(
                    DeformableTransformer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    decoder_layer_type=decoder_layer_type,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="relu",
                    return_intermediate_dec=True,
                    num_feature_levels=self.num_levels,
                    dec_n_points=4,
                    enc_n_points=4,
                    hidden_dim=hidden_dim,
                    with_box_refine=with_box_refine,
                    two_stage=False,
                    two_stage_num_proposals=1,
                    decoder_use_self_attn=decoder_use_self_attn,
                    decoder_share_query=decoder_share_query,
                    num_joints=num_joints
                )
            )
        self.transformer = nn.Sequential(*transformer_modules)


    def forward(self, feat_for_all_stages):
        """Forward function."""
        #[
        # [[2, 256, 8, 6], [2, 256, 16, 12], [2, 256, 32, 24], [2, 256, 64, 48]],
        #]
        # assert len(x) == len
        # import pdb
        # pdb.set_trace()
        if self.neck_type == 'FPN':
            feat_for_all_stages = [feat_for_all_stages]

        outputs_coords = []
        hs_for_all_stages, inter_references_for_all_stages = [], []

        for stage_i, feat_for_one_stage in enumerate(feat_for_all_stages):
            # if self.num_encoder_layers > 0:
            pos_embeds_for_one_stage = [self.position_embedding(feat) for feat in feat_for_one_stage]
            # else:
            #     pos_embeds_for_one_stage = [self.position_embedding[i](feat) for i, feat in enumerate(feat_for_one_stage)]

            feat_for_one_stage = feat_for_one_stage[::-1]
            pos_embeds_for_one_stage = pos_embeds_for_one_stage[::-1]

            query_embed = self.query_embed.weight
            if stage_i == 0:
                hs, init_reference, inter_references = \
                    self.transformer[stage_i](feat_for_one_stage, pos_embeds_for_one_stage, query_embed)
            else:
                src_flatten = []
                spatial_shapes = []
                for lvl, (src, pos_embed) in enumerate(zip(feat_for_one_stage, pos_embeds_for_one_stage)):
                    bs, c, h, w = src.shape
                    spatial_shape = (h, w)
                    spatial_shapes.append(spatial_shape)
                    src = src.flatten(2).transpose(1, 2)
                    src_flatten.append(src)
                    # pos_embed = pos_embed.flatten(2).transpose(1, 2)
                src_flatten = torch.cat(src_flatten, 1)
                spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
                level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
                valid_ratios = torch.ones([src_flatten.size(0), self.num_levels, 2],
                                          dtype=torch.float, device=src_flatten.device)


                bs, _, c = src_flatten.shape
                # torch.Size([17, 512]) -> torch.Size([17, 256]) and torch.Size([17, 256])
                query_embed_i, tgt_i = torch.split(query_embed, c, dim=1)
                # torch.Size([17, 256]) -> torch.Size([bs, 17, 256])
                query_embed_i = query_embed_i.unsqueeze(0).expand(bs, -1, -1)

                hs, inter_references = \
                    self.transformer[stage_i].decoder(
                        hs[-1],
                        inter_references[-1],
                        src_flatten,
                        spatial_shapes,
                        level_start_index,
                        valid_ratios,
                        query_embed_i,
                        )

            hs_for_all_stages.append(hs)
            inter_references_for_all_stages.append(inter_references)

        hs = torch.cat(hs_for_all_stages, 0)
        inter_references = torch.cat(inter_references_for_all_stages, 0)

        for lvl in range(hs.shape[0]):
            stage_i = lvl // self.num_decoder_layers
            stage_lvl = lvl % self.num_decoder_layers
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.transformer[stage_i].decoder.coord_embed[stage_lvl](hs[lvl])
            # debug_tmp.append(tmp)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            # outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # outputs_coord = outputs_coord[-1]
        # N, C = output.shape
        return outputs_coord

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[num_layers, N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        # import pdb
        # pdb.set_trace()
        # losses['reg_loss'] = self.loss(output[-1], target, target_weight).sum()
        losses['reg_loss'] = self.loss(output[-1], target, target_weight)
        if output.dim() == 4:
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers-1):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss'] += self.loss(output[i], target, target_weight)
        else:
            losses['reg_loss'] = self.loss(output, target, target_weight)
        #
        # ############
        #     if isinstance(self.loss, nn.Sequential):
        #         assert len(self.loss) == len(output)
        #     for i in range(len(output)):
        #         target_i = target
        #         target_weight_i = target_weight
        #         if isinstance(self.loss, nn.Sequential):
        #             loss_func = self.loss[i]
        #         else:
        #             loss_func = self.loss
        #         loss_i = loss_func(output[i], target_i, target_weight_i)
        #         if 'reg_loss' not in losses:
        #             losses['reg_loss'] = loss_i
        #         else:
        #             losses['reg_loss'] += loss_i

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints.
            target (torch.Tensor[N, K, 2]): Target keypoints.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        accuracy = dict()
        if output.dim() == 4:
            output = output[-1]
        N = output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['acc_pose'] = avg_acc

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        if output.dim() == 4:
            output = output[-1]

        if flip_pairs is not None:
            output_regression = fliplr_regression(
                output.detach().cpu().numpy(), flip_pairs)
        else:
            output_regression = output.detach().cpu().numpy()
        return output_regression

    def decode_keypoints(self, img_metas, output_regression, img_size):
        """Decode keypoints from output regression.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output_regression (np.ndarray[N, K, 2]): model
                predicted regression vector.
            img_size (tuple(img_width, img_height)): model input image size.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_regression(output_regression, c, s,
                                                   img_size)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def init_weights(self):
        for i in range(len(self.transformer)):
            # import pdb
            # pdb.set_trace()
            for j in range(len(self.transformer[i].decoder.coord_embed)):
                normal_init(self.transformer[i].decoder.coord_embed[j].layers[0], mean=0, std=0.01, bias=0)
                normal_init(self.transformer[i].decoder.coord_embed[j].layers[1], mean=0, std=0.01, bias=0)

                nn.init.constant_(self.transformer[i].decoder.coord_embed[j].layers[-1].weight.data, 0)
                nn.init.constant_(self.transformer[i].decoder.coord_embed[j].layers[-1].bias.data, 0)
            nn.init.constant_(self.transformer[i].decoder.coord_embed[0].layers[-1].bias.data[2:], -2.0)
