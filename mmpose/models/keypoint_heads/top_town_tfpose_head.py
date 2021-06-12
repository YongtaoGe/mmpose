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
from .deformable_transformer import DeformableTransformer, OneQueryDeformableTransformer
from mmpose.core.evaluation import pose_pck_accuracy

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init

@HEADS.register_module()
class TFPoseHead0(nn.Module):
    """Top-down model head of simple baseline paper ref: Bin Xiao. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopDownSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 # in_channels,
                 train_cfg=None,
                 test_cfg=None,
                 num_encoder_layers=0,
                 num_decoder_layers=6,
                 decoder_layer_type="deformable",
                 heatmap_size=[64, 48],
                 num_joints=17,
                 loss_coord_keypoint=None,
                 loss_hp_keypoint=None,
                 num_stages=1,
                 hidden_dim=256,
                 neck_type="SimpleBaselineNeck",
                 with_box_refine=True,
                 use_heatmap_loss=True,
                 use_multi_stage_memory=False,
                 num_levels=3,
                 ):
        super().__init__()
        self.backbone_out_channels = {0:256, 1:256, 2:256, 3:256}
        # self.out_indices = out_indices
        self.num_levels = num_levels
        self.num_stages = num_stages
        # self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_multi_stage_memory = use_multi_stage_memory
        self.neck_type=neck_type
        self.heatmap_size=heatmap_size
        self.loss_coord = build_loss(loss_coord_keypoint)
        # assert num_decoder_layers == len(self.loss_coord)

        self.loss_hp = build_loss(loss_hp_keypoint)
        self.use_heatmap_loss = use_heatmap_loss

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.hidden_dim = hidden_dim

        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.query_embed = nn.Embedding(num_joints, hidden_dim * 2)

        if neck_type == "RSNNeck":
            return_intermediate_enc = True
        else:
            return_intermediate_enc = False

        # self.transformer = OneQueryDeformableTransformer(
        self.transformer = DeformableTransformer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    decoder_layer_type=decoder_layer_type,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation="relu",
                    return_intermediate_enc=return_intermediate_enc,
                    return_intermediate_dec=True,
                    num_feature_levels=self.num_levels,
                    dec_n_points=4,
                    enc_n_points=4,
                    hidden_dim=hidden_dim,
                    with_box_refine=with_box_refine,
                    two_stage=False,
                    two_stage_num_proposals=1,
                    use_heatmap_loss=use_heatmap_loss,
                    use_multi_stage_memory=use_multi_stage_memory,
                    num_joints=self.num_joints,
                )

    def forward(self, feat_for_all_stages):
        # if True:
        #     return feat_for_all_stages

        if self.neck_type == 'RSNNeck':
            # feat_for_all_stages = feat_for_all_stages[0]
            outputs_hp_backbone, feat_for_all_stages = feat_for_all_stages

        if 'FPN' in self.neck_type:
            feat_for_all_stages = [feat_for_all_stages]

        outputs_coords = []
        hs_for_all_stages, inter_references_for_all_stages = [], []

        for stage_i, feat_for_one_stage in enumerate(feat_for_all_stages):
            pos_embeds_for_one_stage = [self.position_embedding(feat) for feat in feat_for_one_stage]
            if self.neck_type == 'FPN':
                feat_for_one_stage = feat_for_one_stage[::-1]
                pos_embeds_for_one_stage = pos_embeds_for_one_stage[::-1]

            query_embed = self.query_embed.weight

            if stage_i == 0:
                if self.use_heatmap_loss:
                    # import pdb
                    # pdb.set_trace()
                    hs, init_reference, inter_references, outputs_hp_enc = \
                        self.transformer(feat_for_one_stage, pos_embeds_for_one_stage, query_embed)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

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
            tmp = self.transformer.decoder.coord_embed[stage_lvl](hs[lvl])
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

        if self.use_multi_stage_memory:
            # outputs_hp_enc = torch.stack(outputs_hp)
            outputs = {
                "coord": outputs_coord,
                "hp": {
                    'backbone': outputs_hp_backbone,
                    "enc": outputs_hp_enc,
                    }
            }

        else:
            outputs = {
                "coord": outputs_coord,
                "hp": outputs_hp_enc
            }
        return outputs


    def get_loss(self, output, coord_target, coord_target_weight, hp_target, hp_target_weight):
        losses = dict()
        coord_output = output["coord"]
        hp_output = output["hp"]
        # import pdb
        # pdb.set_trace()
        coord_loss = self.get_coord_loss(coord_output, coord_target, coord_target_weight)
        hp_loss = self.get_hp_loss(hp_output, hp_target, hp_target_weight)
        losses.update(coord_loss)
        losses.update(hp_loss)

        return losses

    def get_hp_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()

        if isinstance(self.loss_hp, nn.Sequential):
            if not isinstance(output, dict):
                assert len(self.loss_hp) == output.size(0)
                assert target.dim() == 5 and target_weight.dim() == 4
                num_hp_layers = output.size(0)
                for i in range(num_hp_layers):
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                    # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                    losses['mse_loss_{}'.format(i)] = self.loss_hp[i](output[i], target_i, target_weight_i)
            else:
                out_hp_backbone = output['backbone']
                num_hp_layers = out_hp_backbone.size(0)
                for i in range(num_hp_layers):
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                    # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                    losses['mse_loss_backbone_{}'.format(i)] = self.loss_hp[i](out_hp_backbone[i], target_i, target_weight_i)

                out_hp_enc = output['enc']
                for lvl in range(len(out_hp_enc)):
                    if lvl==2 or lvl==5:
                    # if lvl == 5:
                        for i in range(3):
                            target_i = target[:, i+1, :, :, :]
                            target_weight_i = target_weight[:, i+1, :, :]
                        # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                            if lvl == 2:
                                loss_weight = 0.1
                            elif lvl == 5:
                                loss_weight = 1.0

                            losses['mse_loss_enc_layer{}_c{}'.format(lvl, i+3)] = loss_weight * self.loss_hp[i+1](out_hp_enc[lvl][i], target_i, target_weight_i)
        else:
            # import pdb
            # pdb.set_trace()
            assert target.dim() == 4 and target_weight.dim() == 3
            losses['mse_loss'] = self.loss_hp(output, target, target_weight)
        # import pdb
        # pdb.set_trace()
        return losses

    def get_coord_loss(self, output, target, target_weight):
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
        # import pdb
        # pdb.set_trace()
        # assert not isinstance(self.loss_coord, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        # losses['reg_loss'] = self.loss(output[-1], target, target_weight).sum()
        # losses['reg_loss'] = self.loss_coord(output[-1], target, target_weight)
        # losses['reg_loss'] = 0
        if output.dim() == 4 and isinstance(self.loss_coord, nn.Sequential):
            assert self.num_decoder_layers == len(self.loss_coord)
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss_{}'.format(i)] = self.loss_coord[i](output[i], target, target_weight)
        else:
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss_{}'.format(i)] = self.loss_coord(output[i], target, target_weight)

        return losses

    def get_accuracy(self, output, coord_target, coord_target_weight, hp_target, hp_target_weight):
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
        coord_output = output["coord"]
        if self.use_multi_stage_memory:
            # hp_output = output["hp"]["backbone"]
            hp_output_backbone = output["hp"]["backbone"]
            hp_output_enc = output["hp"]["enc"]
        else:
            hp_output = output["hp"]

        accuracy = dict()
        if coord_output.dim() == 4:
            coord_output = coord_output[-1]
        N = coord_output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            coord_output.detach().cpu().numpy(),
            coord_target.detach().cpu().numpy(),
            coord_target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['coord_acc'] = avg_acc

        if self.use_heatmap_loss and self.use_multi_stage_memory:
            assert hp_target.dim() == 5 and hp_target_weight.dim() == 4
            _, avg_acc, _ = pose_pck_accuracy(
                hp_output_backbone[0].detach().cpu().numpy(),
                hp_target[:, 0, ...].detach().cpu().numpy(),
                hp_target_weight[:, 0,
                              ...].detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc_backbone'] = float(avg_acc)

            _, avg_acc, _ = pose_pck_accuracy(
                hp_output_enc[-1][0].detach().cpu().numpy(),
                hp_target[:, 1, ...].detach().cpu().numpy(),
                hp_target_weight[:, 1,
                              ...].detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc_enc'] = float(avg_acc)

        else:
            _, avg_acc, _ = pose_pck_accuracy(
                hp_output.detach().cpu().numpy(),
                hp_target.detach().cpu().numpy(),
                hp_target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc'] = float(avg_acc)

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
        coord_output = output["coord"]
        hp_output = output["hp"]

        if coord_output.dim() == 4:
            output = coord_output[-1]

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
        for j in range(len(self.transformer.decoder.coord_embed)):
            normal_init(self.transformer.decoder.coord_embed[j].layers[0], mean=0, std=0.01, bias=0)
            normal_init(self.transformer.decoder.coord_embed[j].layers[1], mean=0, std=0.01, bias=0)

            nn.init.constant_(self.transformer.decoder.coord_embed[j].layers[-1].weight.data, 0)
            nn.init.constant_(self.transformer.decoder.coord_embed[j].layers[-1].bias.data, 0)
        nn.init.constant_(self.transformer.decoder.coord_embed[0].layers[-1].bias.data[2:], -2.0)


@HEADS.register_module()
class TFPoseHead(nn.Module):

    def __init__(self,
                 *args,
                 train_cfg=None,
                 test_cfg=None,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 aux_loss=True,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.aux_loss = aux_loss
        if self.as_two_stage:
            assert 'as_two_stage' in transformer and \
                   transformer['as_two_stage'] == self.as_two_stage

        super(TFPoseHead, self).__init__(
            *args, transformer=transformer, **kwargs)

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = []
        for _ in range(self.reg_num_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))

        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:

            self.cls_branch = _get_clones(fc_cls, num_pred)
            self.reg_branch = _get_clones(reg_branch, num_pred)
            # TODO find a better way, set_refine_head??
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:

            self.cls_branch = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branch = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            # TODO find a better way
            self.transformer.decoder.bbox_embed = None

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weight()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branch:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branch:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branch[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            # TODO find a better way, set cls_branch??
            self.transformer.decoder.class_embed = self.cls_branch
            for m in self.reg_branch:
                nn.init.constant_(m.layers[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.
        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.
                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord_unact = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        def inverse_sigmoid(x, eps=1e-5):
            x = x.clamp(min=0, max=1)
            x1 = x.clamp(min=eps)
            x2 = (1 - x).clamp(min=eps)
            return torch.log(x1 / x2)

        # TODO save memory when test, only return last lalyer
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # TODO check brach
            outputs_class = self.cls_branch[lvl](hs[lvl])
            tmp = self.reg_branch[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords



    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses





    def get_loss(self, output, coord_target, coord_target_weight, hp_target, hp_target_weight):
        losses = dict()
        coord_output = output["coord"]
        hp_output = output["hp"]
        # import pdb
        # pdb.set_trace()
        coord_loss = self.get_coord_loss(coord_output, coord_target, coord_target_weight)
        hp_loss = self.get_hp_loss(hp_output, hp_target, hp_target_weight)
        losses.update(coord_loss)
        losses.update(hp_loss)

        return losses

    def get_hp_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()

        if isinstance(self.loss_hp, nn.Sequential):
            if not isinstance(output, dict):
                assert len(self.loss_hp) == output.size(0)
                assert target.dim() == 5 and target_weight.dim() == 4
                num_hp_layers = output.size(0)
                for i in range(num_hp_layers):
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                    # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                    losses['mse_loss_{}'.format(i)] = self.loss_hp[i](output[i], target_i, target_weight_i)
            else:
                out_hp_backbone = output['backbone']
                num_hp_layers = out_hp_backbone.size(0)
                for i in range(num_hp_layers):
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                    # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                    losses['mse_loss_backbone_{}'.format(i)] = self.loss_hp[i](out_hp_backbone[i], target_i, target_weight_i)

                out_hp_enc = output['enc']
                for lvl in range(len(out_hp_enc)):
                    if lvl==2 or lvl==5:
                    # if lvl == 5:
                        for i in range(3):
                            target_i = target[:, i+1, :, :, :]
                            target_weight_i = target_weight[:, i+1, :, :]
                        # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                            if lvl == 2:
                                loss_weight = 0.1
                            elif lvl == 5:
                                loss_weight = 1.0

                            losses['mse_loss_enc_layer{}_c{}'.format(lvl, i+3)] = loss_weight * self.loss_hp[i+1](out_hp_enc[lvl][i], target_i, target_weight_i)
        else:
            # import pdb
            # pdb.set_trace()
            assert target.dim() == 4 and target_weight.dim() == 3
            losses['mse_loss'] = self.loss_hp(output, target, target_weight)
        # import pdb
        # pdb.set_trace()
        return losses

    def get_coord_loss(self, output, target, target_weight):
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
        # import pdb
        # pdb.set_trace()
        # assert not isinstance(self.loss_coord, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        # losses['reg_loss'] = self.loss(output[-1], target, target_weight).sum()
        # losses['reg_loss'] = self.loss_coord(output[-1], target, target_weight)
        # losses['reg_loss'] = 0
        if output.dim() == 4 and isinstance(self.loss_coord, nn.Sequential):
            assert self.num_decoder_layers == len(self.loss_coord)
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss_{}'.format(i)] = self.loss_coord[i](output[i], target, target_weight)
        else:
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss_{}'.format(i)] = self.loss_coord(output[i], target, target_weight)

        return losses

    def get_accuracy(self, output, coord_target, coord_target_weight, hp_target, hp_target_weight):
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
        coord_output = output["coord"]
        if self.use_multi_stage_memory:
            # hp_output = output["hp"]["backbone"]
            hp_output_backbone = output["hp"]["backbone"]
            hp_output_enc = output["hp"]["enc"]
        else:
            hp_output = output["hp"]

        accuracy = dict()
        if coord_output.dim() == 4:
            coord_output = coord_output[-1]
        N = coord_output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            coord_output.detach().cpu().numpy(),
            coord_target.detach().cpu().numpy(),
            coord_target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['coord_acc'] = avg_acc

        if self.use_heatmap_loss and self.use_multi_stage_memory:
            assert hp_target.dim() == 5 and hp_target_weight.dim() == 4
            _, avg_acc, _ = pose_pck_accuracy(
                hp_output_backbone[0].detach().cpu().numpy(),
                hp_target[:, 0, ...].detach().cpu().numpy(),
                hp_target_weight[:, 0,
                              ...].detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc_backbone'] = float(avg_acc)

            _, avg_acc, _ = pose_pck_accuracy(
                hp_output_enc[-1][0].detach().cpu().numpy(),
                hp_target[:, 1, ...].detach().cpu().numpy(),
                hp_target_weight[:, 1,
                              ...].detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc_enc'] = float(avg_acc)

        else:
            _, avg_acc, _ = pose_pck_accuracy(
                hp_output.detach().cpu().numpy(),
                hp_target.detach().cpu().numpy(),
                hp_target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc'] = float(avg_acc)

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
        coord_output = output["coord"]
        hp_output = output["hp"]

        if coord_output.dim() == 4:
            output = coord_output[-1]

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
