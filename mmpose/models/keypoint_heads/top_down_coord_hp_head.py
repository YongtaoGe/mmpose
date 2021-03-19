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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


@HEADS.register_module()
class OneQueryTransHead(nn.Module):
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
                 loss_coord_keypoint=None,
                 loss_hp_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 hidden_dim=256,
                 out_indices=(0, 1, 2, 3),
                 with_box_refine=True,
                 num_encoder_layers=0,
                 num_decoder_layers=6,
                 decoder_layer_type="deformable",
                 num_stages=1,
                 neck_type=None,
                 heatmap_size=[64, 48]
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
        self.heatmap_size=heatmap_size
        self.loss_coord = build_loss(loss_coord_keypoint)
        import pdb
        pdb.set_trace()

        self.loss_hp = build_loss(loss_hp_keypoint)


        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.hidden_dim = hidden_dim

        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        query_embed = []
        for i in range(self.num_joints):
            query_embed.append(nn.Embedding(1, hidden_dim * 2))
        self.query_embed = nn.Sequential(*query_embed)

        self.transformer = DeformableTransformer(
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
                    two_stage_num_proposals=1
                )



    # def forward(self, input):
    #     """Forward function."""
    #     #[
    #     # [[2, 256, 8, 6], [2, 256, 16, 12], [2, 256, 32, 24], [2, 256, 64, 48]],
    #     #]
    #     feat_for_all_layers, attention_for_all_layers = input
    #     outputs_coords = []
    #     hs_for_all_joints, inter_references_for_all_joints = [], []
    #     pos_embeds_for_all_layers = [self.position_embedding(feat) for feat in feat_for_all_layers]
    #
    #     # hs = torch.cat(hs_for_all_joints, 0)
    #     # inter_references = torch.cat(inter_references_for_all_joints, 0)
    #     # hs_for_all_joints.append(hs)
    #     # inter_references_for_all_joints.append(inter_references)
    #
    #     for join_i in range(self.num_joints):
    #         attend_feat_for_all_layers = []
    #         for feat_i, attention_i in zip(feat_for_all_layers, attention_for_all_layers):
    #             attention_i = attention_i.sigmoid()
    #             attend_feat_for_all_layers.append(
    #                 feat_i * attention_i[:, join_i, :, :].unsqueeze(1)
    #             )
    #         query_embed = self.query_embed[join_i].weight
    #
    #
    #     hs, init_reference, inter_references = \
    #         self.transformer(attend_feat_for_all_layers, pos_embeds_for_all_layers, query_embed)
    #
    #     # hs: [num_layers, bs*17, 1, 256]
    #     for lvl in range(hs.shape[0]):
    #         # stage_i = lvl // self.num_decoder_layers
    #         # stage_lvl = lvl % self.num_decoder_layers
    #         if lvl == 0:
    #             reference = init_reference
    #         else:
    #             reference = inter_references[lvl - 1]
    #         reference = inverse_sigmoid(reference)
    #         # outputs_class = self.class_embed[lvl](hs[lvl])
    #         tmp = self.transformer.decoder.coord_embed[lvl](hs[lvl])
    #         # debug_tmp.append(tmp)
    #         if reference.shape[-1] == 4:
    #             tmp += reference
    #         else:
    #             assert reference.shape[-1] == 2
    #             tmp[..., :2] += reference
    #         outputs_coord = tmp.sigmoid()
    #         # outputs_classes.append(outputs_class)
    #     outputs_coords.append(outputs_coord)
    #
    #
    #     # outputs_class = torch.stack(outputs_classes)
    #     outputs_coord = torch.cat(outputs_coords, dim=1)
    #     # outputs_coord = outputs_coord[-1]
    #     # N, C = output.shape
    #     outputs_hp = []
    #     for attention_i in attention_for_all_layers:
    #         outputs_hp.append(
    #             nn.functional.interpolate(
    #                     attention_i, size=self.heatmap_size, mode='bilinear', align_corners=True)
    #         )
    #
    #     outputs = {
    #         "coord": outputs_coord,
    #         "hp": outputs_hp
    #     }
    #     import pdb
    #     pdb.set_trace()
    #
    #     return outputs


    def forward(self, input):
        """Forward function."""
        #[
        # [[2, 256, 8, 6], [2, 256, 16, 12], [2, 256, 32, 24], [2, 256, 64, 48]],
        #]
        feat_for_all_layers, attention_for_all_layers = input
        outputs_coords = []
        hs_for_all_joints, inter_references_for_all_joints = [], []
        pos_embeds_for_all_layers = [self.position_embedding(feat) for feat in feat_for_all_layers]

        # hs = torch.cat(hs_for_all_joints, 0)
        # inter_references = torch.cat(inter_references_for_all_joints, 0)
        # hs_for_all_joints.append(hs)
        # inter_references_for_all_joints.append(inter_references)

        for join_i in range(self.num_joints):
            attend_feat_for_all_layers = []
            for feat_i, attention_i in zip(feat_for_all_layers, attention_for_all_layers):
                attention_i = attention_i.sigmoid()
                attend_feat_for_all_layers.append(
                    feat_i * attention_i[:, join_i, :, :].unsqueeze(1)
                )
            query_embed = self.query_embed[join_i].weight
            hs, init_reference, inter_references = \
                self.transformer(attend_feat_for_all_layers, pos_embeds_for_all_layers, query_embed)
            # hs: [num_layers, bs, 1, 256]

            for lvl in range(hs.shape[0]):
                # stage_i = lvl // self.num_decoder_layers
                # stage_lvl = lvl % self.num_decoder_layers
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                # outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.transformer.decoder.coord_embed[lvl](hs[lvl])
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
        outputs_coord = torch.cat(outputs_coords, dim=1)
        # outputs_coord = outputs_coord[-1]
        # N, C = output.shape
        outputs_hp = []
        for attention_i in attention_for_all_layers:
            outputs_hp.append(
                nn.functional.interpolate(
                        attention_i, size=self.heatmap_size, mode='bilinear', align_corners=True)
            )

        # outputs_hp = torch.stack(outputs_hp, dim=1)

        outputs = {
            "coord": outputs_coord,
            "hp": outputs_hp
        }
        # import pdb
        # pdb.set_trace()

        return outputs


    def get_loss(self, output, coord_target, coord_target_weight, hp_target, hp_target_weight):
        losses = dict()
        coord_output = output["coord"]
        hp_output = output["hp"]
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
            num_outputs: O
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxOxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxOxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxOxKx1]):
                Weights across different joint types.
        """
        losses = dict()
        # import pdb
        # pdb.set_trace()
        assert isinstance(output, list)
        assert target.dim() == 5 and target_weight.dim() == 4
        assert target.size(1) == len(output)

        if isinstance(self.loss_hp, nn.Sequential):
            assert len(self.loss_hp) == len(output)
        for i in range(len(output)):
            target_i = target[:, i, :, :, :]
            target_weight_i = target_weight[:, i, :, :]

            if isinstance(self.loss_hp, nn.Sequential):
                loss_func = self.loss_hp[i]
            else:
                loss_func = self.loss_hp

            loss_i = loss_func(output[i], target_i, target_weight_i)
            if 'mse_loss' not in losses:
                losses['mse_loss'] = loss_i
            else:
                losses['mse_loss'] += loss_i

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
        assert not isinstance(self.loss_coord, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        # losses['reg_loss'] = self.loss(output[-1], target, target_weight).sum()
        losses['reg_loss'] = self.loss_coord(output[-1], target, target_weight)
        if output.dim() == 4:
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers - 1):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss'] += self.loss_coord(output[i], target, target_weight)
        else:
            losses['reg_loss'] = self.loss_coord(output, target, target_weight)
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
        hp_output = output["hp"]
        accuracy = dict()
        if coord_output.dim() == 4:
            coord_output = output[-1]
        N = coord_output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            coord_output.detach().cpu().numpy(),
            coord_target.detach().cpu().numpy(),
            coord_target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['coord_acc'] = avg_acc

        assert isinstance(hp_output, list)
        assert hp_target.dim() == 5 and hp_target_weight.dim() == 4
        _, avg_acc, _ = pose_pck_accuracy(
            hp_output[-1].detach().cpu().numpy(),
            hp_target[:, -1, ...].detach().cpu().numpy(),
            hp_target_weight[:, -1,
            ...].detach().cpu().numpy().squeeze(-1) > 0)
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
        """Initialize model weights."""
        for j in range(len(self.transformer.decoder.coord_embed)):
            normal_init(self.transformer.decoder.coord_embed[j].layers[0], mean=0, std=0.01, bias=0)
            normal_init(self.transformer.decoder.coord_embed[j].layers[1], mean=0, std=0.01, bias=0)

            nn.init.constant_(self.transformer.decoder.coord_embed[j].layers[-1].weight.data, 0)
            nn.init.constant_(self.transformer.decoder.coord_embed[j].layers[-1].bias.data, 0)
        nn.init.constant_(self.transformer.decoder.coord_embed[0].layers[-1].bias.data[2:], -2.0)


from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from mmpose.models.utils.ops import resize

@HEADS.register_module()
class SimpleBaselineOneQueryTransHead(nn.Module):
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
                 with_box_refine=True
                 ):
        super().__init__()
        self.backbone_out_channels = {0:256, 1:256, 2:256, 3:256}
        # self.out_indices = out_indices
        self.num_levels = 4
        self.num_stages = num_stages
        # self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.neck_type=neck_type
        self.heatmap_size=heatmap_size
        self.loss_coord = build_loss(loss_coord_keypoint)
        self.loss_hp = build_loss(loss_hp_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.hidden_dim = hidden_dim

        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        # self.query_embed = nn.Embedding(17, hidden_dim * 2)
        query_embed = []
        for i in range(self.num_joints):
            query_embed.append(nn.Embedding(1, hidden_dim * 2))
        self.query_embed = nn.Sequential(*query_embed)
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
                    return_intermediate_dec=True,
                    num_feature_levels=self.num_levels,
                    dec_n_points=4,
                    enc_n_points=4,
                    hidden_dim=hidden_dim,
                    with_box_refine=with_box_refine,
                    two_stage=False,
                    two_stage_num_proposals=1
                )

    def forward(self, input):
        """Forward function."""
        #[
        # [[2, 256, 8, 6], [2, 256, 16, 12], [2, 256, 32, 24], [2, 256, 64, 48]],
        #]
        feat_for_all_layers, attention = input
        outputs_coords = []
        hs_for_all_joints, inter_references_for_all_joints = [], []
        pos_embeds_for_all_layers = [self.position_embedding(feat) for feat in feat_for_all_layers]

        # hs = torch.cat(hs_for_all_joints, 0)
        # inter_references = torch.cat(inter_references_for_all_joints, 0)
        # hs_for_all_joints.append(hs)
        # inter_references_for_all_joints.append(inter_references)
        sigmoid_attention = attention.sigmoid()

        for join_i in range(self.num_joints):
            attend_feat_for_all_layers = []
            # for feat_i in feat_for_all_layers:
            #     sigmoid_attention_i = nn.functional.interpolate(
            #             sigmoid_attention, size=[feat_i.size(2), feat_i.size(3)], mode='bilinear', align_corners=True)
            #
            #     attend_feat_for_all_layers.append(
            #         feat_i * sigmoid_attention_i[:, join_i, :, :].unsqueeze(1)
            #     )
            query_embed = self.query_embed[join_i].weight


            hs, init_reference, inter_references = \
                self.transformer(feat_for_all_layers, pos_embeds_for_all_layers, query_embed)
                # self.transformer(attend_feat_for_all_layers, pos_embeds_for_all_layers, query_embed)
            # hs: [num_layers, bs, 1, 256]

            for lvl in range(hs.shape[0]):
                # stage_i = lvl // self.num_decoder_layers
                # stage_lvl = lvl % self.num_decoder_layers
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                # outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.transformer.decoder.coord_embed[lvl](hs[lvl])
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
        outputs_coord = torch.cat(outputs_coords, dim=1)
        # outputs_coord = outputs_coord[-1]
        # N, C = output.shape
        outputs_hp = attention
        # outputs_hp = torch.stack(outputs_hp, dim=1)

        outputs = {
            "coord": outputs_coord,
            "hp": outputs_hp
        }

        return outputs


    def forward11(self, input):
        """Forward function."""
        #[
        # [[2, 256, 8, 6], [2, 256, 16, 12], [2, 256, 32, 24], [2, 256, 64, 48]],
        #]

        feat_for_all_layers, attention = input
        bs = attention.size(0)
        outputs_coords = []
        hs_for_all_joints, inter_references_for_all_joints = [], []
        pos_embeds_for_all_layers = [self.position_embedding(feat) for feat in feat_for_all_layers]

        # hs = torch.cat(hs_for_all_joints, 0)
        # inter_references = torch.cat(inter_references_for_all_joints, 0)
        # hs_for_all_joints.append(hs)
        # inter_references_for_all_joints.append(inter_references)
        sigmoid_attention = attention.sigmoid()


        attent_feat_for_all_joints_all_layers = []
        for join_i in range(self.num_joints):
            attend_feat_for_all_layers = []
            for feat_i in feat_for_all_layers:
                sigmoid_attention_i = nn.functional.interpolate(
                        sigmoid_attention, size=[feat_i.size(2), feat_i.size(3)], mode='bilinear', align_corners=True)
                feat_i = feat_i * sigmoid_attention_i[:, join_i, :, :].unsqueeze(1)

                attend_feat_for_all_layers.append(
                    feat_i.flatten(2).transpose(1, 2)
                )
            attend_feat_for_all_layers = torch.cat(attend_feat_for_all_layers, 1)
            attent_feat_for_all_joints_all_layers.append(attend_feat_for_all_layers)

        attent_feat_for_all_joints_all_layers = torch.stack(attent_feat_for_all_joints_all_layers, dim=0)
        attent_feat_for_all_joints_all_layers = attent_feat_for_all_joints_all_layers.reshape(-1,
                                                                                              attent_feat_for_all_joints_all_layers.size(2),
                                                                                              attent_feat_for_all_joints_all_layers.size(3))

        query_embed = self.query_embed.weight
        hs, init_reference, inter_references = \
            self.transformer(attent_feat_for_all_joints_all_layers, pos_embeds_for_all_layers, query_embed)

        # import pdb
        # pdb.set_trace()
        hs = hs.reshape(self.num_decoder_layers, bs, self.num_joints, self.hidden_dim)
        init_reference = init_reference.reshape(bs, self.num_joints, 2)
        inter_references = inter_references.reshape(self.num_decoder_layers, bs, self.num_joints, 2)
        # hs: [num_layers, bs*17, 1, 256]

        for lvl in range(hs.shape[0]):
            # stage_i = lvl // self.num_decoder_layers
            # stage_lvl = lvl % self.num_decoder_layers
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.transformer.decoder.coord_embed[lvl](hs[lvl])
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
        outputs_coord = torch.cat(outputs_coords, dim=1)
        # outputs_coord = outputs_coord[-1]
        # N, C = output.shape
        outputs_hp = attention
        # outputs_hp = torch.stack(outputs_hp, dim=1)

        outputs = {
            "coord": outputs_coord,
            "hp": outputs_hp
        }

        return outputs


    def get_loss(self, output, coord_target, coord_target_weight, hp_target, hp_target_weight):
        losses = dict()
        coord_output = output["coord"]
        hp_output = output["hp"]
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
        assert not isinstance(self.loss_hp, nn.Sequential)
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
        assert not isinstance(self.loss_coord, nn.Sequential)
        assert target.dim() == 3 and target_weight.dim() == 3
        # losses['reg_loss'] = self.loss(output[-1], target, target_weight).sum()
        losses['reg_loss'] = self.loss_coord(output[-1], target, target_weight)
        if output.dim() == 4:
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers - 1):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss'] += self.loss_coord(output[i], target, target_weight)
        else:
            losses['reg_loss'] = self.loss_coord(output, target, target_weight)
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
        hp_output = output["hp"]
        accuracy = dict()
        if coord_output.dim() == 4:
            coord_output = output[-1]
        N = coord_output.shape[0]

        _, avg_acc, cnt = keypoint_pck_accuracy(
            coord_output.detach().cpu().numpy(),
            coord_target.detach().cpu().numpy(),
            coord_target_weight[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=np.ones((N, 2), dtype=np.float32))
        accuracy['coord_acc'] = avg_acc

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
        """Initialize model weights."""
        for j in range(len(self.transformer.decoder.coord_embed)):
            normal_init(self.transformer.decoder.coord_embed[j].layers[0], mean=0, std=0.01, bias=0)
            normal_init(self.transformer.decoder.coord_embed[j].layers[1], mean=0, std=0.01, bias=0)

            nn.init.constant_(self.transformer.decoder.coord_embed[j].layers[-1].weight.data, 0)
            nn.init.constant_(self.transformer.decoder.coord_embed[j].layers[-1].bias.data, 0)
        nn.init.constant_(self.transformer.decoder.coord_embed[0].layers[-1].bias.data[2:], -2.0)






@HEADS.register_module()
class HybridTransHead(nn.Module):
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
        assert num_decoder_layers == len(self.loss_coord)

        self.loss_hp = build_loss(loss_hp_keypoint)
        self.use_heatmap_loss = use_heatmap_loss

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.hidden_dim = hidden_dim

        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.query_embed = nn.Embedding(num_joints, hidden_dim * 2)

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

        if self.neck_type == 'RSNNeck':
            # feat_for_all_stages = feat_for_all_stages[0]
            feat_c2 = feat_for_all_stages[0].pop(0)

        if self.neck_type == 'FPN':
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
                    hs, init_reference, inter_references, outputs_hp = \
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
        # import pdb
        # pdb.set_trace()
        if self.use_multi_stage_memory:
            outputs_hp.insert(0, feat_c2)
            outputs_hp = torch.stack(outputs_hp)

        outputs = {
            "coord": outputs_coord,
            "hp": outputs_hp
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
        # import pdb
        # pdb.set_trace()
        if isinstance(self.loss_hp, nn.Sequential):
            assert len(self.loss_hp) == output.size(0)
            assert target.dim() == 5 and target_weight.dim() == 4
            num_hp_layers = output.size(0)
            for i in range(num_hp_layers):
                target_i = target[:, i, :, :, :]
                target_weight_i = target_weight[:, i, :, :]
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['mse_loss_{}'.format(i)] = self.loss_hp[i](output[i], target_i, target_weight_i)

        else:
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
        if output.dim() == 4:
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers):
                # losses['reg_loss'] += self.loss(output[i], target, target_weight).sum()
                losses['reg_loss_{}'.format(i)] = self.loss_coord[i](output[i], target, target_weight)
        else:
            raise NotImplementedError
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
                hp_output[0].detach().cpu().numpy(),
                hp_target[:, 0, ...].detach().cpu().numpy(),
                hp_target_weight[:, 0,
                              ...].detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['hp_acc'] = float(avg_acc)
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