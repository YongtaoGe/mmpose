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
from mmpose.models.utils.transformer_utils import PositionEmbeddingSine
from .deformable_transformer import DeformableTransformer
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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


@HEADS.register_module()
class coord_pose_head(nn.Module):
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
                 hidden_dim = 256,
                 out_indices=(0, 1, 2, 3),
                 with_box_refine = True):
        super().__init__()
        self.backbone_out_channels = {0:256,1:256,2:256,3:256}
        self.num_level = len(out_indices)

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.hidden_dim = hidden_dim
        self.position_embedding = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        self.out_indices = out_indices

        self.query_embed = nn.Embedding(self.num_joints, hidden_dim*2)
        self.input_proj = nn.ModuleList()
        for index in out_indices:
            self.input_proj.append(                
                    nn.Sequential(
                    nn.Conv2d(self.backbone_out_channels[index], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))

        # self.decoder = build_neck(decoder_cfg)
        self.transformer = DeformableTransformer(
            d_model=self.hidden_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=self.num_level,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=False,
            two_stage_num_proposals=1
        )

        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(self.coord_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coord_embed.layers[-1].bias.data, 0)
        num_pred = self.transformer.decoder.num_layers
        if with_box_refine:
            # self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coord_embed = _get_clones(self.coord_embed, num_pred)
            nn.init.constant_(self.coord_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            # self.transformer.decoder.coord_embed = self.coord_embed
        else:
            nn.init.constant_(self.coord_embed.layers[-1].bias.data[2:], -2.0)
            # self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.coord_embed = nn.ModuleList([self.coord_embed for _ in range(num_pred)])
            # self.transformer.decoder.coord_embed = None

    def forward(self, x):
        """Forward function."""
        # import pdb
        # pdb.set_trace()
        #[
        # [[2, 256, 8, 6],[2, 256, 8, 6],[2, 256, 8, 6],[2, 256, 8, 6]],
        #]
        # assert len(x) == len(self.out_indices)
        if len(x)==1:
            x = x[0]
        else:
            raise NotImplementedError

        pos = [self.position_embedding(feat) for feat in x]
        features = []
        for i in range(len(x)):
            features.append(self.input_proj[i](x[i]))
        query_embed = self.query_embed.weight
        hs, init_reference, inter_references = \
            self.transformer(features, pos, query_embed)
        
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # outputs_class = self.class_embed[lvl](hs[lvl])

            tmp = self.coord_embed[lvl](hs[lvl])
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
        losses['reg_loss'] = 0

        if output.dim() == 4:
            num_decode_layers = output.size(0)
            for i in range(num_decode_layers):
                losses['reg_loss'] += self.loss(output[i], target, target_weight)
            # import pdb
            # pdb.set_trace()
        else:
            losses['reg_loss'] = self.loss(output, target, target_weight)
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
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        pass
