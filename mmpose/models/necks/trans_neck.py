import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
from ..registry import NECKS


@NECKS.register_module()
class InputProj(nn.Module):
    """Global Average Pooling neck.
    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    """

    def __init__(self, in_channels, out_channel, backbone_type=None):
        super().__init__()
        self.input_proj = nn.ModuleList()
        self.backbone_type = backbone_type
        for in_channel in in_channels:
            self.input_proj.append(
                    nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1),
                    # nn.GroupNorm(32, out_channel),
                    nn.BatchNorm2d(out_channel),
                ))

    def init_weights(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, inputs):
        if self.backbone_type == 'HRNet':
            inputs = inputs[1:]
            # inputs.pop(0)

        assert len(inputs) == len(self.input_proj)
        out_list = []
        for i, feat in enumerate(inputs):
            out_list.append(self.input_proj[i](feat))
        return [out_list]


@NECKS.register_module()
class OneQueryInputProj(nn.Module):
    """Global Average Pooling neck.
    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    """

    def __init__(self, in_channels, out_channel, num_joints):
        super().__init__()
        self.input_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.query_proj = nn.ModuleList()

        for in_channel in in_channels:
            self.input_proj.append(
                ConvModule(
                    in_channel,
                    out_channel,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                )
            self.attention_proj.append(
                ConvModule(
                    out_channel,
                    num_joints,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                )
            self.query_proj.append(
                ConvModule(
                    out_channel,
                    out_channel,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                )

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        if len(inputs)==1 and len(inputs[0])==4:
            inputs = inputs[0]

        assert len(inputs) == len(self.input_proj) and len(inputs) == len(self.attention_proj)
        query_list = []
        attention_list = []
        for i, feat in enumerate(inputs):
            feat = self.input_proj[i](feat)
            query_list.append(self.query_proj[i](feat))
            attention_list.append(self.attention_proj[i](feat))

        return [query_list, attention_list]


from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from mmpose.models.utils.ops import resize


@NECKS.register_module()
class SimpleBaselineNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 query_in_channels,
                 query_out_channels=256,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

        self.query_proj_layers = nn.ModuleList()
        for in_channel in query_in_channels:
            self.query_proj_layers.append(
                ConvModule(
                    in_channel,
                    query_out_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                )

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

    def forward(self, x):
        """Forward function."""
        # import pdb
        # pdb.set_trace()
        query_list = []
        for i, feat in enumerate(x):
            query_list.append(self.query_proj_layers[i](feat))

        x = x[-1]
        # [bs,512,8,6]
        x = self._transform_inputs(x)
        # [bs,256,64,48]
        x = self.deconv_layers(x)
        # [bs,17,64,48]
        x = self.final_layer(x)
        # [[bs,256,64,48],[bs,256,64,48],[bs,256,64,48],[bs,256,64,48],  [bs,17,64,48]
        return query_list, x

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.query_proj_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)



from mmpose.models.keypoint_heads.top_down_multi_stage_head import PredictHeatmap
import copy as cp
from mmcv.cnn import kaiming_init, normal_init
@NECKS.register_module()
class RSNNeck(nn.Module):
    """Heads for multi-stage multi-unit heads used in Multi-Stage Pose
    estimation Network (MSPN), and Residual Steps Networks (RSN).

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        num_stages (int): Number of stages.
        num_units (int): Number of units in each stage.
        use_prm (bool): Whether to use pose refine machine (PRM).
            Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 out_shape=(64, 48),
                 unit_channels=256,
                 out_channels=17,
                 num_stages=1,
                 num_units=4,
                 use_prm=False,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)

        self.out_shape = out_shape
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.num_units = num_units

        # self.predict_layers = nn.ModuleList([])
        # for i in range(self.num_stages):
        #     for j in range(self.num_units):
        #         self.predict_layers.append(
        #             PredictHeatmap(
        #                 unit_channels,
        #                 out_channels,
        #                 out_shape,
        #                 use_prm,
        #                 norm_cfg=norm_cfg))

        self.predict_layer = PredictHeatmap(
                            unit_channels,
                            out_channels,
                            out_shape,
                            use_prm,
                            norm_cfg=norm_cfg)

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages
                                and units.
        """
        # out = []
        # import pdb
        # pdb.set_trace()
        assert isinstance(x, list)
        assert len(x) == self.num_stages
        assert isinstance(x[0], list)
        assert len(x[0]) == self.num_units
        assert x[0][0].shape[1] == self.unit_channels

        # for i in range(self.num_stages):
        #     for j in range(self.num_units):
        #         y = self.predict_layers[i * self.num_units + j](x[i][j])
        #         out.append(y)

        feat_c2 = self.predict_layer(x[0][3])
        feat_c3 = x[0][2]
        feat_c4 = x[0][1]
        feat_c5 = x[0][0]

        return [[feat_c2, feat_c3, feat_c4, feat_c5]]


    def init_weights(self):
        """Initialize model weights."""
        for m in self.predict_layer.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)