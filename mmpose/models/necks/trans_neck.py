import torch
import torch.nn as nn

from ..registry import NECKS


@NECKS.register_module()
class InputProj(nn.Module):
    """Global Average Pooling neck.
    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    """

    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                    nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1),
                    nn.GroupNorm(32, out_channel),
                ))

    def init_weights(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.input_proj)
        out_list = []
        for i, feat in enumerate(inputs):
            out_list.append(self.input_proj[i](feat))
        return [out_list]