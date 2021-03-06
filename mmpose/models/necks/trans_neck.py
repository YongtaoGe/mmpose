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

    def __init__(self, in_channals, out_channal):
        super().__init__()
        self.input_proj = nn.ModuleList()
        for in_channal in in_channals:
            self.input_proj.append(
                    nn.Sequential(
                    nn.Conv2d(in_channal, out_channal, kernel_size=1),
                    nn.GroupNorm(32, out_channal),
                ))

    def init_weights(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, inputs):
        if len(inputs)==1:
            inputs = inputs[0]
        else:
            raise NotImplementedError

        out_list = []
        for i, feat in enumerate(inputs):
            out_list.append(self.input_proj[i](feat))
        return out_list