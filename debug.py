from mmpose.models import HRNet
import torch

extra = dict(
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block='BOTTLENECK',
        num_blocks=(4,),
        num_channels=(64,)),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(32, 64)),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(32, 64, 128)),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(32, 64, 128, 256))),

model = HRNet(*extra, in_channels=1)
model.eval()
inputs = torch.rand(1, 1, 32, 32)
level_outputs = model.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))