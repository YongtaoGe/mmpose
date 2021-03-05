import copy

import torch
import torch.nn as nn
from mmcv.runner import OptimizerHook
from torch.nn.utils import clip_grad


class ParamwiseOptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
    """

    def __init__(self, paramwise_cfg=None, grad_clip=None):
        self.grad_clip = grad_clip
        self.paramwise_cfg = paramwise_cfg

    def match_name_keywords(self, n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    def clip_grads(self, model):
        grad_norm_dict = {}
        param_name_list = []
        if self.paramwise_cfg is not None and 'custom_keys' in self.paramwise_cfg:
            # [print(param_name) for param_name, grad_clip in self.paramwise_cfg['custom_keys']]
            for param_name, grad_clip in self.paramwise_cfg['custom_keys'].items():
                param_name_list.append(param_name)
                params = []
                for n, p in model.named_parameters():
                    if self.match_name_keywords(n, [param_name]) and p.requires_grad:
                        params.append(p)
                        # named_params.pop(n)
                if len(params) > 0:
                    key = param_name+'_grad_norm'
                    grad_norm_dict[key] = float(clip_grad.clip_grad_norm_(params, **grad_clip['grad_clip']))

        if self.grad_clip is not None:
            rest_params = []
            for n, p in model.named_parameters():
                if p.requires_grad and not self.match_name_keywords(n, param_name_list):
                    rest_params.append(p)

            if len(rest_params) > 0:
                grad_norm_dict['rest_params_grad_norm'] = float(clip_grad.clip_grad_norm_(rest_params, **self.grad_clip))

        # import pdb
        # pdb.set_trace()
        # total_params = [n for n, p in model.named_parameters()]
        # len(total_params) == len(params) + len(rest_params)

        return grad_norm_dict

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        grad_norm_dict = self.clip_grads(runner.model)
        if len(grad_norm_dict) > 0:
            # Add grad norm to the logger
            runner.log_buffer.update(grad_norm_dict,
                                     runner.outputs['num_samples'])
        runner.optimizer.step()


