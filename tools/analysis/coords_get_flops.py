import argparse

from mmcv import Config

from mmpose.models import build_posenet
from fvcore.nn import flop_count
from fvcore.nn.jit_handles import Handle, batchnorm_flop_jit, get_shape
from typing import Any, Dict
import torch

import typing
from collections import defaultdict

import tabulate
from torch import nn
from collections import Counter
from numpy import prod

def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.
    Args:
        model: a torch module
    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

def mul_flop_jit(inputs, outputs):
    outputs_shape = get_shape(outputs[0])
    flop = prod(outputs_shape)
    flop_counter = Counter({"mul": flop})
    return flop_counter

def grid_samp_flop_jit(inputs, outputs):
    # input_shape = get_shape(inputs[0])
    outputs_shape = get_shape(outputs[0])
    flop = prod(outputs_shape)
    flop_counter = Counter({"grid_sampler": flop})
    return flop_counter

def layer_norm_flop_jit(inputs, outputs):
    # input_shape = get_shape(inputs[0])
    outputs_shape = get_shape(outputs[0])
    flop = prod(outputs_shape)
    flop_counter = Counter({"layer_norm": flop})
    return flop_counter

custom_ops: Dict[str, Handle] = {
    "aten::batch_norm": batchnorm_flop_jit,
    "aten::mul": mul_flop_jit,
    "aten::grid_sampler": grid_samp_flop_jit,
    "aten::layer_norm": layer_norm_flop_jit
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 192],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_posenet(cfg.model)
    model = model.cuda()
    model.eval()
    model.forward = model.forward_dummy

    dump_input = torch.rand(input_shape).cuda()

    # export_onnx_file = "rsn.onnx"  # 目的ONNX文件名
    #
    # torch.onnx.export(model,
    #                   dump_input,
    #                   export_onnx_file,
    #                   opset_version=11,
    #                   do_constant_folding=True,  # 是否执行常量折叠优化
    #                   input_names=["input"],  # 输入名
    #                   output_names=["output"],  # 输出名
    #                   dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
    #                                 "output": {0: "batch_size"}})


    flop_dict1, _ = flop_count(model, (dump_input,), supported_ops=custom_ops)
    print(flop_dict1)


    max_depth = 10
    count: typing.DefaultDict[str, int] = parameter_count(model)
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x > 1e5:
            return "{:.1f}M".format(x / 1e6)
        if x > 1e2:
            return "{:.1f}K".format(x / 1e3)
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append((indent + name, indent + str(param_shape[name])))
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    print(tab)
    flops = 0
    for i in flop_dict1.values():
        flops += i
    print('total flops = {}'.format(flops))
    print('total param = {}'.format(table[0][1]))

if __name__ == '__main__':
    main()