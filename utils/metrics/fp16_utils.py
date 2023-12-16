"""
❤Description: from mmcv.fp16_utils
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-12-03 20:45:59
❤LastEditTime: 2023-12-03 20:48:19
❤FilePath: fp16_utils
❤Github: https://github.com/MilknoCandy
"""
from collections import abc

import numpy as np
import torch
import torch.nn as nn
from mmcv.utils import TORCH_VERSION, digit_version


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs
    
def wrap_fp16_model(model):
    """Wrap the FP32 model to FP16.

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    For PyTorch >= 1.6, this function will
    1. Set fp16 flag inside the model to True.

    Otherwise:
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.
    3. Set `fp16_enabled` flag inside the model to True.

    Args:
        model (nn.Module): Model in FP32.
    """
    if (TORCH_VERSION == 'parrots'
            or digit_version(TORCH_VERSION) < digit_version('1.6.0')):
        # convert model to fp16
        model.half()
        # patch the normalization layers to make it work in fp32 mode
        patch_norm_fp32(model)
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True


def patch_norm_fp32(module):
    """Recursively convert normalization layers from FP16 to FP32.

    Args:
        module (nn.Module): The modules to be converted in FP16.

    Returns:
        nn.Module: The converted module, the normalization layers have been
            converted to FP32.
    """
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
        if isinstance(module, nn.GroupNorm) or torch.__version__ < '1.3':
            module.forward = patch_forward_method(module.forward, torch.half,
                                                  torch.float)
    for child in module.children():
        patch_norm_fp32(child)
    return module

def patch_forward_method(func, src_type, dst_type, convert_output=True):
    """Patch the forward method of a module.

    Args:
        func (callable): The original forward method.
        src_type (torch.dtype): Type of input arguments to be converted from.
        dst_type (torch.dtype): Type of input arguments to be converted to.
        convert_output (bool): Whether to convert the output back to src_type.

    Returns:
        callable: The patched forward method.
    """

    def new_forward(*args, **kwargs):
        output = func(*cast_tensor_type(args, src_type, dst_type),
                      **cast_tensor_type(kwargs, src_type, dst_type))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type)
        return output

    return new_forward