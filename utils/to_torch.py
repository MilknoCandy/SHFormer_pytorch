"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-30 15:51:46
❤LastEditTime: 2022-11-30 15:51:46
❤Github: https://github.com/MilknoCandy
"""
import torch

def maybe_to_torch(d):
    """Converting data to Tensor type.

    Args:
        d (_type_): _description_

    Returns:
        _type_: Tensor
    """
    if isinstance(d, list):
        # HACK: 和nnUNet不一样
        d = [i if isinstance(i, torch.Tensor) else maybe_to_torch(i) for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    """Putting data on GPU.

    Args:
        data (_type_): _description_
        non_blocking (bool, optional): Controls whether data operations are asynchronous. Defaults to True.
        gpu_id (int, optional): gpu number. Defaults to 0.

    Returns:
        _type_: cuda type
    """
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data