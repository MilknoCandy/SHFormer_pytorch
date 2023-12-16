"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-01 10:51:56
❤LastEditTime: 2022-12-01 10:51:57
❤Github: https://github.com/MilknoCandy
"""
import numpy as np
import torch
# from nnunet/utilities/tensor_utilities.py
def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def mean_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.mean(int(ax))
    return inp