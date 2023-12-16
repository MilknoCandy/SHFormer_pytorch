"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-08 21:58:52
❤LastEditTime: 2022-12-08 21:58:53
❤Github: https://github.com/MilknoCandy
"""
import numpy as np
import torch


def mask2onehot(mask, num_classes):
    """    
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    Args:
        mask (_type_): (1, H, W)
        num_classes (_type_): label's number

    Returns:
        _type_: _description_
    """
    # _mask = [mask == i for i in range(num_classes)]
    # return np.array(_mask).astype(np.uint8)
    mask_onehot = torch.zeros((num_classes, *mask.shape[1:]), dtype=mask.dtype)
    for i in range(num_classes):
        mask_onehot[i][mask[0] == i] = 1
    return mask_onehot

def mask2onehot_batch(mask, num_classes):
    """    
    Converts a segmentation mask (B,H,W) to (B,K,H,W) where the last dim is a one
    hot encoding vector

    Args:
        mask (_type_): (B, H, W)
        num_classes (_type_): label's number

    Returns:
        _type_: _description_
    """
    B, H, W = mask.shape
    mask_onehot = torch.zeros((B, num_classes, H, W), dtype=mask.dtype, device=mask.device)
    for i in range(B):
        mask_onehot[i] = mask2onehot(mask[i:i+1], num_classes)
    
    return mask_onehot.float()

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    # _mask = np.argmax(mask, axis=0).astype(np.uint8)
    _mask = torch.argmax(mask, axis=0)
    return _mask