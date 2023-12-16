"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-08 21:07:55
❤LastEditTime: 2023-12-16 21:17:58
❤Github: https://github.com/MilknoCandy
"""
import torch
import torch.nn.functional as F
from torch import nn
from network_architecture.simImple import softmax_helper

def weighted_IoU_and_BCE(pred, mask):
    """compute weighted IoU and binary cross entropy loss where boundary pixel is given greater weight

    Args:
        pred (Tensor): prediction
        mask (Tensor): groundtruth
    """
    # from https://github.com/Rayicer/TransFuse
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    # wbce = wbce.sum(dim=(2,3))

    if pred.size(1) == 1:
        pred = torch.sigmoid(pred)
    else:
        pred = softmax_helper(pred)

    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def IoU_loss(pred, gt):
    """compute IoU loss

    Args:
        pred (Tensor): prediction
        gt (Tensor): grountruth

    Returns:
        _type_: _description_
    """
    pred = torch.sigmoid(pred)
    inter = (pred * gt).sum(dim=(2, 3))
    union = (pred + gt).sum(dim=(2, 3))
    iou = 1 - (inter + 1)/(union - inter + 1)   # iou loss
    return iou.mean()

def LIoU_loss(pred, gt):
    """compute LIoU loss

    Args:
        pred (Tensor): prediction
        gt (Tensor): grountruth

    Returns:
        _type_: _description_
    """
    pred = torch.sigmoid(pred)
    inter = (pred * gt).sum(dim=(2, 3))
    union = (pred + gt).sum(dim=(2, 3))
    iou = (inter + 1)/(union - inter + 1)
    return -torch.log(iou).mean()
    