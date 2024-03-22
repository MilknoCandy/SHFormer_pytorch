"""
❤Descripttion: this file is used to compute all metrics
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-14 17:07:26
❤LastEditTime: 2024-03-22 15:04:01
❤Github: https://github.com/MilknoCandy
"""
from monai.metrics.meandice import compute_meandice
from monai.metrics.meaniou import compute_meaniou
from monai.metrics.hausdorff_distance import compute_percent_hausdorff_distance
from monai.metrics.confusion_matrix import (compute_confusion_matrix_metric,
                                            get_confusion_matrix)
from torch import Tensor
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val.append(val if isinstance(val, (int, float)) else val.item())
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AllMterics(object):
    def __init__(self) -> None:
        self.tji = AverageMeter()
        self.jc = AverageMeter()
        self.dice = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.acc = AverageMeter()
        self.prec = AverageMeter()
        # self.assd = AverageMeter()
        # self.hd95 = AverageMeter()
        # self.fps = AverageMeter()
    
    def refresh(self, pred: Tensor, gt: Tensor):
        """All calculations shall be performed on the CPU
        monai使用tensor进行计算, 且需要onehot格式, 且是按照批次进行计算的

        Args:
            pred (Tensor): BxNxHxW
            gt (Tensor): BxNxHxW
        """
        nums = pred.shape[0]
        jc = compute_meaniou(pred, gt)                                              # TP / (TP + FP + FN)
        tji = 0 if jc < 0.65 else jc
        dsc = compute_meandice(pred, gt)                                            # 2TP / (2TP + FP + FN)
        confusion_matrix_ = get_confusion_matrix(pred, gt)
        sen = compute_confusion_matrix_metric("sensitivity", confusion_matrix_)     # TP / (TP + FN)
        spe = compute_confusion_matrix_metric("specificity", confusion_matrix_)     # TN / (TN + FP)
        acc = compute_confusion_matrix_metric("accuracy", confusion_matrix_)        # (TP + TN) / (TP + FP + TN + FN)
        prec = compute_confusion_matrix_metric("precision", confusion_matrix_)      # TP / (TP + FP)
        if torch.isnan(prec):
            prec = 0.
        # assd_ = assd(pred.numpy(), gt.numpy())
        # hd95_ = hd95(pred.numpy(), gt.numpy())
        self.tji.update(tji, nums)
        self.jc.update(jc, nums)
        self.dice.update(dsc, nums)
        self.sen.update(sen, nums)
        self.spe.update(spe, nums)
        self.acc.update(acc, nums)
        self.prec.update(prec, nums)
        # self.assd.update(assd_, nums)
        # self.hd95.update(hd95_, nums)
        