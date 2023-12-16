"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-01 09:25:04
❤LastEditTime: 2022-12-08 14:54:47
❤Github: https://github.com/MilknoCandy
"""
import torch
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler


def build_lr_scheduler(scheduler_name, optimizer, lr_scheduler_patience, lr_scheduler_eps):
##########################################################  
    if scheduler_name == "ReduceLROnPlateau":
        lr_scheduler_ = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=lr_scheduler_patience, verbose=True, threshold=lr_scheduler_eps,threshold_mode="abs"
        )
        return lr_scheduler_

    if scheduler_name == "CosineAnnealingLR":
        lr_scheduler_ = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )
        return lr_scheduler_
        
    if scheduler_name == "StepLR":
        lr_scheduler_ = lr_scheduler.StepLR(
            optimizer, step_size=40, gamma=0.1
        )
        lr_scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=lr_scheduler_)
        return lr_scheduler_warmup

