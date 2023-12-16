"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-01 09:05:39
❤LastEditTime: 2022-12-08 14:52:53
❤Github: https://github.com/MilknoCandy
"""
import torch
def build_optimizer(optimizer_name, model_params, initial_lr, weight_decay):
############################################   
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model_params, initial_lr, betas=(0.5, 0.999), amsgrad=True)
        return optimizer

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model_params, initial_lr)
        return optimizer

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model_params, initial_lr, weight_decay=weight_decay,
                                         momentum=0.99, nesterov=True) 
        return optimizer 