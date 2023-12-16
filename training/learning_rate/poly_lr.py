"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-08-30 08:50:29
❤LastEditTime: 2022-10-21 14:14:55
❤Github: https://github.com/MilknoCandy
"""
import numpy as np


def poly_scheduler(base_value, final_value, epoch,  max_epochs, niter_per_ep=64, warmup_epochs=5, start_warmup_value=5e-4):
    """_summary_

    Args:
        final_value (_type_): learning rate decays until final_value stops decaying.
        base_value (_type_, optional): initial learning rate.
        max_epochs (_type_, optional): max_epochs.
        niter_per_ep (int, optional): num_batches_per_epoch. Defaults to 64.
        warmup_epochs (int, optional): _description_. Defaults to 5.
        start_warmup_value (_type_, optional): _description_. Defaults to 5e-4.

    Returns:
        _type_: _description_
    """
    warmup_schedule = np.array([])
    # warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs)

    # iters = np.arange(epochs * niter_per_ep - warmup_iters)
    steps = max_epochs - warmup_epochs
    # step_epochs = np.arange(max_epochs - warmup_epochs)
    step_epoch = epoch - warmup_epochs
    # schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * step_epoch / steps))
    schedule = poly_lr(step_epoch, steps, base_value)

    # schedule = np.concatenate((warmup_schedule, schedule))
    # assert len(schedule) == epochs * niter_per_ep
    if epoch < warmup_epochs:
        return warmup_schedule[epoch]
    else:
        return schedule

# from here: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/learning_rate/poly_lr.py
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent