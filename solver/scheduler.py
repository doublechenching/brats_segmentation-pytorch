#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/3/28 13:57
# @Author  : Eric Ching
from torch.optim import lr_scheduler

class PolyLR(lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epoch) ** self.power
                for base_lr in self.base_lrs]