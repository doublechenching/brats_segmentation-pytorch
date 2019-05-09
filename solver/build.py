#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 16:24
# @Author  : Eric Ching
import torch
from .scheduler import PolyLR

def make_optimizer(cfg, model):
    lr = cfg.SOLVER.LEARNING_RATE
    print('initial learning rate is ', lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = PolyLR(optimizer, max_epoch=cfg.SOLVER.NUM_EPOCHS, power=cfg.SOLVER.POWER)

    return optimizer, scheduler
