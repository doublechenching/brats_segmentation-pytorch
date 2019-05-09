#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 15:56
# @Author  : Eric Ching
from .unet import UNet3D

def build_model(cfg):
    model = UNet3D(cfg.DATASET.INPUT_SHAPE,
                   in_channels=len(cfg.DATASET.USE_MODES),
                   out_channels=3,
                   init_channels=cfg.MODEL.INIT_CHANNELS,
                   p=cfg.MODEL.DROPOUT)

    return model
