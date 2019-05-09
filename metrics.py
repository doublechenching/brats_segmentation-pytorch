#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 16:33
# @Author  : Eric Ching
import torch.nn as nn
import numpy as np
from utils.metric.binary import hd

def dice_coef(input, target, threshold=0.5):
    smooth = 1.
    iflat = (input.view(-1) > threshold).float()
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def dice_coef_np(input, target, eps=1e-7):
    input = np.ravel(input)
    target = np.ravel(target)
    intersection = (input * target).sum()

    return (2. * intersection) / (input.sum() + target.sum() + eps)

def hausdorff(batch_pred, batch_y, threshold=0.5):
    """batch size must equal 1"""
    batch_pred = batch_pred.cpu().squeeze().numpy() > threshold
    batch_y = batch_y.cpu().squeeze().numpy()
    metric_dict = {}
    try:
        metric_dict['wt_hd'] = hd(batch_pred[0], batch_y[0])
    except:
        metric_dict['wt_hd'] = 1.0
        print("wt have zero object")
    try:
        metric_dict['tc_hd'] = hd(batch_pred[1], batch_y[1])
    except:
        metric_dict['tc_hd'] = 1.0
        print("tc have zero object")
    try:
        metric_dict['et_hd'] = hd(batch_pred[2], batch_y[2])
    except:
        metric_dict['et_hd'] = 1.0
        print("et have zero object")

    return metric_dict

def get_metrics(cfg):
    metrics = {}
    metrics["mse"] = nn.MSELoss().cuda()
    metrics["hd"] = hausdorff

    return metrics