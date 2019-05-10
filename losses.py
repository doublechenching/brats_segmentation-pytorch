#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 16:31
# @Author  : Eric Ching
import torch
from torch.nn import functional as F


def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)


def vae_loss(recon_x, x, mu, logvar):
    loss_dict = {}
    loss_dict['KLD'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_dict['recon_loss'] = F.mse_loss(recon_x, x, reduction='mean')

    return loss_dict

def unet_vae_loss(cfg, batch_pred, batch_x, batch_y, vout, mu, logvar):
    loss_dict = {}
    loss_dict['wt_loss'] = dice_loss(batch_pred[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['tc_loss'] = dice_loss(batch_pred[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_loss'] = dice_loss(batch_pred[:, 2], batch_y[:, 2])  # enhance tumor
    loss_dict.update(vae_loss(vout, batch_x, mu, logvar))
    weight = cfg.MODEL.LOSS_WEIGHT
    loss_dict['loss'] = loss_dict['wt_loss'] + loss_dict['tc_loss'] + loss_dict['et_loss'] + \
                         weight * loss_dict['recon_loss'] + weight * loss_dict['KLD']

    return loss_dict

def get_losses(cfg):
    losses = {}
    losses['vae'] = vae_loss
    losses['dice'] = dice_loss
    losses['dice_vae'] = unet_vae_loss

    return losses
