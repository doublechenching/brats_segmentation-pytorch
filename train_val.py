#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 16:14
# @Author  : Eric Ching
from config import _C as cfg
from data import make_data_loaders
from models import build_model
from utils import init_env, mkdir
from solver import make_optimizer
import os
import torch
from utils.logger import setup_logger
from utils.metric_logger import MetricLogger
import logging
import time
from losses import get_losses
from metrics import get_metrics
import shutil
import nibabel as nib
import numpy as np

def save_sample(batch_pred, batch_x, batch_y, epoch, batch_id):
    def get_mask(seg_volume):
        seg_volume = seg_volume.cpu().numpy()
        seg_volume = np.squeeze(seg_volume)
        wt_pred = seg_volume[0]
        tc_pred = seg_volume[1]
        et_pred = seg_volume[2]
        mask = np.zeros_like(wt_pred)
        mask[wt_pred > 0.5] = 2
        mask[tc_pred > 0.5] = 1
        mask[et_pred > 0.5] = 4
        mask = mask.astype("uint8")
        mask_nii = nib.Nifti1Image(mask, np.eye(4))
        return mask_nii

    volume = batch_x[:, 0].cpu().numpy()
    volume = volume.astype("uint8")
    volume_nii = nib.Nifti1Image(volume, np.eye(4))
    log_dir = os.path.join(cfg.LOG_DIR, cfg.TASK_NAME, 'epoch'+str(epoch))
    mkdir(log_dir)
    nib.save(volume_nii, os.path.join(log_dir, 'batch'+str(batch_id)+'_volume.nii'))
    pred_nii = get_mask(batch_pred)
    gt_nii = get_mask(batch_y)
    nib.save(pred_nii, os.path.join(log_dir, 'batch' + str(batch_id) + '_pred.nii'))
    nib.save(gt_nii, os.path.join(log_dir, 'batch' + str(batch_id) + '_gt.nii'))

def train_val(model, loaders, optimizer, scheduler, losses, metrics=None):
    n_epochs = cfg.SOLVER.NUM_EPOCHS
    end = time.time()
    best_dice = 0.0
    for epoch in range(n_epochs):
        scheduler.step()
        for phase in ['train', 'eval']:
            meters = MetricLogger(delimiter=" ")
            loader = loaders[phase]
            getattr(model, phase)()
            logger = logging.getLogger(phase)
            total = len(loader)
            for batch_id, (batch_x, batch_y) in enumerate(loader):
                batch_x, batch_y = batch_x.cuda(async=True), batch_y.cuda(async=True)
                with torch.set_grad_enabled(phase == 'train'):
                    output, vout, mu, logvar = model(batch_x)
                    loss_dict = losses['dice_vae'](cfg, output, batch_x, batch_y, vout, mu, logvar)
                meters.update(**loss_dict)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss_dict['loss'].backward()
                    optimizer.step()
                else:
                    if metrics and (epoch + 1) % 20 == 0:
                        with torch.no_grad():
                            hausdorff = metrics['hd']
                            metric_dict = hausdorff(output, batch_y)
                            meters.update(**metric_dict)
                        save_sample(output, batch_x, batch_y, epoch, batch_id)
                logger.info(meters.delimiter.join([f"Epoch: {epoch}, Batch:{batch_id}/{total}",
                                                   f"{str(meters)}",
                                                   f"Time: {time.time() - end: .3f}"
                                                   ]))
                end = time.time()

            if phase == 'eval':
                dice = 1 - (meters.wt_loss.global_avg + meters.tc_loss.global_avg + meters.et_loss.global_avg) / 3
                state = {}
                state['model'] = model.state_dict()
                state['optimizer'] = optimizer.state_dict()
                file_name = os.path.join(cfg.LOG_DIR, cfg.TASK_NAME, 'epoch' + str(epoch) + '.pt')
                torch.save(state, file_name)
                if dice > best_dice:
                    best_dice = dice
                    shutil.copyfile(file_name, os.path.join(cfg.LOG_DIR, cfg.TASK_NAME, 'best_model.pth'))

    return model

def main():
    init_env('1')
    loaders = make_data_loaders(cfg)
    model = build_model(cfg)
    model = model.cuda()
    task_name = 'base_unet'
    log_dir = os.path.join(cfg.LOG_DIR, task_name)
    cfg.TASK_NAME = task_name
    mkdir(log_dir)
    logger = setup_logger('train', log_dir, filename='train.log')
    logger.info(cfg)
    logger = setup_logger('eval', log_dir, filename='eval.log')
    optimizer, scheduler = make_optimizer(cfg, model)
    metrics = get_metrics(cfg)
    losses = get_losses(cfg)
    train_val(model, loaders, optimizer, scheduler, losses, metrics)

if __name__ == "__main__":
    main()