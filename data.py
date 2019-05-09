#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 14:18
# @Author  : Eric Ching
import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random

class Brats2018(Dataset):

    def __init__(self, patients_dir, crop_size, modes, train=True):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + "_" + mode + '.nii')
            volume = nib.load(volume_path).get_data()
            if not mode == "seg":
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        wt_volume = seg_volume > 0  # 坏死和无增强的肿瘤区域：1、增强区域（活跃部分）：4、周边水肿区域：2
        tc_volume = np.logical_or(seg_volume == 4, seg_volume == 1)
        et_volume = (seg_volume == 4)
        seg_volume = [wt_volume, tc_volume, et_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float))


    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())

def split_dataset(data_root, nfold=4, seed=42, select=0):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    n_patients = len(patients_dir)
    print(f"total patients: {n_patients}")
    pid_idx = np.arange(n_patients)
    np.random.seed(seed)
    np.random.shuffle(pid_idx)
    n_fold_list = np.split(pid_idx, nfold)
    print(f"split {len(n_fold_list)} folds and every fold have {len(n_fold_list[0])} patients")
    val_patients_list = []
    train_patients_list = []
    for i, fold in enumerate(n_fold_list):
        if i == select:
            for idx in fold:
                val_patients_list.append(patients_dir[idx])
        else:
            for idx in fold:
                train_patients_list.append(patients_dir[idx])
    print(f"train patients: {len(train_patients_list)}, test patients: {len(val_patients_list)}")

    return train_patients_list, val_patients_list

def make_data_loaders(cfg):
    train_list, val_list = split_dataset(cfg.DATASET.DATA_ROOT, cfg.DATASET.NUM_FOLDS, cfg.DATASET.SELECT_FOLD)
    train_ds = Brats2018(train_list, crop_size=cfg.DATASET.INPUT_SHAPE, modes=cfg.DATASET.USE_MODES, train=True)
    val_ds = Brats2018(val_list, crop_size=cfg.DATASET.INPUT_SHAPE, modes=cfg.DATASET.USE_MODES, train=False)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=cfg.DATALOADER.BATCH_SIZE,
                                  num_workers=cfg.DATALOADER.NUM_WORKERS,
                                  pin_memory=True,
                                  shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=cfg.DATALOADER.BATCH_SIZE,
                                  num_workers=cfg.DATALOADER.NUM_WORKERS,
                                  pin_memory=True,
                                  shuffle=False)
    return loaders


if __name__ == "__main__":
    from config import _C as cfg
    train_list, val_list = split_dataset(cfg.DATASET.DATA_ROOT, cfg.DATASET.NUM_FOLDS, cfg.DATASET.SELECT_FOLD)
    train_ds = Brats2018(train_list, crop_size=cfg.DATASET.INPUT_SHAPE, modes=cfg.DATASET.USE_MODES, train=True)
    val_ds = Brats2018(val_list, crop_size=cfg.DATASET.INPUT_SHAPE, modes=cfg.DATASET.USE_MODES, train=False)

    for i in range(len(train_ds)):
        x, y  = train_ds[i]
        volume = (x.numpy()[0] * 255).astype('uint8')
        seg = (np.sum(y.numpy(), axis=0)).astype('uint8')
        volume = nib.Nifti1Image(volume, np.eye(4))
        seg = nib.Nifti1Image(seg, np.eye(4))
        nib.save(volume, 'test'+str(i)+'_volume.nii')
        nib.save(seg, 'test' + str(i) + '_seg.nii')