#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 16:06
# @Author  : Eric Ching
import os
import random
import torch
import warnings
import numpy as np

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('exist path: ', path)