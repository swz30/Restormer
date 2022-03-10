import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx


src = 'Datasets/Downloads/SIDD'
tar = 'Datasets/train/SIDD'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

files = natsorted(glob(os.path.join(src, '*', '*.PNG')))

lr_files, hr_files = [], []
for file_ in files:
    filename = os.path.split(file_)[-1]
    if 'GT' in filename:
        hr_files.append(file_)
    if 'NOISY' in filename:
        lr_files.append(file_)

files = [(i, j) for i, j in zip(lr_files, hr_files)]

patch_size = 512
overlap = 128
p_max = 0

def save_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.png')
                hr_savename = os.path.join(hr_tar, filename + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)

    else:
        lr_savename = os.path.join(lr_tar, filename + '.png')
        hr_savename = os.path.join(hr_tar, filename + '.png')
        
        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)

from joblib import Parallel, delayed
import multiprocessing
num_cores = 10
Parallel(n_jobs=num_cores)(delayed(save_files)(file_) for file_ in tqdm(files))
