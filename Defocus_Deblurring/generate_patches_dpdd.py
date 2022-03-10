##### Data preparation file for training Restormer on the DPDD Dataset ########

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from copy import deepcopy

from joblib import Parallel, delayed
import multiprocessing
from pdb import set_trace as stx

def shapness_measure(img_temp,kernel_size):
    conv_x = cv2.Sobel(img_temp,cv2.CV_64F,1,0,ksize=kernel_size)
    conv_y = cv2.Sobel(img_temp,cv2.CV_64F,0,1,ksize=kernel_size)
    temp_arr_x=deepcopy(conv_x*conv_x)
    temp_arr_y=deepcopy(conv_y*conv_y)
    temp_sum_x_y=temp_arr_x+temp_arr_y
    temp_sum_x_y=np.sqrt(temp_sum_x_y)
    return np.sum(temp_sum_x_y)

def filter_patch_sharpness(patches_src_c_temp, patches_trg_c_temp, patches_src_l_temp, patches_src_r_temp):
    patches_src_c, patches_trg_c, patches_src_l, patches_src_r = [], [], [], []
    fitnessVal_3=[]
    fitnessVal_7=[]
    fitnessVal_11=[]
    fitnessVal_15=[]
    num_of_img_patches=len(patches_trg_c_temp)
    for i in range(num_of_img_patches):
        fitnessVal_3.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY),3))
        fitnessVal_7.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY),7))
        fitnessVal_11.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY),11))
        fitnessVal_15.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY),15))
    fitnessVal_3=np.asarray(fitnessVal_3)
    fitnessVal_7=np.asarray(fitnessVal_7)
    fitnessVal_11=np.asarray(fitnessVal_11)
    fitnessVal_15=np.asarray(fitnessVal_15)
    fitnessVal_3=(fitnessVal_3-np.min(fitnessVal_3))/np.max((fitnessVal_3-np.min(fitnessVal_3)))
    fitnessVal_7=(fitnessVal_7-np.min(fitnessVal_7))/np.max((fitnessVal_7-np.min(fitnessVal_7)))
    fitnessVal_11=(fitnessVal_11-np.min(fitnessVal_11))/np.max((fitnessVal_11-np.min(fitnessVal_11)))
    fitnessVal_15=(fitnessVal_15-np.min(fitnessVal_15))/np.max((fitnessVal_15-np.min(fitnessVal_15)))
    fitnessVal_all=fitnessVal_3*fitnessVal_7*fitnessVal_11*fitnessVal_15
    
    to_remove_patches_number=int(to_remove_ratio*num_of_img_patches)
    
    for itr in range(to_remove_patches_number):
        minArrInd=np.argmin(fitnessVal_all)
        fitnessVal_all[minArrInd]=2
    for itr in range(num_of_img_patches):
        if fitnessVal_all[itr]!=2:
            patches_src_c.append(patches_src_c_temp[itr])
            patches_trg_c.append(patches_trg_c_temp[itr])
            patches_src_l.append(patches_src_l_temp[itr])
            patches_src_r.append(patches_src_r_temp[itr])
    
    return patches_src_c, patches_trg_c, patches_src_l, patches_src_r

def slice_stride(_img_src_c, _img_trg_c, _img_src_l, _img_src_r):
    coordinates_list=[]
    coordinates_list.append([0,0,0,0])
    patches_src_c_temp, patches_trg_c_temp, patches_src_l_temp, patches_src_r_temp = [], [], [], []
    for r in range(0,_img_src_c.shape[0],stride[0]):
        for c in range(0,_img_src_c.shape[1],stride[1]):
            if (r+patch_size[0]) <= _img_src_c.shape[0] and (c+patch_size[1]) <= _img_src_c.shape[1]:
                patches_src_c_temp.append(_img_src_c[r:r+patch_size[0],c:c+patch_size[1]])
                patches_trg_c_temp.append(_img_trg_c[r:r+patch_size[0],c:c+patch_size[1]])
                patches_src_l_temp.append(_img_src_l[r:r+patch_size[0],c:c+patch_size[1]])
                patches_src_r_temp.append(_img_src_r[r:r+patch_size[0],c:c+patch_size[1]])

            elif (r+patch_size[0]) <= _img_src_c.shape[0] and not ([r,r+patch_size[0],_img_src_c.shape[1]-patch_size[1],_img_src_c.shape[1]] in coordinates_list):
                patches_src_c_temp.append(_img_src_c[r:r+patch_size[0],_img_src_c.shape[1]-patch_size[1]:_img_src_c.shape[1]])
                patches_trg_c_temp.append(_img_trg_c[r:r+patch_size[0],_img_trg_c.shape[1]-patch_size[1]:_img_trg_c.shape[1]])
                patches_src_l_temp.append(_img_src_l[r:r+patch_size[0],_img_src_l.shape[1]-patch_size[1]:_img_src_l.shape[1]])
                patches_src_r_temp.append(_img_src_r[r:r+patch_size[0],_img_src_r.shape[1]-patch_size[1]:_img_src_r.shape[1]])
                coordinates_list.append([r,r+patch_size[0],_img_src_c.shape[1]-patch_size[1],_img_src_c.shape[1]])
                
            elif (c+patch_size[1]) <= _img_src_c.shape[1] and not ([_img_src_c.shape[0]-patch_size[0],_img_src_c.shape[0],c,c+patch_size[1]] in coordinates_list):
                patches_src_c_temp.append(_img_src_c[_img_src_c.shape[0]-patch_size[0]:_img_src_c.shape[0],c:c+patch_size[1]])
                patches_trg_c_temp.append(_img_trg_c[_img_trg_c.shape[0]-patch_size[0]:_img_trg_c.shape[0],c:c+patch_size[1]])
                patches_src_l_temp.append(_img_src_l[_img_src_l.shape[0]-patch_size[0]:_img_src_l.shape[0],c:c+patch_size[1]])
                patches_src_r_temp.append(_img_src_r[_img_src_r.shape[0]-patch_size[0]:_img_src_r.shape[0],c:c+patch_size[1]])
                coordinates_list.append([_img_src_c.shape[0]-patch_size[0],_img_src_c.shape[0],c,c+patch_size[1]])
                
            elif not ([_img_src_c.shape[0]-patch_size[0],_img_src_c.shape[0],_img_src_c.shape[1]-patch_size[1],_img_src_c.shape[1]] in coordinates_list):
                patches_src_c_temp.append(_img_src_c[_img_src_c.shape[0]-patch_size[0]:_img_src_c.shape[0],_img_src_c.shape[1]-patch_size[1]:_img_src_c.shape[1]])
                patches_trg_c_temp.append(_img_trg_c[_img_trg_c.shape[0]-patch_size[0]:_img_trg_c.shape[0],_img_trg_c.shape[1]-patch_size[1]:_img_trg_c.shape[1]])
                patches_src_l_temp.append(_img_src_l[_img_src_l.shape[0]-patch_size[0]:_img_src_l.shape[0],_img_src_l.shape[1]-patch_size[1]:_img_src_l.shape[1]])
                patches_src_r_temp.append(_img_src_r[_img_src_r.shape[0]-patch_size[0]:_img_src_r.shape[0],_img_src_r.shape[1]-patch_size[1]:_img_src_r.shape[1]])
                coordinates_list.append([_img_src_c.shape[0]-patch_size[0],_img_src_c.shape[0],_img_src_c.shape[1]-patch_size[1],_img_src_c.shape[1]])

    return patches_src_c_temp, patches_trg_c_temp, patches_src_l_temp, patches_src_r_temp

def train_files(file_):
    lrL_file, lrR_file, lrC_file, hrC_file = file_
    filename = os.path.splitext(os.path.split(lrC_file)[-1])[0]
    lrL_img = cv2.imread(lrL_file, -1)
    lrR_img = cv2.imread(lrR_file, -1)
    lrC_img = cv2.imread(lrC_file, -1)
    hrC_img = cv2.imread(hrC_file, -1)

    lrC_patches, hrC_patches, lrL_patches, lrR_patches = slice_stride(lrC_img, hrC_img, lrL_img, lrR_img)
    lrC_patches, hrC_patches, lrL_patches, lrR_patches = filter_patch_sharpness(lrC_patches, hrC_patches, lrL_patches, lrR_patches)
    num_patch = 0
    for lrC_patch, hrC_patch, lrL_patch, lrR_patch in zip(lrC_patches, hrC_patches, lrL_patches, lrR_patches):
        num_patch += 1
                
        lrL_savename = os.path.join(lrL_tar, filename + '-' + str(num_patch) + '.png')
        lrR_savename = os.path.join(lrR_tar, filename + '-' + str(num_patch) + '.png')
        lrC_savename = os.path.join(lrC_tar, filename + '-' + str(num_patch) + '.png')
        hrC_savename = os.path.join(hrC_tar, filename + '-' + str(num_patch) + '.png')
        
        cv2.imwrite(lrL_savename, lrL_patch)
        cv2.imwrite(lrR_savename, lrR_patch)
        cv2.imwrite(lrC_savename, lrC_patch)
        cv2.imwrite(hrC_savename, hrC_patch)

def val_files(file_):
    lrL_file, lrR_file, lrC_file, hrC_file = file_
    filename = os.path.splitext(os.path.split(lrC_file)[-1])[0]

    lrL_savename = os.path.join(lrL_tar, filename + '.png')
    lrR_savename = os.path.join(lrR_tar, filename + '.png')
    lrC_savename = os.path.join(lrC_tar, filename + '.png')
    hrC_savename = os.path.join(hrC_tar, filename + '.png')

    lrL_img = cv2.imread(lrL_file, -1)
    lrR_img = cv2.imread(lrR_file, -1)
    lrC_img = cv2.imread(lrC_file, -1)
    hrC_img = cv2.imread(hrC_file, -1)

    w, h = lrC_img.shape[:2]

    i = (w-val_patch_size)//2
    j = (h-val_patch_size)//2
                
    lrL_patch = lrL_img[i:i+val_patch_size, j:j+val_patch_size,:]
    lrR_patch = lrR_img[i:i+val_patch_size, j:j+val_patch_size,:]
    lrC_patch = lrC_img[i:i+val_patch_size, j:j+val_patch_size,:]
    hrC_patch = hrC_img[i:i+val_patch_size, j:j+val_patch_size,:]
                
    cv2.imwrite(lrL_savename, lrL_patch)
    cv2.imwrite(lrR_savename, lrR_patch)
    cv2.imwrite(lrC_savename, lrC_patch)
    cv2.imwrite(hrC_savename, hrC_patch)


############ Prepare Training data ####################
num_cores = 10
src = 'Datasets/Downloads/DPDD/train'
tar = 'Datasets/train/DPDD'

lrL_tar = os.path.join(tar, 'inputL_crops')
lrR_tar = os.path.join(tar, 'inputR_crops')
lrC_tar = os.path.join(tar, 'inputC_crops')
hrC_tar = os.path.join(tar, 'target_crops')

os.makedirs(lrL_tar, exist_ok=True)
os.makedirs(lrR_tar, exist_ok=True)
os.makedirs(lrC_tar, exist_ok=True)
os.makedirs(hrC_tar, exist_ok=True)

lrL_files = natsorted(glob(os.path.join(src, 'train', 'inputL', '*.png')))
lrR_files = natsorted(glob(os.path.join(src, 'train', 'inputR', '*.png')))
lrC_files = natsorted(glob(os.path.join(src, 'train', 'inputC', '*.png')))
hrC_files = natsorted(glob(os.path.join(src, 'train', 'target', '*.png')))

files = [(i, j, k, l) for i, j, k, l in zip(lrL_files, lrR_files, lrC_files, hrC_files)]

patch_size = [512, 512]
stride = [204, 204]
p_max = 0
to_remove_ratio = 0.3

Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))


############ Prepare validation data ####################
val_patch_size = 256
src = 'Datasets/Downloads/DPDD/val'
tar = 'Datasets/val/DPDD'

lrL_tar = os.path.join(tar, 'inputL_crops')
lrR_tar = os.path.join(tar, 'inputR_crops')
lrC_tar = os.path.join(tar, 'inputC_crops')
hrC_tar = os.path.join(tar, 'target_crops')

os.makedirs(lrL_tar, exist_ok=True)
os.makedirs(lrR_tar, exist_ok=True)
os.makedirs(lrC_tar, exist_ok=True)
os.makedirs(hrC_tar, exist_ok=True)

lrL_files = natsorted(glob(os.path.join(src, 'val', 'inputL', '*.png')))
lrR_files = natsorted(glob(os.path.join(src, 'val', 'inputR', '*.png')))
lrC_files = natsorted(glob(os.path.join(src, 'val', 'inputC', '*.png')))
hrC_files = natsorted(glob(os.path.join(src, 'val', 'target', '*.png')))

files = [(i, j, k, l) for i, j, k, l in zip(lrL_files, lrR_files, lrC_files, hrC_files)]

Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
