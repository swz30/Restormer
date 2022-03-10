 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.


import numpy as np
import scipy.io as sio
import os
import h5py

def bundle_submissions_raw(submission_folder,session):
    '''
    Bundles submission data for raw denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''

    out_folder = os.path.join(submission_folder, session)
    # out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass

    israw = True
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )

def bundle_submissions_srgb(submission_folder,session):
    '''
    Bundles submission data for sRGB denoising
    
    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, session)
    # out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )



def bundle_submissions_srgb_v1(submission_folder,session):
    '''
    Bundles submission data for sRGB denoising
    
    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, session)
    # out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )