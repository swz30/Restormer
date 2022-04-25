## Training

- To download DPDD training data, run
```
python download_data.py --data train
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_dpdd.py 
```

- To train Restormer on **single-image** defocus deblurring task, run
```
cd Restormer
./train.sh Defocus_Deblurring/Options/DefocusDeblur_Single_8bit_Restormer.yml
```

- To train Restormer on **dual-pixel** defocus deblurring task, run
```
cd Restormer
./train.sh Defocus_Deblurring/Options/DefocusDeblur_DualPixel_16bit_Restormer.yml
```

**Note:** The above training scripts use 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [DefocusDeblur_Single_8bit_Restormer.yml](Options/DefocusDeblur_Single_8bit_Restormer.yml) 


## Evaluation

- Download the pre-trained [models](https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4?usp=sharing) and place them in `./pretrained_models/`

- Download test dataset, run
```
python download_data.py --data test
```

- Testing on **single-image** defocus deblurring task, run
```
python test_single_image_defocus_deblur.py --save_images
```

- Testing on **dual-pixel** defocus deblurring task, run
```
python test_dual_pixel_defocus_deblur.py --save_images
```

The above testing scripts will reproduce image quality scores of Table 3 in the paper. 
