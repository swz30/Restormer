# Image Denoising
- [Gaussian Image Denoising](#gaussian-image-denoising)
  * [Training](#training)
  * [Evaluation](#evaluation)
      - [Grayscale blind image denoising testing](#grayscale-blind-image-denoising-testing)
      - [Grayscale non-blind image denoising testing](#grayscale-non-blind-image-denoising-testing)
      - [Color blind image denoising testing](#color-blind-image-denoising-testing)
      - [Color non-blind image denoising testing](#color-non-blind-image-denoising-testing)
- [Real Image Denoising](#real-image-denoising)
  * [Training](#training-1)
  * [Evaluation](#evaluation-1)
      - [Testing on SIDD dataset](#testing-on-sidd-dataset)
      - [Testing on DND dataset](#testing-on-dnd-dataset)

# Gaussian Image Denoising

- **Blind Denoising:** One model to handle various noise levels
- **Non-Blind Denoising:** Separate models for each noise level

## Training

- Download training (DIV2K, Flickr2K, WED, BSD) and testing datasets, run
```
python download_data.py --data train-test --noise gaussian
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_dfwb.py 
```

- Train Restormer for **grayscale blind** image denoising, run
```
cd Restormer
./train.sh Denoising/Options/GaussianGrayDenoising_Restormer.yml
```

- Train Restormer for **grayscale non-blind** image denoising, run
```
cd Restormer
./train.sh Denoising/Options/GaussianGrayDenoising_RestormerSigma15.yml
./train.sh Denoising/Options/GaussianGrayDenoising_RestormerSigma25.yml
./train.sh Denoising/Options/GaussianGrayDenoising_RestormerSigma50.yml
```

- Train Restormer for **color blind** image denoising, run
```
cd Restormer
./train.sh Denoising/Options/GaussianColorDenoising_Restormer.yml
```

- Train Restormer for **color non-blind** image denoising, run
```
cd Restormer
./train.sh Denoising/Options/GaussianColorDenoising_RestormerSigma15.yml
./train.sh Denoising/Options/GaussianColorDenoising_RestormerSigma25.yml
./train.sh Denoising/Options/GaussianColorDenoising_RestormerSigma50.yml
```

**Note:** The above training scripts use 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and the yaml file corresponding to each task (e.g., [Denoising/Options/GaussianGrayDenoising_Restormer.yml](Options/GaussianGrayDenoising_Restormer.yml))

## Evaluation

- Download the pre-trained [models](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0?usp=sharing) and place them in `./pretrained_models/`

- Download testsets (Set12, BSD68, CBSD68, Kodak, McMaster, Urban100), run 
```
python download_data.py --data test --noise gaussian
```

#### Grayscale blind image denoising testing

- To obtain denoised predictions, run
```
python test_gaussian_gray_denoising.py --model_type blind --sigmas 15,25,50
```

- To reproduce PSNR Table 4 (top super-row), run
```
python evaluate_gaussian_gray_denoising.py --model_type blind --sigmas 15,25,50
```

#### Grayscale non-blind image denoising testing

- To obtain denoised predictions, run
```
python test_gaussian_gray_denoising.py --model_type non_blind --sigmas 15,25,50
```

- To reproduce PSNR Table 4 (bottom super-row), run
```
python evaluate_gaussian_gray_denoising.py --model_type non_blind --sigmas 15,25,50
```

#### Color blind image denoising testing

- To obtain denoised predictions, run
```
python test_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50
```

- To reproduce PSNR Table 5 (top super-row), run
```
python evaluate_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50
```

#### Color non-blind image denoising testing

- To obtain denoised predictions, run
```
python test_gaussian_color_denoising.py --model_type non_blind --sigmas 15,25,50
```

- To reproduce PSNR Table 5 (bottom super-row), run
```
python evaluate_gaussian_color_denoising.py --model_type non_blind --sigmas 15,25,50
```

<hr />

# Real Image Denoising

## Training

- Download SIDD training data, run
```
python download_data.py --data train --noise real
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_sidd.py 
```

- Train Restormer
```
cd Restormer
./train.sh Denoising/Options/RealDenoising_Restormer.yml
```

**Note:** This training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Denoising/Options/RealDenoising_Restormer.yml](Options/RealDenoising_Restormer.yml)

## Evaluation

- Download the pre-trained [model](https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view?usp=sharing) and place it in `./pretrained_models/`

#### Testing on SIDD dataset

- Download SIDD validation data, run 
```
python download_data.py --noise real --data test --dataset SIDD
```

- To obtain denoised results, run
```
python test_real_denoising_sidd.py --save_images
```

- To reproduce PSNR/SSIM scores on SIDD data (Table 6), run
```
evaluate_sidd.m
```

#### Testing on DND dataset

- Download the DND benchmark data, run 
```
python download_data.py --noise real --data test --dataset DND
```

- To obtain denoised results, run
```
python test_real_denoising_dnd.py --save_images
```

- To reproduce PSNR/SSIM scores (Table 6), upload the results to the DND benchmark website.
