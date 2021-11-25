
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-test100)](https://paperswithcode.com/sota/single-image-deraining-on-test100?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-rain100h)](https://paperswithcode.com/sota/single-image-deraining-on-rain100h?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-rain100l)](https://paperswithcode.com/sota/single-image-deraining-on-rain100l?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-test1200)](https://paperswithcode.com/sota/single-image-deraining-on-test1200?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-test2800)](https://paperswithcode.com/sota/single-image-deraining-on-test2800?p=restormer-efficient-transformer-for-high)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=restormer-efficient-transformer-for-high)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/image-denoising-on-dnd)](https://paperswithcode.com/sota/image-denoising-on-dnd?p=restormer-efficient-transformer-for-high)

# Restormer: Efficient Transformer for High-Resolution Image Restoration

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), and [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)


**Paper**: https://arxiv.org/abs/2111.09881

**Supplementary**: [pdf](https://drive.google.com/file/d/13AgJJEjYRNrUuMWkiRWPo_19F6xCzgIV/view?usp=sharing)

#### News
- Testing codes and pre-trained models are released!

<hr />

> **Abstract:** *Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).* 
<hr />

## Network Architecture

<img src = "https://i.imgur.com/ulLoEig.png"> 

## Installation

The model is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).

For installing, follow these intructions
```
conda create -n pytorch181 python=3.7
conda activate pytorch181
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
```

## Results

<strong>Image Deraining</strong> comparisons on the Test100, Rain100H, Rain100L, Test1200, and Test2800 testsets. You can download Restormer's predictions from this Google Drive [link](https://drive.google.com/drive/folders/1HcLc6v03q_sP_lRPcl7_NJmlB9f48TWU?usp=sharing)

<img src = "https://i.imgur.com/mMoqYJi.png"> 

<hr />

<strong>Single-Image Motion Deblurring</strong> results. Our Restormer is trained only on the GoPro dataset and directly applied to the HIDE and RealBlur benchmark datasets. You can download Restormer's predictions from this Google Drive [link](https://drive.google.com/drive/folders/1qla3HEOuGapv1hqBwXEMi2USFPB2qmx_?usp=sharing)

<img src = "https://i.imgur.com/htagDSl.png" width="500"> 

<hr />

<strong>Defocus Deblurring</strong> comparisons on the DPDD testset (containing 37 indoor and 39 outdoor scenes). **S**: single-image defocus
deblurring. **D**: dual-pixel defocus deblurring. You can download Restormer's predictions from this Google Drive [link](https://drive.google.com/drive/folders/1V_pLc9CZFe4vN7c4SxtXsXKi2FnLUt98?usp=sharing)

<img src = "https://i.imgur.com/sfKnLG2.png"> 

<hr />

<b>Gaussian Image Denoising</b> comparisons for two categories of methods. Top super row: learning a single model to handle various noise levels. Bottom super row: training a separate model for each noise level. You can download Restormer's predictions from this Google Drive [link](https://drive.google.com/drive/folders/1rEAHUBkA9uCe9Q0AzI5zkYxePSgxYDEG?usp=sharing)

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/4vzV8Qy.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/Sx986Xs.png" width="500"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Grayscale</b></p></td>
    <td><p align="center"><b>Color</b></p></td>
  </tr>
</table>

<b>Real Image Denoising</b> on SIDD and DND datasets. ∗ denotes methods using additional training data. Our Restormer is
trained only on the SIDD images and directly tested on DND. You can download Restormer's predictions from this Google Drive [link](https://drive.google.com/drive/folders/1rEAHUBkA9uCe9Q0AzI5zkYxePSgxYDEG?usp=sharing)

<img src = "https://i.imgur.com/6v5PRxj.png">


## Citation
If you use Restormer, please consider citing:

    @article{Zamir2021Restormer,
        title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
                and Fahad Shahbaz Khan and Ming-Hsuan Yang},
        journal={ArXiv 2111.09881},
        year={2021}
    }

## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org

