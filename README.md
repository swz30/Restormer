
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-test100)](https://paperswithcode.com/sota/single-image-deraining-on-test100?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-rain100h)](https://paperswithcode.com/sota/single-image-deraining-on-rain100h?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-rain100l)](https://paperswithcode.com/sota/single-image-deraining-on-rain100l?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-test1200)](https://paperswithcode.com/sota/single-image-deraining-on-test1200?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/single-image-deraining-on-test2800)](https://paperswithcode.com/sota/single-image-deraining-on-test2800?p=restormer-efficient-transformer-for-high)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/image-denoising-on-dnd)](https://paperswithcode.com/sota/image-denoising-on-dnd?p=restormer-efficient-transformer-for-high)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=restormer-efficient-transformer-for-high)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/restormer-efficient-transformer-for-high/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=restormer-efficient-transformer-for-high)



# Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022 -- Oral)

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), and [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.09881)
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)](https://drive.google.com/file/d/1oKGON8vG4uDWMmZKqHeTMnFowhOubifK/view?usp=sharing)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://www.youtube.com/watch?v=3mqu6N4_0pY&t)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/19wKhnQtr3mcD6IsLj0ZFSwCgIRKUkDQJ/view?usp=sharing)
[![Summary](https://img.shields.io/badge/Summary-Slide-87CEEB)](https://drive.google.com/file/d/1wyKAMLzJpDqHiF6AMsmnmGQC241GyT8q/view?usp=sharing)


#### News
- **April 4, 2022:** Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/swzamir/Restormer)
- **March 30, 2022:** Added Colab Demo. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing)
- **March 29, 2022:** Restormer is selected for an ORAL presentation at CVPR 2022 :dizzy:
- **March 10, 2022:** Training codes are released :fire:
- **March 3, 2022:** Paper accepted at CVPR 2022 :tada: 
- **Nov 21, 2021:** Testing codes and pre-trained models are released!

<hr />

> **Abstract:** *Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).* 
<hr />

## Network Architecture

<img src = "https://i.imgur.com/ulLoEig.png"> 

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.

## Demo

To test the pre-trained Restormer models of [Deraining](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u), [Motion Deblurring](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK), [Defocus Deblurring](https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4?usp=sharing), and [Denoising](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0) on your own images, you can either use Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing), or command line as following
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```
Example usage to perform Defocus Deblurring on a directory of images:
```
python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
```
Example usage to perform Defocus Deblurring on an image directly:
```
python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'
```

## Training and Evaluation

Training and Testing instructions for Deraining, Motion Deblurring, Defocus Deblurring, and Denoising are provided in their respective directories. Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="left">Task</th>
    <th align="center">Training Instructions</th>
    <th align="center">Testing Instructions</th>
    <th align="center">Restormer's Visual Results</th>
  </tr>
  <tr>
    <td align="left">Deraining</td>
    <td align="center"><a href="Deraining/README.md#training">Link</a></td>
    <td align="center"><a href="Deraining/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1HcLc6v03q_sP_lRPcl7_NJmlB9f48TWU?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Motion Deblurring</td>
    <td align="center"><a href="Motion_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Motion_Deblurring/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1qla3HEOuGapv1hqBwXEMi2USFPB2qmx_?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Defocus Deblurring</td>
    <td align="center"><a href="Defocus_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Defocus_Deblurring/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1V_pLc9CZFe4vN7c4SxtXsXKi2FnLUt98?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Gaussian Denoising</td>
    <td align="center"><a href="Denoising/README.md#training">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1rEAHUBkA9uCe9Q0AzI5zkYxePSgxYDEG?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Real Denoising</td>
    <td align="center"><a href="Denoising/README.md#training-1">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation-1">Link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/1CsEiN6R0hlmEoSTyy48nnhfF06P5aRR7/view?usp=sharing">Download</a></td>
  </tr>
</table>

## Results
Experiments are performed for different image processing tasks including, image deraining, single-image motion deblurring, defocus deblurring (both on single image and dual pixel data), and image denoising (both on Gaussian and real data). 

<details>
<summary><strong>Image Deraining</strong> (click to expand) </summary>

<img src = "https://i.imgur.com/mMoqYJi.png"> 
</details>

<details>
<summary><strong>Single-Image Motion Deblurring</strong> (click to expand) </summary>

<p align="center"><img src = "https://i.imgur.com/htagDSl.png" width="400"></p></details>

<details>
<summary><strong>Defocus Deblurring</strong> (click to expand) </summary>

S: single-image defocus deblurring.
D: dual-pixel defocus deblurring.

<img src = "https://i.imgur.com/sfKnLG2.png"> 
</details>


<details>
<summary><strong>Gaussian Image Denoising</strong> (click to expand) </summary>

Top super-row: learning a single model to handle various noise levels.
Bottom super-row: training a separate model for each noise level.

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
</details>

<details>
<summary><strong>Real Image Denoising</strong> (click to expand) </summary>

<img src = "https://i.imgur.com/6v5PRxj.png">
</details>

## Citation
If you use Restormer, please consider citing:

    @inproceedings{Zamir2021Restormer,
        title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
                and Fahad Shahbaz Khan and Ming-Hsuan Yang},
        booktitle={CVPR},
        year={2022}
    }


## Contact
Should you have any question, please contact waqas.zamir@inceptioniai.org


**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox and [HINet](https://github.com/megvii-model/HINet). 

## Our Related Works
- Learning Enriched Features for Fast Image Restoration and Enhancement, TPAMI 2022. [Paper](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/) | [Code](https://github.com/swz30/MIRNetv2)
- Multi-Stage Progressive Image Restoration, CVPR 2021. [Paper](https://arxiv.org/abs/2102.02808) | [Code](https://github.com/swz30/MPRNet)
- Learning Enriched Features for Real Image Restoration and Enhancement, ECCV 2020. [Paper](https://arxiv.org/abs/2003.06792) | [Code](https://github.com/swz30/MIRNet)
- CycleISP: Real Image Restoration via Improved Data Synthesis, CVPR 2020. [Paper](https://arxiv.org/abs/2003.07761) | [Code](https://github.com/swz30/CycleISP)
