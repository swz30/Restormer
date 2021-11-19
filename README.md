# Restormer: Efficient Transformer for High-Resolution Image Restoration

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), and [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)


**Paper**: https://arxiv.org/abs/2111.09881


## Testing codes and pre-trained models will be released on Nov 22nd!

<hr />

> **Abstract:** *Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).* 
<hr />

## Network Architecture

<img src = "https://i.imgur.com/ulLoEig.png"> 

## Results

<strong>Image Deraining</strong> comparisons on the Test100, Rain100H, Rain100L, Test1200, and Test2800 testsets.

<img src = "https://i.imgur.com/mMoqYJi.png"> 

<hr />

<strong>Single-Image Motion Deblurring</strong> results. Our Restormer is trained only on the GoPro dataset and directly applied to the HIDE and RealBlur benchmark datasets.

<img src = "https://i.imgur.com/htagDSl.png" width="500"> 

<hr />

<strong>Defocus Deblurring</strong> comparisons on the DPDD testset (containing 37 indoor and 39 outdoor scenes). **S**: single-image defocus
deblurring. **D**: dual-pixel defocus deblurring.

<img src = "https://i.imgur.com/sfKnLG2.png"> 

<hr />

<b>Gaussian Image Denoising</b> comparisons for two categories of methods. Top super row: learning a single model to handle various noise levels. Bottom super row: training a separate model for each noise level.

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
trained only on the SIDD images and directly tested on DND.

<img src = "https://i.imgur.com/6v5PRxj.png">

@article{ba2016layer,
  title={Layer normalization},
  author={Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E},
  journal={arXiv:1607.06450},
  year={2016}
}


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

