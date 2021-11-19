# Restormer: Efficient Transformer for High-Resolution Image Restoration

[Syed Waqas Zamir](https://scholar.google.es/citations?user=WNGPkVQAAAAJ&hl=en), [Aditya Arora](https://adityac8.github.io/), [Salman Khan](https://salman-h-khan.github.io/), [Munawar Hayat](https://scholar.google.com/citations?user=Mx8MbWYAAAAJ&hl=en), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), and [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)

<hr />

> **Abstract:** *Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).* 
<hr />

<strong>Image Deraining</strong>

<img src = "https://i.imgur.com/mMoqYJi.png"> 

<strong>Image Deblurring</strong>

<img src = "https://i.imgur.com/htagDSl.png" width="500"> 

<strong>Dual-Pixel Defocus Deblurring</strong>

<img src = "https://i.imgur.com/sfKnLG2.png"> 

<strong>Gaussian Grayscale Denoising</strong>

<img src = "https://i.imgur.com/4vzV8Qy.png" width="500"> 

<strong>Gaussian Color Denoising</strong>

<img src = "https://i.imgur.com/Sx986Xs.png" width="500"> 

<strong>Real Denoising</strong>

<img src = "https://i.imgur.com/6v5PRxj.png"> 

## Testing codes and pre-trained models will be released on Nov 22nd!

