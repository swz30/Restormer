# Evaluation

## Gaussian Image Denoising

- Download the [models](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0?usp=sharing) and place them in `./pretrained_models/`
- Download testsets from [here](https://drive.google.com/drive/folders/1uQ6QSTeY4BqY2PYo3a4BNClJAucDV8-2?usp=sharing) and place it in `./Datasets/test/`

### Grayscale Denoising Testing

##### Blind Denoising: One model to handle various noise levels
- To obtain denoised predictions, run
```
python test_gaussian_gray_denoising.py --model_type blind --sigmas 15,25,50
```

- To reproduce PSNR Table 4 (top super-row), run
```
python evaluate_gaussian_gray_denoising.py --model_type blind --sigmas 15,25,50
```

##### Non-Blind Denoising: Separate models for each noise level
- To obtain denoised predictions, run
```
python test_gaussian_gray_denoising.py --model_type non_blind --sigmas 15,25,50
```

- To reproduce PSNR Table 4 (bottom super-row), run
```
python evaluate_gaussian_gray_denoising.py --model_type non_blind --sigmas 15,25,50
```

### Color Denoising Testing

##### Blind Denoising: One model to handle various noise levels
- To obtain denoised predictions, run
```
python test_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50
```

- To reproduce PSNR Table 5 (top super-row), run
```
python evaluate_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50
```

##### Non-Blind Denoising: Separate models for each noise level
- To obtain denoised predictions, run
```
python test_gaussian_color_denoising.py --model_type non_blind --sigmas 15,25,50
```

- To reproduce PSNR Table 5 (bottom super-row), run
```
python evaluate_gaussian_color_denoising.py --model_type non_blind --sigmas 15,25,50
```

<hr />

## Real Image Denoising

- Download the [model](https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view?usp=sharing) and place it in `./pretrained_models/`


#### Testing on SIDD dataset
- Download SIDD Validation Noisy Data from [here](https://competitions.codalab.org/my/datasets/download/260ed944-21e7-4317-bd66-a6dd42343838) and Ground Truth from [here](https://competitions.codalab.org/my/datasets/download/5349efe0-dfa2-499e-a46f-49c623285ba7) and place them in `./Datasets/test/SIDD/`
- Run
```
python test_real_denoising_SIDD.py --save_images
```
#### Testing on DND dataset
- Download DND Benchmark Data from [here](https://noise.visinf.tu-darmstadt.de/downloads/) and place it in `./Datasets/test/DND/`
- Run
```
python test_real_denoising_DND.py --save_images
```
##### To reproduce PSNR/SSIM scores of Table 6, run
```
evaluate_SIDD.m
```

