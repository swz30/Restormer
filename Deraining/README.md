
## Training

1. To download Rain13K training and testing data, run
```
python download_data.py --data train-test
```

2. To train Restormer with default settings, run
```
cd Restormer
./train.sh Deraining/Options/Deraining_Restormer.yml
```

**Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Deraining/Options/Deraining_Restormer.yml](Options/Deraining_Restormer.yml)

## Evaluation

1. Download the pre-trained [model](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u?usp=sharing) and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200, Test2800), run 
```
python download_data.py --data test
```

3. Testing
```
python test.py
```

#### To reproduce PSNR/SSIM scores of Table 1, run

```
evaluate_PSNR_SSIM.m 
```
