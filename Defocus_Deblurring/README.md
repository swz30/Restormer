## Evaluation

1. Download the [models](https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4?usp=sharing) and place them in `./pretrained_models/`

2. Download test dataset from [here](https://drive.google.com/drive/folders/1P4W5DcmEiwaClZhOfFquPMNbk7Lrq7H9?usp=sharing) and place it in `./Datasets/test/`

##### Testing on Single Image
- Run
```
python test_single_image_defocus_deblur.py --save_images
```
##### Testing on Dual-Pixel Data
- Run
```
python test_dual_pixel_defocus_deblur.py --save_images
```
