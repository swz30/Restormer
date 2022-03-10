## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

## Download training and testing data for image deraining task
import os
# import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
args = parser.parse_args()

### Google drive IDs ######
rain13k_train = '14BidJeG4nSNuFNFDf99K-7eErCq4i47t'   ## https://drive.google.com/file/d/14BidJeG4nSNuFNFDf99K-7eErCq4i47t/view?usp=sharing
rain13k_test  = '1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8'   ## https://drive.google.com/file/d/1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8/view?usp=sharing

for data in args.data.split('-'):
    if data == 'train':
        print('Rain13K Training Data!')
        # gdown.download(id=rain13k_train, output='Datasets/train.zip', quiet=False)
        os.system(f'gdrive download {rain13k_train} --path Datasets/')
        print('Extracting Rain13K data...')
        shutil.unpack_archive('Datasets/train.zip', 'Datasets')
        os.remove('Datasets/train.zip')

    if data == 'test':
        print('Download Deraining Testing Data')
        # gdown.download(id=rain13k_test, output='Datasets/test.zip', quiet=False)
        os.system(f'gdrive download {rain13k_test} --path Datasets/')
        print('Extracting test data...')
        shutil.unpack_archive('Datasets/test.zip', 'Datasets')
        os.remove('Datasets/test.zip')

   
# print('Download completed successfully!')


