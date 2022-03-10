## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

## Download training and testing data for single-image motion deblurring task
import os
# import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='train, test or train-test')
parser.add_argument('--dataset', type=str, default='GoPro', help='all, GoPro, HIDE, RealBlur_R, RealBlur_J')
args = parser.parse_args()

### Google drive IDs ######
GoPro_train = '1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI'      ## https://drive.google.com/file/d/1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI/view?usp=sharing
GoPro_test  = '1k6DTSHu4saUgrGTYkkZXTptILyG9RRll'      ## https://drive.google.com/file/d/1k6DTSHu4saUgrGTYkkZXTptILyG9RRll/view?usp=sharing
HIDE_test = '1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A'        ## https://drive.google.com/file/d/1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A/view?usp=sharing
RealBlurR_test = '1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS'   ## https://drive.google.com/file/d/1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS/view?usp=sharing
RealBlurJ_test = '1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW'   ## https://drive.google.com/file/d/1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW/view?usp=sharing

dataset = args.dataset

for data in args.data.split('-'):
    if data == 'train':
        print('GoPro Training Data!')
        os.makedirs(os.path.join('Datasets', 'Downloads'), exist_ok=True)
        # gdown.download(id=GoPro_train, output='Datasets/Downloads/train.zip', quiet=False)
        os.system(f'gdrive download {GoPro_train} --path Datasets/Downloads/')
        print('Extracting GoPro data...')
        shutil.unpack_archive('Datasets/Downloads/train.zip', 'Datasets/Downloads')
        os.rename(os.path.join('Datasets', 'Downloads', 'train'), os.path.join('Datasets', 'Downloads', 'GoPro'))
        os.remove('Datasets/Downloads/train.zip')

    if data == 'test':
        if dataset == 'all' or dataset == 'GoPro':
            print('GoPro Testing Data!')
            # gdown.download(id=GoPro_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {GoPro_test} --path Datasets/')
            print('Extracting GoPro Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')

        if dataset == 'all' or dataset == 'HIDE':
            print('HIDE Testing Data!')
            # gdown.download(id=HIDE_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {HIDE_test} --path Datasets/')
            print('Extracting HIDE Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')

        if dataset == 'all' or dataset == 'RealBlur_R':
            print('RealBlur_R Testing Data!')
            # gdown.download(id=RealBlurR_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {RealBlurR_test} --path Datasets/')
            print('Extracting RealBlur_R Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')

        if dataset == 'all' or dataset == 'RealBlur_J':
            print('RealBlur_J testing Data!')
            # gdown.download(id=RealBlurJ_test, output='Datasets/test.zip', quiet=False)
            os.system(f'gdrive download {RealBlurJ_test} --path Datasets/')
            print('Extracting RealBlur_J Data...')
            shutil.unpack_archive('Datasets/test.zip', 'Datasets')
            os.remove('Datasets/test.zip')


# print('Download completed successfully!')
