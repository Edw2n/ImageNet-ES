import os
import pickle
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import re
import pandas as pd

from modules.custom_datasets import DatasetFolder_withpath
from torchvision.datasets.folder import default_loader
from PIL import ImageOps, Image

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CURR_PATH)

from configs.config import DATASET_SUBPATH, LIGHT_OPTIONS, NUM_PARAMS_VAL, NUM_PARAMS_TEST, NUM_PARAMS_AUTO, LPIPS_THRESHOLD

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

val_transform_c = transforms.Compose([                
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

train_transform_comp = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

train_transform_basic = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.RandomSolarize(192, p=0.3),
            transforms.RandomPosterize(2, p=0.1),
            transforms.ToTensor(),
            normalize,
        ])

train_transform_AdvPre = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
preprocess = transforms.Compose([transforms.ToTensor(), normalize])


def dataset_path(dataset_name, data_root):
    return os.path.join(data_root, DATASET_SUBPATH[dataset_name])


def _get_dataset(data_dir, cls_filter, fname_filter, _transform):
    return DatasetFolder_withpath(
        data_dir,
        loader=default_loader,
        extensions=IMG_EXTENSIONS,
        transform=_transform,
        cls_filter=cls_filter,
        fname_filter=fname_filter)
     

## Get Test datasets
def get_dataset(dataset_name, data_root, arch):

    dataset_dict = {}

    cls_filter, fname_filter = None, None
    
    if dataset_name in ['imagenet-tin', 'imagenet-c-tin']:
        with open(os.path.join(BASE_PATH, 'metadata', 'tin_wnids.txt'), 'r') as f:
            cls_filter = sorted([c.strip()for c in f.readlines()])

        with open(os.path.join(BASE_PATH, 'metadata', 'fnames_tin_test.pkl'), 'rb') as f:
            fname_filter = pickle.load(f)


    if dataset_name == 'imagenet':
        # data_dir = os.path.join(data_root, 'imagenet', 'val')
        data_dir = os.path.join(dataset_path(dataset_name, data_root),'val')
        dataset = _get_dataset(data_dir, cls_filter, fname_filter, val_transform)
        dataset_dict[f'{dataset_name}_{arch}'] = dataset

    elif dataset_name == 'imagenet-tin':
        data_dir = os.path.join(dataset_path(dataset_name, data_root),'val')
        dataset = _get_dataset(data_dir, cls_filter, fname_filter, val_transform)
        dataset_dict[f'{dataset_name}_{arch}'] = dataset

    elif dataset_name in ['imagenet-a', 'imagenet-r']:

        data_dir = dataset_path(dataset_name, data_root)
        dataset = _get_dataset(data_dir, cls_filter, fname_filter, val_transform)
        dataset_dict[f'{dataset_name}_{arch}'] = dataset
        
    elif dataset_name in ['imagenet-c','imagenet-c-tin']:
        distortions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]

        severity = list(range(1,6))

        errs = {}

        for d_name in distortions:
            errs[d_name] = []            
            for s in severity:
                
                data_dir = os.path.join(dataset_path(dataset_name, data_root), d_name, str(s))
                dataset = _get_dataset(data_dir, cls_filter, fname_filter, val_transform_c)
                dataset_dict[f'{dataset_name}__{arch}__{d_name}__{s}'] = dataset

    elif dataset_name in ['imagenet-es','imagenet-es-auto']:
        param_sets = []
        light = LIGHT_OPTIONS
        num_params = NUM_PARAMS_TEST if dataset_name == 'imagenet-es' else NUM_PARAMS_AUTO

        for i in range(1,1+num_params):            
            param_sets.append(f'param_{i}')

        for l in light:        
            for param in param_sets:
                data_dir = os.path.join(dataset_path(dataset_name, data_root), l, param)
                dataset = _get_dataset(data_dir, cls_filter, fname_filter, val_transform)
                dataset_dict[f'{dataset_name}__{arch}__{l}__{param}'] = dataset
    
    return dataset_dict


## AugMix functions
def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)

def autocontrast(pil_img, level=None):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, level=None):
    return ImageOps.equalize(pil_img)

def rotate(pil_img, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)

def solarize(pil_img, level):
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((224, 224), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def shear_y(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((224, 224), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_x(pil_img, level):
    level = int_parameter(rand_lvl(level), 224 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((224, 224), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_y(pil_img, level):
    level = int_parameter(rand_lvl(level), 224 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((224, 224), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=128)

def posterize(pil_img, level):
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

aug_severity = 1
augmentations = [
    autocontrast,
    equalize,
    lambda x: rotate(x, aug_severity),
    lambda x: solarize(x, aug_severity),
    lambda x: shear_x(x, aug_severity),
    lambda x: shear_y(x, aug_severity),
    lambda x: translate_x(x, aug_severity),
    lambda x: translate_y(x, aug_severity),
    lambda x: posterize(x, aug_severity),
]


def get_mixture(x_orig, x_processed):
    aug_width = 3
    alpha = 1
    if aug_width > 1:
        w = np.float32(np.random.dirichlet([alpha] * aug_width))
    else:
        w = [1.]
    m = np.float32(np.random.beta(alpha, alpha))

    mix = torch.zeros_like(x_processed)
    for i in range(aug_width):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(augmentations)(x_aug)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMix(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        # print(self.dataset[i])
        x_orig, y, paths = self.dataset[i]

        x_processed = preprocess(x_orig)
        mix1 = get_mixture(x_orig, x_processed)
        mix2 = get_mixture(x_orig, x_processed)

        # done so that on 2 GPUs, there is an equal split of clean images for each GPU's batch norm
        return [mix1, x_processed, mix2], y, paths

    def __len__(self):
        return len(self.dataset)



def parse_lpips(input_str):
    # input: l{1,5}_p{1~64}
    # output: ({1,5}, {1~64})
    num1 = re.findall(r'\d+', input_str.split('_')[0])[0]
    num2 = re.findall(r'\d+', input_str.split('_')[1])[0]

    return (int(num1), int(num2))



## Get train dataset for Domain Generalization (Augmentation) analysis
def get_train_dataset(data_root, exp_settings=0, use_es_training=False):

    traindir = os.path.join(dataset_path('imagenet', data_root), 'val')
    valdir_im = os.path.join(dataset_path('imagenet', data_root), 'val')
    valdir_im_c = dataset_path('imagenet-c', data_root)
    valdir_im_as = os.path.join(dataset_path('imagenet-es', data_root), 'test', 'param_control')

    cls_filter, fname_filter_test,  fname_filter_val = None, None, None
    # if dataset_name in ['imagenet-tin', 'imagenet-c-tin']:
    with open(os.path.join(BASE_PATH, 'metadata', 'tin_wnids.txt'), 'r') as f:
        cls_filter = sorted([c.strip()for c in f.readlines()])

    with open(os.path.join(BASE_PATH, 'metadata', 'fnames_tin_test.pkl'), 'rb') as f:
        fname_filter_test = pickle.load(f)

    with open(os.path.join(BASE_PATH, 'metadata', 'fnames_tin_val.pkl'), 'rb') as f:
        fname_filter_val = pickle.load(f)

    train_transform_map = {0:train_transform_comp, 
                        1:train_transform_basic, 
                        2:train_transform_AdvPre}
    
    train_transform = train_transform_map[exp_settings]

    replace_dict = None
    if use_es_training:
        lpips_val = pd.read_csv(os.path.join(BASE_PATH, 'lpips', 'lpips_val_byparams.csv'))
        filtered_params = lpips_val[lpips_val['lpips'] < LPIPS_THRESHOLD]['param']
        valid_params = [parse_lpips(p) for p in filtered_params]
        replace_dict = {'ILSVRC12/val':[f'ImageNet-ES/es-val/param_control/l{i}/param_{j}' for i, j in valid_params]}
    

    train_dataset = DatasetFolder_withpath(
        traindir,
        loader=default_loader,
        cls_filter=cls_filter,
        fname_filter=fname_filter_val,
        transform=train_transform,
        replace_dict=replace_dict
        )

    if exp_settings == 2:
        # Add distorted images
        edsr_dataset = DatasetFolder_withpath(
            os.path.join(BASE_PATH,'EDSR'),
            loader=default_loader,
            cls_filter=cls_filter,
            fname_filter=fname_filter_val,
            transform=train_transform)

        cae_dataset = DatasetFolder_withpath(
            os.path.join(BASE_PATH,'CAE'),
            loader=default_loader,
            cls_filter=cls_filter,
            fname_filter=fname_filter_val,
            transform=train_transform)

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, edsr_dataset, cae_dataset])

        train_dataset = AugMix(train_dataset)

    return train_dataset