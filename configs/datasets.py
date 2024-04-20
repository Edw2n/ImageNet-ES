from .user_configs import IMAGENET_ES_ROOT_DIR

import os
import torch
import torchvision.models as models

DATA_ROOT_DIR = IMAGENET_ES_ROOT_DIR

SAMPLE_DIR_MAP = {
    'es-train': 'tin_no_resize_sample_removed', #S3
    'es-val': 'sampled_tin_no_resize', #S1
    'es-test': 'sampled_tin_no_resize2', #S2
}

OPS_NUM = {
    'es-val': 64,
    'es-test': 27,
}

ENVS = {
    'es-val': ['l1', 'l5'],
    'es-test': ['l1', 'l5'],
}

DATASET_SUBPATH = {
    'imagenet': os.path.join('ILSVRC12'),
    'imagenet-c': os.path.join('ImageNet-C'),
    'imagenet-c-tin': os.path.join('ImageNet-C'),
    'imagenet-r': os.path.join('ImageNet-R'),
    'imagenet-a': os.path.join('ImageNet-A'),
    'imagenet-tin': os.path.join('ILSVRC12'),
    'imagenet-es': os.path.join('ImageNet-ES', 'es-test', 'param_control'),
    'imagenet-es-auto': os.path.join('ImageNet-ES', 'es-test', 'auto_exposure')
}

TIMM_MODELS = {
    'dinov2': 'vit_giant_patch14_dinov2.lvd142m',
    'dinov2_b': 'vit_large_patch14_dinov2.lvd142m',
    'eff_b0': 'efficientnet_b0.ra_in1k',
    'eff_b3': 'efficientnet_b0.ra_in1k',
    'res50': 'resnet50.a1_in1k',
    'res152': 'resnet152.a1_in1k',
    'swin_t': 'swin_tiny_patch4_window7_224.ms_in1k',
    'swin_s': 'swin_small_patch4_window7_224.ms_in1k',
    'swin_b': 'swin_base_patch4_window7_224.ms_in1k',
    'swin_l': 'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
    'clip_b': 'vit_base_patch16_clip_224.laion2b_ft_in1k',
    'clip_l': 'vit_large_patch14_clip_224.laion2b_ft_in1k',
    'clip_h': 'vit_huge_patch14_clip_224.laion2b_ft_in1k'
}

TORCH_MODELS = {
    'dinov2': torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_lc"),
    'dinov2_b': torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc"),
    'eff_b0': models.efficientnet_b0(pretrained=True),
    'eff_b3':  models.efficientnet_b3(pretrained=True),
    'res50': models.resnet50(pretrained=True),
    'res50_aug': models.resnet50(pretrained=False),
    'res152': models.resnet152(pretrained=True),
    'swin_t': models.swin_v2_t(pretrained=True),
    'swin_s': models.swin_v2_s(pretrained=True),
    'swin_b': models.swin_v2_b(pretrained=True)
}

# LIGHT_OPTIONS = ['l1','l5']
# NUM_PARAMS_VAL = 64
# NUM_PARAMS_TEST = 27
NUM_PARAMS_AUTO = 5
LPIPS_THRESHOLD = 0.8 # LPIPS Threshold to filter out too distorted manual settings in ImangeNet-ES