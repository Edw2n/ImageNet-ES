import os, sys
import timm
import torch
import torchvision.models as models
import torch.nn as nn

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CURR_PATH)

sys.path.append(BASE_PATH)

from configs.datasets import TIMM_MODELS, TORCH_MODELS


def get_pretrained_models(is_timm, arch):
    if is_timm:
        if arch not in TIMM_MODELS:
            print("WARNING: No timm version is provided, use torch version...")
            exit(0)
        
        model = timm.create_model(TIMM_MODELS[arch], pretrained=True)

        #Fine-tuned resnet 50 model weights          
        if arch == 'res50':
            from functools import reduce 
            getter_path = ['fc']
            fv_module_getter = reduce(getattr, [model, *getter_path[:-1]])
            fv_in = getattr(fv_module_getter, getter_path[-1]).in_features
            model.fc = nn.Linear(fv_in, 200, bias=True)
            state_dict = torch.load(os.path.join(BASE_PATH, 'pretrained', 'lr-[0.005]_epoch-19.pth'))
            model.load_state_dict(state_dict)
        print("Model loaded from timm:", arch)
    else:
        if arch not in TORCH_MODELS:
            print("WARNING: pytorch model is not provided. Please use timm version")
            exit(0)
        
        model = TORCH_MODELS[arch]

        if arch == 'res50_aug':
            state_dict = torch.load(os.path.join(BASE_PATH, 'pretrained', 'deepaugment_and_augmix.pth.tar'))['state_dict']
            new_state_dict = {}

            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)

        print("Model loaded from PyTorch:", arch)
    return model