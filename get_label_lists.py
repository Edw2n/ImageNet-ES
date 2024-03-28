from configs.datasets import DATA_ROOT_DIR, OPS_NUM, SAMPLE_DIR_MAP, ENVS
from configs.tin_config import LOADER_CONFIG as tin_config
from configs.models.model_config import timm_config
from utils.wrapper_nn import W_NN
from configs.models.model_kwargs import KWARGS_MAP
from utils.labeler import Labeler
from utils.experiment_setup import cuda_setup, load_model, get_labeler_args

import argparse
import os
from PIL import ImageFile
import torch as ch

parser = argparse.ArgumentParser()
args = get_labeler_args(parser)

if __name__ == '__main__':
    device_num = cuda_setup(args.gpu_num)
    model_name = args.model
    model, model_kwargs = load_model(model_name, device_num)
    
    if not model:
        exit()
        
    device, = list(set(p.device for p in model.parameters()))    
    model_config = timm_config[model_name]
    cfg = model_config['CFG']

    labeler = Labeler(model, {
            'n_workers': tin_config['num_workers']['val'],
            'bs': cfg['batch_size'],
            'transformations': tin_config['DATA_TRANSFORMS']['val'],
            'device': device
    }, model_kwargs["arch"], envs=ENVS['es-val'])
        
    for target, sample_dir  in SAMPLE_DIR_MAP.items():
        ROOT_DIR = f'{DATA_ROOT_DIR}/{target}'
        
        if 'train' not in target:
            # for ImageNet-ES(parameter controled)
            labeler.generate_option_labels(OPS_NUM[target], ROOT_DIR)
            accs = labeler.get_param_accs(ROOT_DIR, OPS_NUM[target])
            
        # for sampled data
        labeler.generate_sample_labels(ROOT_DIR, sample_dir) # id/ood labeling
        labeler.generate_standard_labels(ROOT_DIR, sample_dir) # original labeling (deal all samples as id)

     