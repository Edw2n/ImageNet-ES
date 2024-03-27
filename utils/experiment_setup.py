import argparse
import os
import torch as ch

from configs.models.model_kwargs import KWARGS_MAP
from utils.wrapper_nn import W_NN

def get_labeler_args(parser: argparse.ArgumentParser):
    parser.add_argument('-model', default='en') 
    parser.add_argument('-gpu_num', help='1 or 0')
    parser.add_argument('-bs', default=100)
    args = parser.parse_args()
    print('Experiments arguments:')
    print(args)
    return args

def get_evalood_args(parser: argparse.ArgumentParser):
    parser.add_argument('-model', default='en') 
    parser.add_argument('-gpu_num', help='1 or 0')
    parser.add_argument('-bs', default=100)
    parser.add_argument('-id_name')
    parser.add_argument('-test_dir', default='es-test')
    parser.add_argument('-val_dir', default='es-val')
    parser.add_argument('-init', default='n')
    parser.add_argument('-test_op_num', default=27) 
    parser.add_argument('-val_op_num', default=64)
    parser.add_argument('-output_dir', default='./results') 
    args = parser.parse_args()
    
    print('Experiments arguments:')
    print(args)
    return args

def cuda_setup(gpu_num):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    device_num = 0
    return device_num
    
def oodexp_setup(args):
    device_num = cuda_setup(args.gpu_num)
    EXPERIMENT_ID = f'TIN2-{args.id_name.upper()}'
    OUTPUT_ROOT_DIR = args.output_dir
    return device_num, EXPERIMENT_ID, OUTPUT_ROOT_DIR

def load_model(model_name, device_num):
    # if no problem, load model and return model else return None (exit() please)
    model = None
    try:
        model_kwargs = KWARGS_MAP[model_name]
        device = f"cuda:{device_num}" if ch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        model = W_NN(**model_kwargs)
        model.to(device)
        print(model.eval())
    except Exception as e:
        print(e)
        print('not supporting model keyword:', model_name)
        print('supporting list:')
        for key, val in KWARGS_MAP.items():
            print(f'keyword: {key}, architecture:{val}')
    return model, model_kwargs
    
    