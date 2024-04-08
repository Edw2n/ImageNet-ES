import lpips
import os, sys

import cv2
from tqdm import tqdm
import numpy as np
import argparse

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CURR_PATH)

sys.path.append(BASE_PATH)
from configs.config import DATASET_SUBPATH, LIGHT_OPTIONS, NUM_PARAMS_VAL, NUM_PARAMS_TEST, NUM_PARAMS_AUTO, LPIPS_THRESHOLD

parser = argparse.ArgumentParser(description='LPIPS Calc')
parser.add_argument('--setting', default='param_control', type=str,
                    help='Which setting is used? param_control / auto_exposure')

parser.add_argument('--data_root', default='/home/datasets', type=str,
                    help='Datset root directory')

args = parser.parse_args()



imagenet_dir = os.path.join(args.data_root, DATASET_SUBPATH['imagenet'], 'val')

if args.setting == 'auto_exposure':
    num_params = NUM_PARAMS_AUTO
    imagenet_es_dir =  os.path.join(args.data_root, DATASET_SUBPATH['imagenet-es-auto'])
    imagenet_es_dir = imagenet_es_dir.replace('es-test', 'es-val')
    log_file = 'lpips_calc_auto.csv'
    log_file_byparams = 'lpips_auto_byparams.csv'
elif args.setting == 'param_control':
    num_params = NUM_PARAMS_VAL
    imagenet_es_dir =  os.path.join(args.data_root, DATASET_SUBPATH['imagenet-es'])
    imagenet_es_dir = imagenet_es_dir.replace('es-test', 'es-val')
    log_file = 'lpips_calc_val.csv'
    log_file_byparams = 'lpips_val_byparams.csv'

param_set = list(range(1, 1+num_params))

loss_fn = lpips.LPIPS(net='alex').cuda()

calc_result = open(log_file, 'w+')
calc_result_byparams = open(log_file_byparams, 'w+')
lpips_dict = {}

for light in LIGHT_OPTIONS:
    print('Light condition:', light)
    for p in param_set:
        print('Parameter set:', p)
        curr_dir = os.path.join(imagenet_es_dir, light, 'param_' + str(p))
        classes = os.listdir(curr_dir)
        distances = []
        
        for c in tqdm(classes):
            curr_class_dir = os.path.join(curr_dir, c)
            for f in os.listdir(curr_class_dir):
                img_es = cv2.imread(os.path.join(curr_class_dir, f))[:,:,::-1]
                img_orig = cv2.imread(os.path.join(imagenet_dir, c, f))[:,:,::-1]

                h, w, _ = img_orig.shape
                img_es = cv2.resize(img_es, (w, h))

                img_es = lpips.im2tensor(img_es).cuda()
                img_orig = lpips.im2tensor(img_orig).cuda()

                # img_es = lpips.im2tensor(lpips.load_image(os.path.join(curr_class_dir, f))).cuda()
                # img_orig = lpips.im2tensor(lpips.load_image(os.path.join(imagenet_dir, c, f))).cuda()

                dist01 = loss_fn.forward(img_es, img_orig).cpu().detach().numpy()
                distances.append(dist01)
                # print('Distance: %.3f'%dist01)

                calc_result.write(f'{light},{str(p)},{c},{f},' +'%.3f'%dist01 + '\n')
        
        lpips_dict[f'{light}_p{p}'] = np.mean(distances)
        

for k, v in lpips_dict.items():
    calc_result_byparams.write(f'{k},{v}' + '\n')

calc_result.close()         
calc_result_byparams.close()         

            

                