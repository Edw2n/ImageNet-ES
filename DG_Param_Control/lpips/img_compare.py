import lpips
import os

import cv2
from tqdm import tqdm
import numpy as np

setting = 'param_control'

imagenet_dir = '/datasets/ILSVRC12/val'

if setting == 'auto_exposure':
    num_params = 5
    imagenet_as_dir = '/datasets/ImageNet-ES/auto_exposure'
    log_file = 'lpips_calc_auto.csv'
    log_file_byparams = 'lpips_auto_byparams.csv'
elif setting == 'param_control':
    num_params = 64
    imagenet_as_dir = '/datasets/ImageNet-ES/val/param_control'
    log_file = 'lpips_calc_val.csv'
    log_file_byparams = 'lpips_val_byparams.csv'
light_cond = ['l1','l5']
param_set = list(range(1, 1+num_params))

loss_fn = lpips.LPIPS(net='alex').cuda()

calc_result = open(log_file, 'w+')
calc_result_byparams = open(log_file_byparams, 'w+')
lpips_dict = {}

for light in light_cond:
    print('Light condition:', light)
    for p in param_set:
        print('Parameter set:', p)
        curr_dir = os.path.join(imagenet_as_dir, light, 'param_' + str(p))
        classes = os.listdir(curr_dir)
        distances = []
        
        for c in tqdm(classes):
            curr_class_dir = os.path.join(curr_dir, c)
            for f in os.listdir(curr_class_dir):
                img_as = cv2.imread(os.path.join(curr_class_dir, f))[:,:,::-1]
                img_orig = cv2.imread(os.path.join(imagenet_dir, c, f))[:,:,::-1]

                h, w, _ = img_orig.shape
                img_as = cv2.resize(img_as, (w, h))

                img_as = lpips.im2tensor(img_as).cuda()
                img_orig = lpips.im2tensor(img_orig).cuda()

                # img_as = lpips.im2tensor(lpips.load_image(os.path.join(curr_class_dir, f))).cuda()
                # img_orig = lpips.im2tensor(lpips.load_image(os.path.join(imagenet_dir, c, f))).cuda()

                dist01 = loss_fn.forward(img_as, img_orig).cpu().detach().numpy()
                distances.append(dist01)
                # print('Distance: %.3f'%dist01)

                calc_result.write(f'{light},{str(p)},{c},{f},' +'%.3f'%dist01 + '\n')
        
        lpips_dict[f'{light}_p{p}'] = np.mean(distances)
        

for k, v in lpips_dict.items():
    calc_result_byparams.write(f'{k},{v}' + '\n')

calc_result.close()         
calc_result_byparams.close()         

            

                