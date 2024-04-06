
# CUDA_VISIBLE_DEVICES=0 python augment_analysis.py -a resnet50 --seed 1001 --epochs 10 -b 256 --exp-settings 0  
# CUDA_VISIBLE_DEVICES=0 python augment_analysis.py -a resnet50 --seed 1001 --epochs 10 -b 256 --exp-settings 1  
# CUDA_VISIBLE_DEVICES=0 python augment_analysis.py -a resnet50 --seed 1001 --epochs 10 -b 256 --exp-settings 2  


CUDA_VISIBLE_DEVICES=3 python augment_analysis.py -a resnet50 --seed 1001 --epochs 10 -b 256 --exp-settings 0 --use-es-training
CUDA_VISIBLE_DEVICES=3 python augment_analysis.py -a resnet50 --seed 1001 --epochs 10 -b 256 --exp-settings 1 --use-es-training
CUDA_VISIBLE_DEVICES=3 python augment_analysis.py -a resnet50 --seed 1001 --epochs 10 -b 128 --exp-settings 2 --use-es-training