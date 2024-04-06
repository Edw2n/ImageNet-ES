
################################### PyTorch evaluation script ############################################

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a eff_b0 -b 1024 --pretrained     --dataset imagenet-tin --log_file logs_eff_b0_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a eff_b0 -b 1024 --pretrained     --dataset imagenet-es     --log_file logs_eff_b0_imagenet-es.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a eff_b0 -b 1024 --pretrained     --dataset imagenet-es-auto     --log_file logs_eff_b0_imagenet-es-auto.txt


CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 -b 1024 --pretrained     --dataset imagenet-tin --log_file logs_res50_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 -b 1024 --pretrained     --dataset imagenet-es     --log_file logs_res50_imagenet-es.txt --save_features
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 -b 1024 --pretrained     --dataset imagenet-es-auto     --log_file logs_res50_imagenet-es-auto.txt --save_features

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res152 --pretrained --dataset imagenet-tin --log_file logs_res152_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res152 --pretrained --dataset imagenet-es     --log_file logs_res152_imagenet-es.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res152 --pretrained --dataset imagenet-es-auto     --log_file logs_res152_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --dataset imagenet-tin --log_file logs_res50_aug_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --dataset imagenet-es     --log_file logs_res50_aug_imagenet-es.txt
CUDA_VISIBLE_DEVICES=0 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --dataset imagenet-es-auto     --log_file logs_res50_aug_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_t -b 1024 --pretrained --dataset imagenet-tin --log_file logs_swin_t_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_t -b 1024 --pretrained --dataset imagenet-es     --log_file logs_swin_t_imagenet-es.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_t -b 1024 --pretrained --dataset imagenet-es-auto     --log_file logs_swin_t_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_s -b 1024 --pretrained --dataset imagenet-tin --log_file logs_swin_s_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_s -b 1024 --pretrained --dataset imagenet-es     --log_file logs_swin_s_imagenet-es.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_s -b 1024 --pretrained --dataset imagenet-es-auto     --log_file logs_swin_s_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained --dataset imagenet-tin --log_file logs_swin_b_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained --dataset imagenet-es     --log_file logs_swin_b_imagenet-es.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained --dataset imagenet-es-auto     --log_file logs_swin_b_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained --dataset imagenet-tin --log_file logs_dinov2_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained --dataset imagenet-es --log_file logs_dinov2_imagenet-es.txt
CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a dinov2 --pretrained --dataset imagenet-es-auto --log_file logs_dinov2_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2_b --pretrained --dataset imagenet-tin --log_file logs_dinov2_b_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2_b --pretrained --dataset imagenet-es --log_file logs_dinov2_b_imagenet-es.txt
CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a dinov2_b --pretrained --dataset imagenet-es-auto --log_file logs_dinov2_b_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a eff_b3 -b 1024 --pretrained     --dataset imagenet-tin --log_file logs_eff_b3_imagenet-tin.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a eff_b3 -b 1024 --pretrained     --dataset imagenet-es     --log_file logs_eff_b3_imagenet-es.txt
CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a eff_b3 -b 1024 --pretrained     --dataset imagenet-es-auto     --log_file logs_eff_b3_imagenet-es-auto.txt

CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_b --pretrained --timm    --dataset imagenet-tin --log_file logs_clip_l_imagenet-tin_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_b --pretrained --timm    --dataset imagenet-es --log_file logs_clip_b_imagenet-es_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_b --pretrained --timm    --dataset imagenet-es-auto --log_file logs_clip_b_imagenet-es-auto_timm.txt

CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_h --pretrained --timm    --dataset imagenet-tin --log_file logs_clip_h_imagenet-tin_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_h --pretrained --timm    --dataset imagenet-es --log_file logs_clip_h_imagenet-es_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_h --pretrained --timm    --dataset imagenet-es-auto --log_file logs_clip_h_imagenet-es-auto_timm.txt

################################### TIMM evaluation script ############################################

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 -b 1024 --pretrained --timm    --dataset imagenet-tin --log_file logs_res50_imagenet-tin_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 -b 1024 --pretrained --timm    --dataset imagenet-es     --log_file logs_res50_imagenet-es_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 -b 1024 --pretrained --timm    --dataset imagenet-es-auto     --log_file logs_res50_imagenet-es-auto_timm.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res152 --pretrained --timm    --dataset imagenet-tin --log_file logs_res152_imagenet-tin_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res152 --pretrained --timm    --dataset imagenet-es     --log_file logs_res152_imagenet-es_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res152 --pretrained --timm    --dataset imagenet-es-auto     --log_file logs_res152_imagenet-es-auto_timm.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --timm            --dataset imagenet-tin --log_file logs_res50_aug_imagenet-tin.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --timm     --dataset imagenet-es     --log_file logs_res50_aug_imagenet-es.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --timm     --dataset imagenet-es-auto     --log_file logs_res50_aug_imagenet-es-auto.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_t -b 1024 --pretrained --timm    --dataset imagenet-tin --log_file logs_swin_t_imagenet-tin_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_t -b 1024 --pretrained --timm    --dataset imagenet-es     --log_file logs_swin_t_imagenet-es_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_t -b 1024 --pretrained --timm    --dataset imagenet-es-auto     --log_file logs_swin_t_imagenet-es-auto_timm.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_s -b 1024 --pretrained --timm    --dataset imagenet-tin --log_file logs_swin_s_imagenet-tin_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_s -b 1024 --pretrained --timm    --dataset imagenet-es     --log_file logs_swin_s_imagenet-es_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_s -b 1024 --pretrained --timm    --dataset imagenet-es-auto     --log_file logs_swin_s_imagenet-es-auto_timm.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained --timm    --dataset imagenet-tin --log_file logs_swin_b_imagenet-tin_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained --timm    --dataset imagenet-es     --log_file logs_swin_b_imagenet-es_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained --timm    --dataset imagenet-es-auto     --log_file logs_swin_b_imagenet-es-auto_timm.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_l --pretrained --timm    --dataset imagenet-tin --log_file logs_swin_l_imagenet-tin_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_l --pretrained --timm    --dataset imagenet-es     --log_file logs_swin_l_imagenet-es_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_l --pretrained --timm    --dataset imagenet-es-auto     --log_file logs_swin_l_imagenet-es-auto_timm.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained --timm    --dataset imagenet-tin --log_file logs_dinov2_imagenet-tin_timm.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained --timm    --dataset imagenet-es --log_file logs_dinov2_imagenet-es_timm.txt
# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a dinov2 --pretrained --timm    --dataset imagenet-es-auto --log_file logs_dinov2_imagenet-es-auto_timm.txt

CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_b --pretrained --timm    --dataset imagenet-tin --log_file logs_clip_l_imagenet-tin_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_b --pretrained --timm    --dataset imagenet-es --log_file logs_clip_b_imagenet-es_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_b --pretrained --timm    --dataset imagenet-es-auto --log_file logs_clip_b_imagenet-es-auto_timm.txt

# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a clip_l --pretrained --timm    --dataset imagenet-tin --log_file logs_clip_l_imagenet-tin_timm.txt 
# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a clip_l --pretrained --timm    --dataset imagenet-es --log_file logs_clip_l_imagenet-es_timm.txt --data_root ~/datasets
# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a clip_l --pretrained --timm    --dataset imagenet-es-auto --log_file logs_clip_l_imagenet-es-auto_timm.txt --data_root ~/datasets


CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_h --pretrained --timm    --dataset imagenet-tin --log_file logs_clip_h_imagenet-tin_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_h --pretrained --timm    --dataset imagenet-es --log_file logs_clip_h_imagenet-es_timm.txt 
CUDA_VISIBLE_DEVICES=3 python imagenet_es_eval.py -a clip_h --pretrained --timm    --dataset imagenet-es-auto --log_file logs_clip_h_imagenet-es-auto_timm.txt




###########################################################################################################

# Dinov2 Imagenet linear classification
# CUDA_VISIBLE_DEVICES=0 python imagenet_eval.py --seed 2481757 -a dinov2 --pretrained --evaluate data/ILSVRC12/


# Dinov2 Imagenet-C evaluation
# CUDA_VISIBLE_DEVICES=1 python robustness/ImageNet-C/test.py --model-name dinov2 --test_bs 1024 --num_workers 8


# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 --pretrained            --dataset imagenet --log_file logs_res50_imagenet.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 --pretrained            --dataset imagenet-a --log_file logs_res50_imagenet-a.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50 --pretrained            --dataset imagenet-r --log_file logs_res50_imagenet-r.txt
# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a res50 -b 1024 --pretrained     --dataset imagenet-c     --log_file logs_res50_imagenet-c.txt
# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a res50 -b 1024 --pretrained     --dataset imagenet-c-tin --log_file logs_res50_imagenet-c-tin.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug --pretrained            --dataset imagenet --log_file logs_res50_aug_imagenet.txt
# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --dataset imagenet-c     --log_file logs_res50_aug_imagenet-c.txt
# CUDA_VISIBLE_DEVICES=2 python imagenet_es_eval.py -a res50_aug -b 1024 --pretrained --dataset imagenet-c-tin --log_file logs_res50_aug_imagenet-c-tin.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug --pretrained            --dataset imagenet-a --log_file logs_res50_aug_imagenet-a.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_aug --pretrained            --dataset imagenet-r --log_file logs_res50_aug_imagenet-r.txt


# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained     --dataset imagenet-a --log_file logs_swin_b_imagenet-a.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained     --dataset imagenet-r --log_file logs_swin_b_imagenet-r.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained     --dataset imagenet --log_file logs_swin_b_imagenet.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained     --dataset imagenet-c-tin --log_file logs_swin_b_imagenet-c-tin.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a swin_b -b 1024 --pretrained     --dataset imagenet-c     --log_file logs_swin_b_imagenet-c.txt


# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained            --dataset imagenet --log_file logs_dinov2_imagenet.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained            --dataset imagenet-a --log_file logs_dinov2_imagenet-a.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained            --dataset imagenet-r --log_file logs_dinov2_imagenet-r.txt
# CUDA_VISIBLE_DEVICES=0 python imagenet_es_eval.py -a dinov2 --pretrained            --dataset imagenet-c     --log_file logs_dinov2_imagenet-c.txt
# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a dinov2 --pretrained            --dataset imagenet-c-tin --log_file logs_dinov2_imagenet-c-tin.txt

# CUDA_VISIBLE_DEVICES=1 python imagenet_es_eval.py -a res50_madry -b 1024 --pretrained     --dataset imagenet --log_file logs_res50_madry_imagenet.txt
