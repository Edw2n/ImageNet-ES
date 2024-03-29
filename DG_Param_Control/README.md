# ImageNet-ES
## Domain generalization
- This code produces the experimental results presented in Table 2 and 3 in Paper 7880: **Unexplored Faces of Robustness and Out-of-Distribution: Covariate Shifts in Environment and Sensor Domains**

### Environment setup
We use PyTorch and other packages. Please use the following command to install the necessary packages:
```
conda env create -f environment.yaml
```

### Datasets
We use ImageNet, ImageNet-C and ImageNet-ES for evaluation. Please prepare the datasets as following in the same directory. `data_root` argument is used to specify the parent directory which includes all datasets.
- ImageNet: `ILSVRC12`
- ImageNet-C: `ImageNet-C`
- ImageNet-ES: `ImageNet-ES`

### Domain generalization techniques (Table 2)
Please use following command to run the experiments proposed in Table 2.
```
python imagnet_es_eval.py --data_root [DATASET DIRECTORY] -a resnet50 --seed [SEED] --epochs [NUM_EPOCHS] -b [BATCH_SIZE] --exp-settings [EXPERIMENT SETTING] --use-es-training (Optional)
```
- Description of `exp-settings` argument:
    - 0 for compositional augmentation only (RandomCrop, RandomResize, RandomFlip)
    - 1 for basic augmentation (ColorJitter, RandomSolarize, RandomPosterize)
    - 2 for advanced augmentation (DeepAugment and AugMix)
    - If `use-es-training` is **not** used, 0,1 and 2 correspond to Experiment 1,2 and 3 in the paper, respectively
    - If `use-es-training` is used, 0,1 and 2 correspond to Experiment 4,5 and 6 in the paper, respectively

- Description on `use-es-training` argument:
    - Use this argument to conduct experiment 4,5 and 6 in the paper
    - For example, `--exp-settings 0/1/2 --use-es-training` corresponds to experiment 4/5/6

- The logs are stored in `aug_logs` directory under following name: `aug_experiments_{exp_settings}_{use-es-training}.txt`

- Please refer to `aug_analysis.sh` file for the commands used for experiments.


- Note that to use DeepAugment, you need to prepare the distorted datasets as described in https://github.com/hendrycks/imagenet-r. The created dataset should be stored in `CAE` and `EDSR` directories.


### Evaluation of various models on ImageNet-ES (Table 3)
Please use following command to run the experiment proposed in Table 2.
```
python imagenet_as_eval.py -a [MODEL ARCHITECTURE] -b [BATCH_SIZE] --pretrained --dataset [EVALUATION DATASET] --log_file [LOG FILE NAME]
```
- Available model architecture (`-a` argument):
    - `eff_b0`: EfficientNet-B0
    - `eff_b3`: EfficientNet-B3
    - `res50`: ResNet-50
    - `res50_aug`: ResNet-50 trained with DeepAugment and AugMix (Nedd to download from https://github.com/hendrycks/imagenet-r)
    - `res152`: ResNet-152
    - `swin_t`: SwinV2 Tiny
    - `swin_s`: SwinV2 Small
    - `swin_b`: SwinV2 Base
    - `dinov2_b`: DINOv2 Base
    - `dinov2`: DINOv2 Giant    

- Available dataset (`--dataset` argument):
    - `imagenet-tin`: Subset of tiny that matches ImageNet-ES
    - `imagenet-es`: ImageNet-ES, Manual parameter settings
    - `imagenet-es-auto`: ImageNet-ES, Auto exposure settings

- The logs are stored in `logs` directory under the following name: `logs_{model architecture}_{dataset}.txt`

- Please refer to `eval_scripts.sh` file for the commands used for experiments.







