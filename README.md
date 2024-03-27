# Unexplored Faces of Robustness and Out-of-Distribution:  Covariate Shifts in Environment and Sensor Domains (CVPR 2024)

## ImageNet-ES dataset and target model files download link
- ImageNet-ES dataset : https://drive.google.com/file/d/1GqWfza9wH3l1Ga6YKCO3NNd6RnlCqgA0/view?usp=share_link

- swin : https://drive.google.com/file/d/1olqC0PYmWhl4VCmI1dPd7wCxqtYTJT2O/view?usp=share_link

- EfficientNet : https://drive.google.com/file/d/1zzt2n1x-6-0ACqIKnAndiP7HPkr1dEuo/view?usp=share_link

- reesnet18 : https://drive.google.com/file/d/1L4S4hFRXF7S35wh3RfJOvsz-kZ__qxPJ/view?usp=share_link

## Get Started
### 0. Configurations
- In configs/user_configs.py, fill your path informations of directory and files of imagenet-ES datasets and models
```
IMAGENET_ES_ROOT_DIR = 'path/to/root-dir/of/imagenet-es'
SWIN_PT = "path/to/swin_model_weights/file"
RESNET18_PT = 'path/to/resnet18_model_weight/file'
EN_PT = "path/to/efficientnet_model_weight/file"
```

### 1. Environment Settings
- python 3.10.12, torch 2.0.1  

```
conda create -n ies python=3.10
conda activate ies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd ImageNet-ES/
pip install -r requirements.txt
```

### 2. Run scripts

#### 2.1 Run Labeler: adopt ImageNet-ES to [openood](https://github.com/Jingkang50/OpenOOD) evaluation api.
```
python get_label_lists.py -model ${MODEL_NAME} -gpu_num {DEVICE_NUM} -bs ${BATCH_SIZE}
```

#### 2.2 Run OOD Evaluator
```
python ood_exp.py -model ${MODEL_NAME} -gpu_num {DEVICE_NUM} -id_name ${ID_NAME} -output_dir ./results -init {YES_OR_NO}
```
* If first run of the ood evaluator, set {YES_OR_NO} as "y".

#### 2.3 Available Options
* Available options of "MODLE_NAME" can be refereced by key of timm_config in configs/models/model_config.py.
* Available options of "ID_NAME"
    * "SC" : Semantics-centric framework setting
    * "MC" : Model-specific framework setting
    * "ES" : Enhancement setting of "MC" with ImageNet-ES
* Reference get_labeler_args and get_evalood_args in utils/experiment_setup.py for more options and details.

#### 2.4. Notebooks
* ood_plots.ipynb: Analysis for ood experiments
* qualitative-analysis.ipynb: Qualitative Analysis of sensor control and ImageNet-ES datasets

### Citations
* To be updated soon
