from PIL import ImageFile
import torch as ch
from tqdm import tqdm
import numpy as np
from pathlib import Path

from utils.datasets import ImageFolderWithPaths
from configs.datasets import ENVS

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Labeler():

    def __init__(self, model, loader_info, arch_name, envs=ENVS['es-test']):
        self.model = model
        self.__dict__.update(loader_info) #bs, n_workers, transformations, device
        self.arch_name = arch_name
        self.envs = envs

        print('Labeler is set.')
    
    def generate_option_labels(self, OPS_NUM, ROOT_DIR, envs=None):
        '''
        Generate an image list with model-centric label files for each setting option (env-param_control)
        and aggregate them into two separate files(id/ood).
        '''
        
        if not envs:
            envs = self.envs
        # Generate an image list with model-centric label files for each setting option
        PARAM_CONTROL_IMGLIST_DIR = f'{ROOT_DIR}/{self.arch_name}/imglist/param_control'
        Path(PARAM_CONTROL_IMGLIST_DIR).mkdir(parents=True, exist_ok=True)
        for env in envs:
            for op_num in range(1,OPS_NUM+1):
                DATASET_ROOT_DIR = f'{ROOT_DIR}/param_control/{env}/param_{op_num}'
                paths_id, paths_ood = self.get_labeled_info(DATASET_ROOT_DIR, 'param_control')
                for label_type, paths_info in [('id', paths_id), ('ood', paths_ood)]:
                    TARGET_FILE = f'{PARAM_CONTROL_IMGLIST_DIR}/{env}-param_{op_num}_{label_type}.txt'
                    try:
                        with open(TARGET_FILE, 'w') as fp:
                            fp.write('\n'.join(paths_info))
                            fp.write('\n')
                    except Exception as e:
                        print(e)
        
        # Aggregate them into two separate files(id/ood).
        IMG_LIST_PATH = f'{ROOT_DIR}/{self.arch_name}/imglist'
        Path(IMG_LIST_PATH).mkdir(parents=True, exist_ok=True)
        for target_agg in ['id', 'ood']:
            cancated_txt = f"{IMG_LIST_PATH}/es_{target_agg}.txt"
            with open(cancated_txt, 'w') as f:
                for env in envs:
                    for op_num in range(1, OPS_NUM+1):
                        ID_FILE = f'{PARAM_CONTROL_IMGLIST_DIR}/{env}-param_{op_num}_{target_agg}.txt'
                        try:
                            with open(ID_FILE) as group_f:
                                for line in group_f:
                                    f.write(line)
                        except Exception as e:
                            print(e)

    def generate_sample_labels(self, ROOT_DIR, SAMPLE_DIR_NAME):
        '''
        Generate an image list with model-centric label files for sampled images in SAMPLE_DIR_NAME
        '''
        DATASET_ROOT_DIR = f'{ROOT_DIR}/{SAMPLE_DIR_NAME}'
        paths_id, paths_ood = self.get_labeled_info(DATASET_ROOT_DIR, SAMPLE_DIR_NAME)
        IMG_LIST_PATH = f'{ROOT_DIR}/{self.arch_name}/imglist'
        Path(IMG_LIST_PATH).mkdir(parents=True, exist_ok=True)
        
        for label_type, paths_info in [('id', paths_id), ('ood', paths_ood)]:
                TARGET_FILE = f"{IMG_LIST_PATH}/sample_{label_type}.txt"
                try:
                    with open(TARGET_FILE, 'w') as fp:
                        fp.write('\n'.join(paths_info))
                        fp.write('\n')
                except Exception as e:
                        print(e)
    
    def generate_standard_labels(self, ROOT_DIR, SAMPLE_DIR_NAME):
        '''
        Generate an image list with ground truth label files for sampled images in SAMPLE_DIR_NAME
        '''
        DATASET_ROOT_DIR = f'{ROOT_DIR}/{SAMPLE_DIR_NAME}'
        test_data = ImageFolderWithPaths(f'{DATASET_ROOT_DIR}', transform=self.transformations)
        paths_id = []
        test_loader = ch.utils.data.DataLoader(test_data, batch_size=self.bs, shuffle=False, num_workers=self.n_workers)
        it_sample = tqdm(enumerate(test_loader))
        with ch.no_grad():
            for kk, (X, y, paths) in it_sample:
                it_sample.set_description(f'{kk+1}/{len(test_loader)}')
                paths = [ f'{path} {label}' for path, label in zip(paths, y)]
                paths_id += np.array(paths).tolist()
        paths_id = [SAMPLE_DIR_NAME+line.split(SAMPLE_DIR_NAME)[-1] for line in paths_id]
        IMGLIST_DIR = f'{ROOT_DIR}/{self.arch_name}/imglist'
        Path(IMGLIST_DIR).mkdir(parents=True, exist_ok=True)
        ID_FILE = f'{IMGLIST_DIR}/sample_standard_id.txt'
        with open(ID_FILE, 'w') as fp:
            fp.write('\n'.join(paths_id))
    
    def get_param_accs(self, ROOT_DIR, OPS_NUM, envs=None):
        '''
        return a dictionary of accuracys for each settings(env-options)
        '''
        if not envs:
            envs = self.envs
            
        def get_accs(dataloader):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            self.model.eval()
            correct = 0
            it_sample = tqdm(enumerate(dataloader))
            with ch.no_grad():
                for kk, (X, y, paths) in it_sample:
                    it_sample.set_description(f'{kk+1}/{len(dataloader)}')
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    correct += (pred.argmax(1) == y).type(ch.float).sum().item()
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")
            return correct
        
        accs = {}
        for env in envs:
            accs[env] = {}
            for op_num in range(1, OPS_NUM+1):
                DATASET_ROOT_DIR = f'{ROOT_DIR}/param_control/{env}/param_{op_num}'

                test_data = ImageFolderWithPaths(f'{DATASET_ROOT_DIR}', transform=self.transformations)
                test_loader = ch.utils.data.DataLoader(test_data, batch_size=self.bs, shuffle=False, num_workers=self.n_workers)
                acc = get_accs(test_loader)
                accs[env][op_num] = acc 
        ch.save(accs, f'{ROOT_DIR}/{self.arch_name}_param_control_acc.pt')
        return accs

    def get_labeled_info(self, DATASET_ROOT_DIR, SAMPLE_DIR_NAME):
        '''
        return id imglist and ood imglist with path in SAMPLE_DIRNAME following model centric framework 
        '''
        def get_labels(test_loader):
            it_sample = tqdm(enumerate(test_loader))
            paths_id = []
            paths_ood = []
            with ch.no_grad():
                for kk, (X, y, paths) in it_sample:
                    it_sample.set_description(f'{kk+1}/{len(test_loader)}')
                    paths = [ f'{path} {label}' for path, label in zip(paths, y)]
                    X, y = X.to(self.device), y.to(self.device)
                    pred, fv = self.model(X, with_latent=True)

                    is_correct = (pred.argmax(1) == y)
                    paths_id += np.array(paths)[is_correct.cpu()].tolist()
                    paths_ood += np.array(paths)[~is_correct.cpu()].tolist()
            return paths_id, paths_ood
    
        test_data = ImageFolderWithPaths(f'{DATASET_ROOT_DIR}', transform=self.transformations)
        test_loader = ch.utils.data.DataLoader(test_data, batch_size=self.bs, shuffle=False, num_workers=self.n_workers)
        paths_id, paths_ood = get_labels(test_loader)
        paths_id = [SAMPLE_DIR_NAME+line.split(SAMPLE_DIR_NAME)[-1] for line in paths_id]
        paths_ood = [SAMPLE_DIR_NAME+line.split(SAMPLE_DIR_NAME)[-1] for line in paths_ood]
        return paths_id, paths_ood
        


