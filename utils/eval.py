from openood.evaluation_api import Evaluator
from openood.evaluation_api.datasets import DATA_INFO

from utils.framework_setter import FrameworkSetter
from utils.wrapper_nn import W_NN
from utils.experiment_setup import load_model

import torch as ch
from pathlib import Path


class Evaluator_ES():
    def __init__(self, setting_id_name, args, data_root_dir, output_dir='./results', device_num=0):
        self.setting_id_name = setting_id_name
        
        model, model_kwargs = load_model(args.model, device_num)
        if not model:
            return None
        self.model = model
        self.model_arch = model_kwargs['arch']
        self.bs = int(args.bs)

        self.result_dir = f'{output_dir}/{self.model_arch}'
        Path(f'{self.result_dir}').mkdir(parents=True, exist_ok=True)
        
        if args.init != 'n':
            evaluator.init_datasets()
        
        self.set_data_info(data_root_dir, args)

    def init_datasets(self):
        evaluator = Evaluator(
            self.model,
            id_name='imagenet200',                     # the target ID dataset
            data_root='./data',                    # change if necessary
            config_root=None,                      # see notes above
            preprocessor=None,                     # default preprocessing for the target ID dataset
            postprocessor_name='react', # the postprocessor to use
            postprocessor=None,                    # if you want to use your own postprocessor
            batch_size=self.bs,                        # for certain methods the results can be slightly affected by batch size
            shuffle=False,
            num_workers=8)   
    
    def set_data_info(self, DATA_ROOT_DIR, args):
        DATA_INFO[self.setting_id_name] = DATA_INFO['TIN2-STANDARD'] #default skeleton
        setter = FrameworkSetter(DATA_ROOT_DIR, DATA_INFO[self.setting_id_name], vars(args), self.model_arch)    
        setter.set_data_info(self.setting_id_name)
    
    def eval_ood(self, fsood_in=False, score_return=False, target_postprocessor=None):
        eval_results = {}
        score_results = {}

        if not target_postprocessor:
            target_postprocessor = ['msp', 'odin', 'react', 'vim']

        for pp in target_postprocessor:
            scores = None
            evaluator = Evaluator(
                self.model,
                id_name=self.setting_id_name,          # the target ID dataset
                data_root='./data',                    # change if necessary
                config_root=None,                      # see notes above
                preprocessor=None,                     # default preprocessing for the target ID dataset
                postprocessor_name=pp,                 # the postprocessor to use
                postprocessor=None,                    # if you want to use your own postprocessor
                batch_size=self.bs,                    # for certain methods the results can be slightly affected by batch size
                shuffle=False,
                num_workers=4)
            if score_return:
                metrics, scores = evaluator.eval_ood(fsood=fsood_in, score_return=True)
            else:
                metrics = evaluator.eval_ood(fsood=fsood_in)
            eval_results[pp] = metrics
            score_results[pp] = scores
            FILE_NAME = f'{self.result_dir}/{self.setting_id_name}_{pp}_results.pt'
            SCORES = f'{self.result_dir}/{self.setting_id_name}_{pp}_scores.pt'
            ch.save(eval_results, FILE_NAME)
            ch.save(score_results, SCORES)
            
            print(self.setting_id_name, pp)
            print(eval_results[pp])
            print(f'{FILE_NAME} is saved.')
            print(f'{SCORES} is saved.')
            
        FILE_NAME = f'{self.result_dir}/{self.setting_id_name}.pt' 
        SCORES = f'{self.result_dir}/{self.setting_id_name}_scores.pt' 
        ch.save(eval_results, FILE_NAME)
        print(f'{FILE_NAME} is saved.')
        ch.save(score_results, SCORES)
        print(f'{SCORES} is saved.')
        return eval_results, score_results