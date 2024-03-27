from sklearn import metrics
import numpy as np
from collections import defaultdict
from configs.datasets import OPS_NUM, ENVS
# Setting naming convention = 'XX{env_name}-param_{op_num}XX' 꼴이어야함.

class ScoreManager():
    
    def __init__(self, target_dataset_config=None):
        if not target_dataset_config:
            target_dataset_config = {
                'ENV_LIST': ENVS['es-test'],
                'NUM_OF_PARAM_OPTIONS': OPS_NUM['es-test']
            }
        self.env_list = target_dataset_config['ENV_LIST']
        self.num_of_param_options = target_dataset_config['NUM_OF_PARAM_OPTIONS']
        self.flatten_setting_keys = self.get_flatten_keys()
        
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_f1_score(self, pred, conf, label, tpr_rate=0.95):
        np.set_printoptions(precision=3)
        recall = 0.95
        ood_indicator = np.zeros_like(label)
        ood_indicator[label == -1] = 1

        # in the postprocessor we assume ID samples will have larger
        # "conf" values than OOD samples
        # therefore here we need to negate the "conf" values
        
        # precision_in, recall_in, thresholds_in \
        #     = metrics.precision_recall_curve(ood_indicator, -conf)
        
        
        # f1_score_in = 2*precision_in*recall_in / (precision_in+recall_in)

        precision_out, recall_out, thresholds_out \
            = metrics.precision_recall_curve(1 - ood_indicator, conf)
        fpr_list, tpr_list, thresholds = metrics.roc_curve(1 - ood_indicator, conf)
        
        f1_score = 2*precision_out*recall_out / (precision_out+recall_out)
        tau = thresholds[self.find_nearest(tpr_list, tpr_rate)]
        idx = self.find_nearest(thresholds_out,tau)
        return f1_score[idx]

    def get_f1_score_param(self, setting_name, scores): #TODO 고쳐야함
        id_name = f'test-{setting_name}-id'
        ood_name = f'test-{setting_name}-ood'
        id_pred, id_conf, id_gt  = scores['ood']['near'][id_name]
        ood_pred, ood_conf, ood_gt  = scores['ood']['near'][ood_name]
        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, np.full(ood_gt.shape, -1)])
        return self.get_f1_score(pred, conf, label)

    def get_setting_group_name(self, key_name):
        # check the key_name is f'{env_name}-param_{op_num}-' format
        splits = key_name.split('-param_')
        if len(splits)>=2:
            env_name = splits[0].split('-')[-1]
            op_num = splits[1].split('-')[0]
            if (env_name in self.env_list) and (0<int(op_num)<self.num_of_param_options+1):
                return self.map_flatten_key(env_name, op_num)
        return None

    def get_group_mean_scores(self, score):
        targets = score['ood']['near']
        my_dict = defaultdict(list)
        for (group, val) in targets.items():
            group_key = self.get_setting_group_name(group)
            if group_key:
                my_dict[group_key].append(val[1])
        group_means = map(lambda vals: np.concatenate(vals).mean(), my_dict.values())
        mean_info = dict(zip(my_dict.keys(), group_means))
        
        mean_scores = list(map(lambda setting_name: mean_info[setting_name], self.flatten_setting_keys))
        return mean_scores

    def get_acc_list(self, accs:dict) -> list:
        # dict to list (flatten) by key_order
        accs_vals = sum([list(accs[env_name].values()) for env_name in self.env_list],[])    
        accs_vals = list(map(lambda x: 100*x, accs_vals))
        accs_key = sum([list(map(lambda op_num: self.map_flatten_key(env_name, op_num), accs[env_name].keys())) for env_name in self.env_list],[])    
        accs_flatten = dict(zip(accs_key, accs_vals))
        
        acc_list = list(map(lambda setting_name: accs_flatten[setting_name], self.flatten_setting_keys))
        return acc_list
    
    def get_f1_score_list(self, method_scores): # 이걸 scores 쪽으로 옮겨야 하나..?
        f1s = map(lambda setting_name: self.get_f1_score_param(setting_name, method_scores)*100, self.flatten_setting_keys)
        return list(f1s)        

    def get_flatten_keys(self):
        keys = []
        for env_name in self.env_list:
            for i in range(1, self.num_of_param_options+1):
                keys.append(self.map_flatten_key(env_name, i))
        return keys
    
    def map_flatten_key(self, env_name, op_num):
        return f'{env_name}-param_{op_num}'
        
        
        