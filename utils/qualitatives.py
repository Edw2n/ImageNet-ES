
import torch as ch
from PIL import Image
import seaborn as sns
import numpy as np
from robustness.tools.vis_tools import show_image_column
import matplotlib.pyplot as plt
import colorcet as cc
from sklearn.manifold import TSNE

from configs.tin_config import LOADER_CONFIG
from configs.datasets import DATA_ROOT_DIR, SAMPLE_DIR_MAP, OPS_NUM
from utils.visual_encoding import LABEL_DICT, COLOR

TRANSFORMATIONS = LOADER_CONFIG['DATA_TRANSFORMS']

class Interest():
    def __init__(self, accs_pc_df, accs_ae_df):
        self.accs_pc_df = accs_pc_df
        self.accs_ae_df = accs_ae_df
        
    def get_interest_settings(self, TARGET_ENV):
        TARGETS = self.accs_pc_df[self.accs_pc_df['Light']==TARGET_ENV]
        interest_settings = {
            'BEST5_SETTING': list(TARGETS.nlargest(5,'Acc.')['Camera Parameter'])[:5],
            'WORST5_SETTING': list(TARGETS.nsmallest(5,'Acc.')['Camera Parameter'])[:5],
            'AUTO5_SETTING': list(map(lambda i: f'param_{i}', range(1,5+1))),
        }
        return interest_settings 

class QualCanvas(Interest):
    def __init__(self, accs_pc_df, accs_ae_df, dataset_dir=DATA_ROOT_DIR, target_dir='es-test', sample_dir_map=SAMPLE_DIR_MAP):
        super().__init__(accs_pc_df, accs_ae_df)
        
        self.dataset_dir = dataset_dir
        self.target_dir = target_dir
        self.sample_dir_map = sample_dir_map

        l1_interests = self.get_interest_settings('l1')
        l5_interests = self.get_interest_settings('l5')
        
        # indices
        self.top5_l1 = [int(setting_name.split('_')[-1]) for setting_name in l1_interests['BEST5_SETTING']]
        self.top5_l5 = [int(setting_name.split('_')[-1]) for setting_name in l5_interests['BEST5_SETTING']]
        self.bot5_l1 = [int(setting_name.split('_')[-1]) for setting_name in l1_interests['BEST5_SETTING']]
        self.bot5_l5 = [int(setting_name.split('_')[-1]) for setting_name in l5_interests['BEST5_SETTING']]
        
        def to_avg_acc(target_settings, env, target_df=self.accs_pc_df):
            targets = target_df[target_df['Light']==env]
            return sum(list(targets[targets['Camera Parameter'].isin(target_settings)]['Acc.']))/len(target_settings)
        
        self.avg_acc = {
            'sampled': 86.3,
            'top5_l1': to_avg_acc(l1_interests['BEST5_SETTING'], 'l1'),
            'bot5_l1': to_avg_acc(l1_interests['WORST5_SETTING'], 'l1'),
            'ae5_l1': to_avg_acc(l1_interests['AUTO5_SETTING'], 'l1', target_df=self.accs_ae_df),
            'top5_l5': to_avg_acc(l5_interests['BEST5_SETTING'], 'l5'),
            'bot5_l5': to_avg_acc(l5_interests['WORST5_SETTING'], 'l5'),
            'ae5_l5': to_avg_acc(l5_interests['AUTO5_SETTING'], 'l5', target_df=self.accs_ae_df),
        }
    
    def set_target_dir(self, target_dir_name):
        self.target_dir = target_dir_name
    
    def load_img(self, FULL_PATH):
            try:
                image = Image.open(FULL_PATH)
                tensor = TRANSFORMATIONS['draw'](image)
            except Exception as e:
                print(e, FULL_PATH)
                tensor = ch.ones((3,224,224))
                tensor[1] *= 0
                tensor[2] *= 0
            return tensor
    
    def load_imgs(self, path, obj, splits=None):
        
        ROOT_DIR = f'{self.dataset_dir}/{self.target_dir}'
        obj_path = {
            'ae_l1': 'auto_exposure/l1',
            'ae_l5': 'auto_exposure/l5',
            'pc_l1': 'param_control/l1',
            'pc_l5': 'param_control/l5',
        }
        
        ops_num = OPS_NUM[self.target_dir]
        
        ae_num = splits if splits else 3
        obj_param_num = {
            'ae_l1': [f'param_{i+1}' for i in range(ae_num)],
            'ae_l5': [f'param_{i+1}' for i in range(ae_num)],
            'pc_l1': [f'param_{i}' for i in range(1,ops_num+1)],
            'pc_l5': [f'param_{i}' for i in range(1,ops_num+1)],
        }

        FINAL_ROOT_PATH = f'{ROOT_DIR}/{obj_path[obj]}'
        
        tensors = ch.stack(list(map(lambda obj_param: self.load_img(f'{FINAL_ROOT_PATH}/{obj_param}/{path}'), obj_param_num[obj])),axis=0)
        if splits:
            column_nums = len(tensors)//splits
            tensors_divided = [tensors[column_nums*i:column_nums*(i+1)] for i in range(splits)]
            return tensors_divided
        return tensors
    
    def draw_representatives(self, path, file_name=None, splits=3):
        pc_l1 = self.load_imgs(path,'pc_l1', splits=splits)
        pc_l5 = self.load_imgs(path,'pc_l5', splits=splits)
        ae_l1 = self.load_imgs(path,'ae_l1', splits=splits)
        ae_l5 = self.load_imgs(path,'ae_l5', splits=splits)
        empty = ch.ones(ae_l1[0].shape)

        rows = [
            *[ch.concat([s_ae, s_pc])for s_ae, s_pc in zip(ae_l1, pc_l1)],
            *[ch.concat([s_ae, s_pc])for s_ae, s_pc in zip(ae_l5, pc_l5)],
        ]
        
        titles_rows = [
            *[['On-AE']+ [f'On-{i*len(s_pc)+j+1}' for j in range(len(s_pc))] for i, (_, s_pc) in enumerate(zip(ae_l1, pc_l1))],
            *[['Off-AE']+ [f'Off-{i*len(s_pc)+j+1}' for j in range(len(s_pc))] for i, (_, s_pc) in enumerate(zip(ae_l5, pc_l5))],
        ]
        
        cols = [ ch.stack([*col]) for col in zip(*rows)]
        titles_cols = [ [*col] for col in zip(*titles_rows)]
        
        
        for i, sample in enumerate(cols[0]):
            if i % splits !=0:
                cols[0][i] = empty
                titles_cols[0][i] = ''
        
        ROOT_DIR = f'{self.dataset_dir}/{self.target_dir}/{self.sample_dir_map[self.target_dir]}'
        ORIGINAL_PATH = f'{ROOT_DIR}/{path}'
        original_img = self.load_img(ORIGINAL_PATH)
        show_image_column([[original_img]], tlist=[['Original sample']])
        
        show_image_column(cols,  ['' for i in range(1,len(cols)+1)], filename=file_name, tlist=titles_cols)
    
    def draw_qualitatives(self, path_list, picked_list, file_name=None):
        rows = []
        for path, picked in zip(path_list, picked_list):
            rows.append(self.get_quals(path, picked))
        titles = [
            f'sampled\n{self.avg_acc["sampled"]:.2f}%',
            f'light on top 5\n{self.avg_acc["top5_l1"]:.2f}%',
            f'light on bottom 5\n{self.avg_acc["bot5_l1"]:.2f}%',
            f'light on AE 5\n{self.avg_acc["ae5_l1"]:.2f}%',
            f'light off top 5\n{self.avg_acc["top5_l5"]:.2f}%',
            f'light off bottom 5\n{self.avg_acc["bot5_l5"]:.2f}%',
            f'light off AE 5\n{self.avg_acc["ae5_l5"]:.2f}%',
        ]
        self.draw_multiple_qualitatives(rows, titles, file_name=file_name)
     
    def draw_multiple_qualitatives(self, targets, titles, file_name=None):
        targets = [ch.concat([*xx]) for xx in zip(*targets)]
        print('Qualitative results of ImageNet-ES\n(avgerage accuracy of each group)')
        show_image_column(targets, titles, filename=file_name)
        return None   
    
    def get_quals(self, path, picked):
        b1, w1, b5, w5 = picked
        pc_l1 = self.load_imgs(path,'pc_l1')
        pc_l2 = self.load_imgs(path,'pc_l5')
        ae_l1 = self.load_imgs(path,'ae_l1')
        ae_l5 = self.load_imgs(path,'ae_l5')
        
        targets = [
            pc_l1[self.top5_l1[b1:b1+1]],
            pc_l1[self.bot5_l1[w1:w1+1]],
            ae_l1[:5],
            pc_l2[self.top5_l5[b5:b5+1]],
            pc_l2[self.top5_l5[w5:w5+1]],
            ae_l5[:5],
        ]
        
        ROOT_DIR = f'{self.dataset_dir}/{self.target_dir}/{self.sample_dir_map[self.target_dir]}'
        ORIGINAL_PATH = f'{ROOT_DIR}/{path}'
        image = Image.open(ORIGINAL_PATH)
        tensor = TRANSFORMATIONS['draw'](image)
        final = [ch.stack([tensor])]
        return final+targets

class FeatureCanvas(Interest):
    def __init__(self, accs_pc_df, accs_ae_df, fvs):
        super().__init__(accs_pc_df, accs_ae_df)
        self.fvs = fvs  
    
    def plot_vecs_n_labels(self, v, labels, ax=None):
        ax.axis('off')
        sns.set_style('darkgrid')
        palette = sns.color_palette(cc.glasbey, n_colors=200)
        sns.scatterplot(x=v[:,0], y=v[:,1], hue=labels, legend='brief', palette=palette, ax=ax, s=20)
        ax.get_legend().remove()

    def draw_embeddings(self, data_list, ax, perplexity=30, title=None):
        fs = []
        labels = []
        for data in data_list:
            fs += sum(data.values(),[])
            labels += sum(map(lambda x: [x]*5, data.keys()), [])
        fs = np.array(fs)
        pred_tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(fs)
        self.plot_vecs_n_labels(pred_tsne, labels,  ax=ax)
        if title:
            ax.set_title(f'{title}', fontsize=20)
    
    def draw_multiple_embeddings(self, fv_dict, figsize=(12,4), file_name=None):
        fig, axes = plt.subplots(1,len(fv_dict), figsize=figsize)
        for i, (setting_name, data) in enumerate(fv_dict.items()):
            self.draw_embeddings(data, axes[i], title=LABEL_DICT[setting_name])
        plt.tight_layout()
        
        if file_name:
            plt.savefig(file_name)
            
    def plot_activation_dist(self, activations, file_name=None):
        plt.figure(figsize=(6,2))
        
        for setting_name, fv_dicts in activations.items():
            targets = fv_dicts.values()
            fv_num = len(list(targets)[0][0])
            plt.plot(range(fv_num), np.array(sum(targets,[])).mean(axis=0), label=LABEL_DICT[setting_name], c=COLOR[setting_name])

        plt.xlabel('Feature representation indices', fontsize=10)
        plt.ylabel('Activation', fontsize=10)
        plt.legend()
        plt.tight_layout()
        
        if file_name:
            plt.savefig(file_name)
        
        



