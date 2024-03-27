import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import torch as ch
from scipy.stats import gaussian_kde
import os

from .visual_encoding import METHOD_DICT, HIST_COLOR


class Histogram():
    def __init__(self, fig_size=(5,7.5)):
        self.fig_size = fig_size
    
    def normalized_kde(self, target):
        kde = gaussian_kde(target)
        x = np.linspace(target.min(), target.max(), 1000)
        normalized_density = normalize([kde(x)], norm='l1').reshape(x.shape)
        return normalized_density, x

    def draw_filled_hist(self, data, color, label, transparency=0.2, linestyle='-'):
        y, x = self.normalized_kde(data)
        plt.plot(x, y, color=color, label=label, linestyle=linestyle)
        zero = x*0
        plt.fill_between(x, y, zero, where=(y > zero), color=color, alpha=transparency)
        return x.min(), x.max()

    def draw_scood(self, scores: dict, methods: list, s_ood=None, legend_loc=None, file_name=None):
        '''
        plot ood score distributions of trained with semantics-centric OOD settings

        [in]
        - scores: score data {'method_name': 'score_info'} (dict)
        - methods: target methods to plot (list)
        - s_ood: target ood dataset name (str)
        - legend_loc: one of the methods (str)

        [out]
        - None
        '''
        
        coords = (len(methods), 1)
        fig = plt.figure(figsize=self.fig_size)
        for idx, method in enumerate(methods):
            m_scores = scores[method]
            plt.subplot(*coords, idx+1)
            
            idid = m_scores['id']['test'][1]
            if s_ood:
                if s_ood in m_scores['ood']['far']:
                    standard_ood_scores = m_scores['ood']['far'][s_ood][1]
                else:
                    standard_ood_scores = m_scores['ood']['near'][s_ood][1]
            else: # s_ood = union of every ood datasets
                far_ood_scores = np.concatenate([val[1] for val in m_scores['ood']['far'].values()])
                near_ood_scores = np.concatenate([val[1] for key,val in m_scores['ood']['near'].items() if 'param' not in key and 'sample' not in key])
                standard_ood_scores = np.concatenate([far_ood_scores, near_ood_scores])
            
            es_scores = np.concatenate([val[1] for key,val in m_scores['ood']['near'].items() if 'test' in key])
            
            self.draw_filled_hist(idid, color=HIST_COLOR['ID'], label='ID', transparency=0.2)
            self.draw_filled_hist(standard_ood_scores, color=HIST_COLOR['S-OOD'], label='S-OOD', transparency=0.2)
            self.draw_filled_hist(es_scores, color=HIST_COLOR['ImageNet-ES'], label='ImageNet-ES', transparency=0.85)
            
            if idx+1 == coords[0]*coords[1]:
                plt.xlabel('OOD score', fontsize=15)
                
            plt.ylabel('Density', fontsize=15)
            plt.title(f'{METHOD_DICT[method]}', fontsize=17)
            
            if method==legend_loc:
                plt.legend()
            plt.tight_layout()
        
        if file_name:
            plt.savefig(file_name)
        plt.show()
        

    def draw_msood(self, scores: dict, methods: list, s_ood=None, legend_loc=None, file_name=None):
        '''
        plot ood score distributions of trained with specific OOD settings

        [in]
        - scores: score data {'method_name': 'score_info'} (dict)
        - methods: target methods to plot (list)
        - s_ood: target ood dataset name (str)
        - legend_loc: one of the methods (str)

        [out]
        - None
        '''
        
        fig = plt.figure(figsize=self.fig_size)
        coords = (len(methods), 1)
        
        for idx, method in enumerate(methods):
            m_scores = scores[method]
            plt.subplot(*coords,idx+1)
            
            idid = m_scores['id']['test'][1]
            idood = m_scores['ood']['near']['test_id-'][1] 
            
            if s_ood:
                if s_ood in m_scores['ood']['far']:
                    s_ood_scores = m_scores['ood']['far'][s_ood][1]
                else:
                    s_ood_scores = scores['ood']['near'][s_ood][1]
            else:
                far_ood_scores = np.concatenate([val[1] for val in m_scores['ood']['far'].values()])
                near_ood_scores = np.concatenate([val[1] for key,val in m_scores['ood']['near'].items() if 'param' not in key and 'sample' not in key])
                s_ood_scores = np.concatenate([far_ood_scores, near_ood_scores])
            es_scores_id = np.concatenate([val[1] for key,val in m_scores['ood']['near'].items() if 'test' in key and 'id' in key])
            es_scores_ood = np.concatenate([val[1] for key,val in m_scores['ood']['near'].items() if 'test' in key and 'ood' in key])
            

            self.draw_filled_hist(idid, color=HIST_COLOR['ID+'], label='ID+', transparency=0.2)
            self.draw_filled_hist(idood, color=HIST_COLOR['ID-'], label='ID-', transparency=0.2, linestyle='--')
            self.draw_filled_hist(es_scores_id, color=HIST_COLOR['ImageNet-ES+'], label='ImageNet-ES+', transparency=0.2)
            self.draw_filled_hist(es_scores_ood, color=HIST_COLOR['ImageNet-ES-'], label='ImageNet-ES-', transparency=0.2, linestyle='--')
            
            if idx+1 == coords[0]*coords[1]:
                plt.xlabel('OOD score', fontsize=15)
            
            plt.title(f'{METHOD_DICT[method]}', fontsize=17)
            if method==legend_loc:
                plt.legend(ncol=2)
            plt.tight_layout()
        
        if file_name:
            plt.savefig(file_name)
        plt.show()

