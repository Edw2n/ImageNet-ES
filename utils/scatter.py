import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from .visual_encoding import METHOD_DICT, S_OOD_CATEGORY, NF_MARKER, COLOR
from .scores import ScoreManager

class Scatters():

    def __init__(self, methods, figsize=(5,3), legend_fontsize=14, label_fontsize=15, target=None):
        '''
        [input]
        - methods: a list of target ood methods to plot
        '''
        self.methods = methods
        self.legend_fontsize = legend_fontsize
        self.label_fontsize = label_fontsize
        self.fig_size = figsize
        self.sm = ScoreManager()
    
    def plot_diff(self, mc, es, method, metric_name='FPR@95', file_name=None, legend_dict=None):
        plt.subplots(figsize=self.fig_size)
        
        S_OOD_DATASETS = list(S_OOD_CATEGORY.keys())
        S_OOD_CATEGORY_NAMES = list(S_OOD_CATEGORY.values())

        result_msood = mc[method][metric_name][S_OOD_DATASETS]
        result_es = es[method][metric_name][S_OOD_DATASETS]

        for i in range(len(S_OOD_CATEGORY_NAMES)):
            plt.plot([S_OOD_CATEGORY_NAMES[i], S_OOD_CATEGORY_NAMES[i]], [result_msood[i], result_es[i]], linestyle='dashed', color='orange', markevery=[1])  

        plt.scatter(S_OOD_CATEGORY_NAMES, result_msood, s=25, label='MS-OOD',  color='darkcyan')
        plt.scatter(S_OOD_CATEGORY_NAMES, result_es, s=25, label='Enhancement', marker='v',  color='orange')
        
        plt.xlabel('Datasets', fontsize=self.label_fontsize)
        plt.ylabel(f'{metric_name} (%)',  fontsize=self.label_fontsize)
        
        self.put_legend(plt, legend_dict)

        if file_name:
            plt.savefig(file_name, dpi=300)

        plt.show()

    def plot_sood_performance(self, metrics, target_metric='FPR@95', file_name=None, legend_dict=None):
        plt.subplots(figsize=self.fig_size)
        
        S_OOD_DATASETS = list(S_OOD_CATEGORY.keys())
        S_OOD_CATEGORY_NAMES = list(S_OOD_CATEGORY.values())

        for method in self.methods:
            result = metrics[method][target_metric][S_OOD_DATASETS]
            plt.scatter(S_OOD_CATEGORY_NAMES, result, s=25, label=METHOD_DICT[method], marker=NF_MARKER[method], color=COLOR[method])

        plt.xlabel('Datasets', fontsize=self.label_fontsize)
        plt.ylabel(f'{target_metric} (%)',  fontsize=self.label_fontsize)
        
        if not legend_dict:
            legend_dict = {'ncol': 2}
        self.put_legend(plt, legend_dict)
        
        if file_name:
            plt.savefig(file_name)
        plt.show()
    
    def plot_score_accuracy(self, scores, accs, axis_selection, legend_dict=None, file_name=None):
        
        fig, ax1 = plt.subplots(figsize=self.fig_size)
        
        ax2 = ax1.twinx()
        ax1s = '/'.join(map(lambda method: METHOD_DICT[method], axis_selection['ax1']))
        ax2s=  '/'.join(map(lambda method: METHOD_DICT[method], axis_selection['ax2']))
        ax1.set_xlabel('Classification ACC (%)', fontsize=self.label_fontsize)
        
        line_break = '\n'
        ax1.set_ylabel(f'{ax1s}{line_break if len(axis_selection["ax1"])>2 else ""} OOD score', fontsize=self.label_fontsize)
        ax2.set_ylabel(f'{ax2s} OOD score', fontsize=self.label_fontsize)
        
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        accs_vals = self.sm.get_acc_list(accs)
        
        for method in self.methods:
            score = scores[method]
            mean_scores = self.sm.get_group_mean_scores(score) # key order 지정 또는 활용은 나중에 생각해보기
            if method in axis_selection['ax1']:
                ax = ax1
            else:
                ax = ax2
            ax.scatter(np.array(accs_vals), np.array(mean_scores), color=COLOR[method], marker=NF_MARKER[method], label=METHOD_DICT[method])

        self.put_legend(fig, legend_dict)
        
        if file_name:
            plt.savefig(file_name)
        plt.show()
            
    def plot_f1_accuracy(self, full_scores, accs, file_name=None, legend_dict=None):
        fig, ax1 = plt.subplots(figsize=self.fig_size)
        accs_val = self.sm.get_acc_list(accs)
        for method in self.methods:
            f1s = self.sm.get_f1_score_list(full_scores[method])
            plt.scatter(accs_val, f1s, label=METHOD_DICT[method], s=25, marker=NF_MARKER[method], color=COLOR[method])
        plt.xlabel('Classification ACC (%)', fontsize=self.label_fontsize)
        plt.ylabel('F1 score (%)', fontsize=self.label_fontsize)

        self.put_legend(plt, legend_dict)

        if file_name:
            plt.savefig(file_name)
        plt.show()
    
    def plot_f1_accuracy_enhancement(self, full_scores, accs, file_name=None, legend_dict=None):
        fig, ax1 = plt.subplots(figsize=self.fig_size)
        accs_val = self.sm.get_acc_list(accs)
        for setting, scores in full_scores.items():
            f1s = self.sm.get_f1_score_list(scores)
            plt.scatter(accs_val, f1s, label=setting, s=25, marker=NF_MARKER[setting], color=COLOR[setting])
        plt.xlabel('Classification ACC (%)', fontsize=self.label_fontsize)
        plt.ylabel('F1 score (%)', fontsize=self.label_fontsize)
        
        self.put_legend(plt, legend_dict)

        if file_name:
            plt.savefig(file_name, dpi=300)
        plt.show()
    
    def put_legend(self, target, legend_dict):
        canvas = target
        if legend_dict:
            canvas.legend(**legend_dict, fontsize=self.legend_fontsize)
        else:
            canvas.legend(fontsize=self.legend_fontsize)
        canvas.tight_layout()
        
        
    

