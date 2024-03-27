import pickle
import matplotlib.cm as cm
import torch as ch
import matplotlib.pyplot as plt
from utils.visual_encoding import CLASS_MAP

class GridCanvas():
    aps = None
    isos = None
    sss = None
    def __init__(self, file, color_encoding='viridis'):
        self.get_grids(file)
        self.color_encoding = color_encoding
    
    def get_grids(self, file):
        log2ap = lambda x: float(x['taken_aperture'][1:])
        log2iso = lambda x: float(x['taken_iso'])/100
        log2ss = lambda x: float(x['taken_ss'][2:]) # inverse of speed
        with open(file, 'rb') as f:
            data = pickle.load(f)
            examples = list(data.values())[0].values()
            self.aps = list(map(log2ap, examples))
            self.isos = list(map(log2iso, examples))
            self.sss = list(map(log2ss, examples))

    def draw_grid_plots(self, accs_info, models, file_name=None, color_encoding=None):
        if not color_encoding:
            color_encoding = self.color_encoding
        fig = plt.figure(figsize=(8, 4))
        axs = []
        for idx, model in enumerate(models):
            accs1 = accs_info[model]['l1']
            data_color1 = [x/100 for x in accs1]
            my_cmap = plt.colormaps.get_cmap(color_encoding)
            colors1 = my_cmap(data_color1)

            ax1 = fig.add_subplot(2,len(models),idx+1, projection='3d')
            ax1.patch.set_facecolor('white')
            ax1.scatter(self.isos, self.aps, self.sss, color=colors1, marker='o', s=15)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])

            ax1.w_xaxis.set_pane_color('lightgrey')
            ax1.w_yaxis.set_pane_color('darkgrey')
            ax1.w_zaxis.set_pane_color('grey')

            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.get_zaxis().set_visible(False)

            ax1.set_xlabel('iso', linespacing=1)
            ax1.set_ylabel('aperture')
            ax1.set_zlabel('ss')
            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = -12
            ax1.zaxis.labelpad = -12
            
            if model=='resnet50':
                model = 'ResNet50'
            ax1.set_title(f'{model} ')
            
            sm = cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,1))
            sm.set_array([])
            axs.append(ax1)
        cbar = fig.colorbar(sm, orientation='vertical', ax=axs, shrink=0.75)
        cbar.set_label('Accuracy')
        if file_name:
            plt.savefig(file_name, bbox_inches='tight')
    
    def draw_class_wise_grid_plots(self, accs_info, file_name=None, targets=None, rc=None, figsize=None, color_encoding=None):
        
        if not color_encoding:
            color_encoding = self.color_encoding
        if not figsize:
            figsize=(60, 40)
        if not rc:
            rc = (10, 20)
        fig = plt.figure(figsize=figsize)
        if not targets:
            targets = accs_info.keys()
        axs = []
        
        for idx, class_idx in enumerate(targets):
            accs = accs_info[class_idx]
            data_color = [x for x in accs]
            my_cmap = plt.colormaps.get_cmap(color_encoding)
            colors1 = my_cmap(data_color)
            
            ax1 = fig.add_subplot(rc[0],rc[1], idx+1, projection='3d')
            ax1.patch.set_facecolor('white')
            ax1.scatter(self.isos, self.aps, self.sss, color=colors1, marker='o', s=15)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])

            ax1.w_xaxis.set_pane_color('lightgrey')
            ax1.w_yaxis.set_pane_color('darkgrey')
            ax1.w_zaxis.set_pane_color('grey')

            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.get_zaxis().set_visible(False)

            ax1.set_xlabel('iso', linespacing=1)
            ax1.set_ylabel('aperture')
            ax1.set_zlabel('ss')
            ax1.xaxis.labelpad = -12
            ax1.yaxis.labelpad = -12
            ax1.zaxis.labelpad = -12
            ax1.set_title(f"'{CLASS_MAP[class_idx]}'", fontsize=15)
            sm = cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,1))
            sm.set_array([])
            axs.append(ax1)
        cbar = fig.colorbar(sm, orientation='vertical', ax=axs, shrink=0.75)
        cbar.set_label('Accuracy')
        
        if file_name:
            plt.savefig(file_name, bbox_inches='tight')

