from torchvision import datasets
import torch as ch
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm

# TODO: 이거 배포할 때는 밑에 Reconstuctor, EffectDataset는 빼고 배포해야함. (sensing for dnn)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class Reconstructor():
    def __init__(self, s_dict):
        self.s_dict = s_dict
        
    def to_ss_val(self, ss_str):
        ss_str = ss_str.replace('"','.').split('/')
        if len(ss_str)>1:
            ss = float(ss_str[0])/float(ss_str[1])
        else:
            ss = float(ss_str[0])
        return ss

    def to_aperture_val(self, ap_str):
        ap = float(ap_str.replace('f',''))
        return ap

    def get_setting_diff(self, ref_idx, after_idx):
        # someting 
        ref_iso = self.s_dict.loc[ref_idx-1,'iso']
        ref_ss = self.to_ss_val(self.s_dict.loc[ref_idx-1,'ss'])
        ref_aperture = self.to_aperture_val(self.s_dict.loc[ref_idx-1,'aperture'])
        delta_iso = self.s_dict.loc[ref_idx-1,'iso'] - self.s_dict.loc[after_idx-1,'iso']
        delta_ss = self.to_ss_val(self.s_dict.loc[ref_idx-1,'ss']) - self.to_ss_val(self.s_dict.loc[after_idx-1,'ss'])
        delta_aperture = self.to_aperture_val(self.s_dict.loc[ref_idx-1,'aperture']) - self.to_aperture_val(self.s_dict.loc[after_idx-1,'aperture'])
        return ch.tensor([ref_iso, ref_ss, ref_aperture]).double(), ch.tensor([-delta_iso, -delta_ss, -delta_aperture]).double()

    def reconstruct_data(self, result_data):
        reconstructed = []
        for _, _, ref_idx, ref_fv  in zip(*result_data):
            for _, after_score, after_idx, after_fv  in zip(*result_data):
                ref_settings, changes = self.get_setting_diff(ref_idx, after_idx)
                reconstructed.append((ref_fv, after_fv, ref_settings, changes, after_score))
        return reconstructed

    def get_full_reconstructs(self, loaded, sample_nums=2000):
        reconstructed = []
        for s_idx in tqdm(range(sample_nums)):
            d = loaded[f'recon_{s_idx}']
            reconstructed += self.reconstruct_data(d)
        return reconstructed

class EffectDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, ind):
        ref_fv = self.data[ind][0].astype(np.float64)
        after_fv = self.data[ind][1].astype(np.float64)
        ref_setting = self.data[ind][2]
        diff_setting = self.data[ind][3]
        
        normed_ref_iso = ref_setting[0] / (40000-100)
        normed_ref_ss = ref_setting[1] / (1-1/4000)
        normed_ref_aps = ref_setting[2] / 18.0

        normed_iso = diff_setting[0] / (40000-100)
        normed_ss = diff_setting[1] / (1-1/4000)
        normed_aps = diff_setting[2] / 18.0
        score = self.data[ind][-1].astype(np.float64)
        return (ref_fv, after_fv, (normed_ref_iso, normed_ref_ss, normed_ref_aps), (normed_iso, normed_ss, normed_aps)), score

