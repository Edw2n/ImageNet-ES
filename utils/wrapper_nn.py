import torch
import torch.nn as nn
import timm
from robustness import model_utils
from functools import reduce 

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class W_NN(nn.Module):
    def __init__(self, **model_kwargs):
        super(W_NN, self).__init__()
        self.__dict__.update(model_kwargs)

        self.origin_nn = timm.create_model(self.arch, pretrained=True)

        getter_path = self.fv_name.split('.')
        fv_module_getter = reduce(getattr, [self.origin_nn, *getter_path[:-1]])
        fv_in = getattr(fv_module_getter, getter_path[-1]).in_features
        setattr(fv_module_getter, getter_path[-1], nn.Linear(fv_in, self.fv_out, bias=True))
        
        self.origin_nn.load_state_dict(torch.load(self.resume_path))
        self.features = None

        getter_path = self.fv_out_name.split('.')
        fv_out_getter = reduce(getattr, [self.origin_nn, *getter_path])
        fv_out_getter.register_forward_hook(get_features('fv'))

    def forward(self, x, with_latent=False, return_feature=False, return_feature_list=False):
        pred = self.origin_nn(x)
        if with_latent or return_feature:
            fv = features['fv']
            return pred, fv
        elif return_feature_list:
            fv = features['fv']
            feature_list = [fv]
            return pred, feature_list
        return pred
        
    def forward_threshold(self, x, threshold):
        _, fv = self.forward(x, with_latent=True)
        fv = fv.clip(max=threshold)
        pred = self.origin_nn.fc(fv)
        return pred

    def get_fc(self):
        fc = self.get_fc_layer()
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
    
    def get_fc_layer(self):
        
        getter_path = self.fv_name.split('.')
        fv_module_getter = reduce(getattr, [self.origin_nn, *getter_path[:-1]])
        fc = getattr(fv_module_getter, getter_path[-1])
        return fc
    
    def eval(self):
        return self.origin_nn.eval()
    
    def train(self, mode= True):
        return self.origin_nn.train(mode)