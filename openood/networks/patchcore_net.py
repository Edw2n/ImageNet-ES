import torch
import torch.nn as nn


class PatchcoreNet(nn.Module):
    def __init__(self, backbone):
        super(PatchcoreNet, self).__init__()

        self.backbone = backbone

        for param in self.parameters():
            param.requires_grad = False
        backbone.cuda()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, x, return_feature):
        _, feature_list = self.backbone(x, return_feature_list=True)
        return [feature_list[-3], feature_list[-2]]

    # def init_features(self):
    #     self.features = []

    # def forward(self, x_t, return_feature):
    #     x_t = x_t.cuda()
    #     self.init_features()
    #     _ = self.module(x_t)

    #     import pdb
    #     pdb.set_trace()

    #     return self.features
