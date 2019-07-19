import torch.nn as nn
import torchvision.models
from torch.autograd import Variable
import torch


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # define model
        self.nc_out = opt.nc_out
        self.frame_height = opt.frame_height
        self.frame_width = opt.frame_width
        resnet = torchvision.models.resnet18(True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:6])
        middleNL = nn.Sigmoid() if opt.middleNL else nn.Tanh()
        self.encoder = nn.Sequential(nn.Linear(128 * 8 * 8, opt.latentDim), middleNL)

    def forward(self, x):
        x = self.resnet_features.forward(x)
        x.detach()
        x = x.view(-1, 128 * 8 * 8)
        x = self.encoder.forward(x)
        return x
