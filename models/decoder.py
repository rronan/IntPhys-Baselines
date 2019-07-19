import torch.nn as nn
from torch.autograd import Variable
import torch


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        # define model
        self.nc_out = opt.nc_out
        self.frame_height = opt.frame_height
        self.frame_width = opt.frame_width
        self.decoder = nn.Linear(opt.latentDim, 128 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.Conv2d(128, opt.nf * 4, 3, 1, 1),
            nn.BatchNorm2d(opt.nf * 4),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(opt.nf * 4, opt.nf * 2, 3, 1, 1),
            nn.BatchNorm2d(opt.nf * 2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(opt.nf * 2, opt.nf, 3, 1, 1),
            nn.BatchNorm2d(opt.nf),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(opt.nf, opt.target_len * opt.nc_out, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder.forward(x)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv(x)
        x = x.view(-1, self.nc_out, self.frame_height, self.frame_width)
        return x
