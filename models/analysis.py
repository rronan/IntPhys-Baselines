import torch.nn as nn
import torch
from torch.autograd import Variable
import utils

from .model import Model

class Analysis(nn.Module, Model):

    def __init__(self, opt):
        super(Analysis, self).__init__()
        # define variables
        self.input = torch.FloatTensor(opt.bsz * opt.input_len, opt.nc_in, 64, 64)
        self.input = Variable(self.input)
        self.bn = nn.BatchNorm2d(3)
        self.bn_de = nn.BatchNorm2d(3)
        self.target = torch.FloatTensor(opt.bsz * opt.target_len, opt.nc_out, 64, 64)
        self.target = Variable(self.target)

    def forward(self, x):
        self.x = x
        return self.x

    def step(self, batch, set_):
        self.input.data.copy_(batch[0])
        self.bn(self.input)
        self.bn_de(self.input)

        self.input.data.fill_(0)
        self.bn_de.apply(utils.disableBNRunningMeanStd)
        self.bn_de(self.input)
        self.bn_de.apply(utils.enableBNRunningMeanStd)
        print(self.bn.running_var.size())
        print(self.bn.running_mean.size())
        print('bn mean/var')
        print(self.bn.running_mean)
        print(self.bn.running_var)
        print('bn_de mean/var')
        print(self.bn_de.running_mean)
        print(self.bn_de.running_var)
        self.target.data.copy_(batch[1])
        self.out = self.forward(self.input)
        res = {'input_mean': batch[0].mean(), 'target_mean': batch[1].float().mean(),
            'input_min': batch[0].min(), 'target_min': batch[1].min(),
            'input_max': batch[0].max(), 'target_max': batch[1].max(),
        }
        return utils.to_number(res)

    def output(self):
        return self.out.data

    def save(self, path, epoch):
        pass

    def load(self, path):
        pass
