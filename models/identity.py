import torch.nn as nn
import torch
from torch.autograd import Variable
import utils

from .model import Model

class Identity(nn.Module, Model):

    def __init__(self, opt):
        super(Identity, self).__init__()

    def forward(self, x):
        self.x = x
        return self.x

    def step(self, batch, set_):
        return {}

    def output(self):
        return self.x.data

    def save(self, path, epoch):
        pass

    def load(self, path):
        pass
