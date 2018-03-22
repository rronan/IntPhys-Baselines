import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models
from torch.autograd import Variable
import torch
import torch.optim as optim
import copy
import utils

from .model import Model

class Resnet_ae(nn.Module, Model):
    def __init__(self, opt, test=False, input_=None, target=None):
        super(Resnet_ae, self).__init__()
        self.__name__ = 'resnet_ae'
        # define variables
        bsz = 1 if test else opt.bsz
        if input_:
            self.input = input_
        else:
            self.input = torch.FloatTensor(bsz * opt.input_len, opt.nc_in, 64, 64)
            self.input = Variable(self.input)
        if target:
            self.target = target
        else:
            self.target = torch.FloatTensor(bsz * opt.target_len, opt.nc_out, 64, 64)
            self.target = Variable(self.target)
        self.criterion = nn.MSELoss()

        if opt.instanceNorm:
            Norm = nn.InstanceNorm2d
        else:
            Norm = nn.BatchNorm2d
        # define model
        self.nc_out = opt.nc_out
        self.latentDim = opt.latentDim
        self.input_len, self.target_len = opt.input_len, opt.target_len
        self.frame_height, self.frame_width = opt.frame_width, opt.frame_height
        resnet = torchvision.models.resnet18(True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:6])

        middleNL = nn.Sigmoid() if opt.middleNL == 'sigmoid' else nn.Tanh()
        self.encoder = nn.Sequential(
            nn.Linear(128 * 8 * 8, opt.latentDim),
            middleNL
        )
        self.decoder = nn.Linear(
            opt.input_len * opt.latentDim,
            opt.target_len * 128 * 8 * 8
        )

        self.deconv = nn.Sequential(
            nn.Conv2d(128,opt.nf*4, 3, 1, 1),
            Norm(opt.nf*4),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(opt.nf*4,opt.nf*2, 3, 1, 1),
            Norm(opt.nf*2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(opt.nf*2,opt.nf, 3, 1, 1),
            Norm(opt.nf),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(opt.nf, opt.target_len * opt.nc_out, 3, 1, 1),
            nn.Sigmoid()
        )


        # define maskPredictor
        if opt.maskPredictor:
            optmp = {
                'frame_width': opt.frame_width,
                'frame_height': opt.frame_height,
                'input_len': 1,
                'target_len': 1,
                'nc_in': opt.nc_in,
                'nc_out': opt.nc_out,
                'nf': opt.nf,
                'latentDim': 128,
                'instanceNorm': False,
                'middleNL': opt.middleNL,
                'bsz': None,
                'maskPredictor': None,
                'lr': None,
                'beta1': None,
            }
            optmp = utils.to_namespace(optmp)
            self.maskPredictor = Resnet_ae(
                optmp, False, self.target, self.target
            ).eval()
            self.maskPredictor.load(opt.maskPredictor)
        else:
            self.maskPredictor = None

        # does this have to be done at the end of __init__ ?
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def forward(self, x):
        x = self.resnet_features.forward(x)
        x.detach()
        x = x.view(-1, 128 * 8 * 8)
        x = self.encoder.forward(x)
        x = x.view(-1, self.input_len * self.latentDim)
        x = self.decoder.forward(x)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv(x)
        x = x.view(-1, self.nc_out, self.frame_height, self.frame_width)
        return x

    def gpu(self):
        self.cuda()
        self.input = self.input.cuda()
        self.target = self.target.cuda()
        if self.maskPredictor:
            self.maskPredictor.cuda()

    def step(self, batch, set_):
        self.input.data.copy_(batch[0])
        self.target.data.copy_(batch[1])
        if self.maskPredictor:
            self.target = self.maskPredictor(self.target).detach()
        self.out = self.forward(self.input)
        err = self.criterion.forward(self.out, self.target)
        if set_ == 'train':
            self.zero_grad()
            err.backward()
            self.optimizer.step()
        return {'err': err.data[0]}

    def output(self):
        d1, d2, d3 = self.out.size(1), self.out.size(2), self.out.size(3)
        return self.out.view(-1, self.target_len, d1, d2, d3).data

    def score(self, batch):
        self.input.data.copy_(batch[0])
        self.target.data.copy_(batch[1])
        if self.maskPredictor:
            self.target = self.maskPredictor(self.target).detach()
        self.out = self.forward(self.input)
        err = self.criterion.forward(self.out, self.target)
        return 1 / err.data[0]
