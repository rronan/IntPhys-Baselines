import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import os
from pydoc import locate

from .encoder import Encoder
from .decoder import Decoder
from .model import Model
import utils

class Linear_rnn(nn.Module, Model):
    def __init__(self, opt, test=False):
        super(Linear_rnn, self).__init__()
        self.__name__ = 'linear_rnn'
        self.input_len, self.target_len = opt.input_len, opt.target_len
        self.latentDim = opt.latentDim
        self.bsz = 1 if test else opt.bsz
        self.input = torch.FloatTensor(opt.bsz * opt.input_len, opt.nc_in, 64, 64)
        self.input = Variable(self.input)
        self.target = torch.FloatTensor(opt.bsz * opt.target_len, opt.nc_out, 64, 64)
        self.target = Variable(self.target)
        self.h0 = Variable(torch.randn(opt.n_layer, 1, opt.n_hidden))
        if opt.rnn == 'LSTM':
            self.h0 = (
                self.h0,
                Variable(torch.randn(opt.n_layer, 1, opt.n_hidden))
            )

        self.criterion = nn.MSELoss()
        self.recurrent_module = locate('torch.nn.%s' %opt.rnn)(
            opt.latentDim,
            opt.n_hidden,
            opt.n_layer
        )

        # does this must be done at the end of __init__ ?
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # def encoder and decoder as volatile
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, x, h):
        x, h = self.recurrent_module(x, h)
        return x, h

    def gpu(self):
        self.cuda()
        self.input = self.input.cuda()
        self.target = self.target.cuda()
        if type(self.h0) is list:
            self.h0[0] = self.h0[0].cuda()
            self.h0[1] = self.h0[1].cuda()
        else:
            self.h0 = self.h0.cuda()

    def step(self, batch, set_):
        self.input.data.copy_(batch[0])
        self.target.data.copy_(batch[1])
        x = self.encoder(self.input).detach()
        x = x.view(-1, self.input_len, self.latentDim)
        y = self.encoder(self.target).detach()
        y = y.view(-1, self.input_len, self.latentDim)
        self.out, _ = self.forward(x, self.h0)
        err = self.criterion.forward(self.out, y)
        if set_ == 'train':
            self.zero_grad()
            err.backward()
            self.optimizer.step()
        return {'err': err.data[0]}

    def output(self):
        out = self.decoder(self.out.view(-1, self.latentDim)).data
        d1, d2, d3 = out.size(1), out.size(2), out.size(3)
        return out.view(-1, self.target_len, d1, d2, d3)

    def load(self, d):
        for e in d:
            if e[0] == self.__name__:
                path = d[0][-1]
                print('loading %s: %s' %(self.__name__, path))
                self.recurrent_module.load_state_dict(torch.load(path))
            if e[0] == 'encoder':
                path = d[0][-1]
                print('loading encoder: %s' %path)
                self.encoder.load_state_dict(
                    utils.filter(
                        torch.load(path),
                        ['resnet_features', 'encoder']
                    )
                )
            if e[0] == 'decoder':
                path = d[0][-1]
                print('loading decoder: %s' %path)
                self.decoder.load_state_dict(
                    utils.filter(
                        torch.load(path),
                        ['decoder', 'deconv']
                    )
                )

    def save(self, path, epoch):
        f = open(os.path.join(path, '%s.txt' %self.__name__), 'w')
        f.write(str(self))
        f.close()
        torch.save(
            self.recurrent_module.state_dict(),
            os.path.join(path, '%s_%d.pth' %(self.__name__, epoch))
        )

    def score():
        self.input.data.copy_(batch[0])
        self.target.data.copy_(batch[1])
        x = self.encoder(self.input).detach()
        x = x.view(-1, self.input_len, self.latentDim)
        y = self.encoder(self.target).detach()
        y = y.view(-1, self.input_len, self.latentDim)
        self.out, _ = self.forward(x, self.h0)
        err = self.criterion.forward(self.out, y)
        return err.data[0]
