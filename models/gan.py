import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models
from torch.autograd import Variable
import torch
import torch.optim as optim
import utils
import os
import copy
import random

from .model import Model
from .resnet_ae import Resnet_ae


def UpSamplingConvolution(nIn, nOut):
    module = nn.Sequential(nn.UpsamplingNearest2d(2), nn.Conv2d(nIn, nOut, 3, 1, 1))
    return module


def SubPixelConvolution(nIn, nOut):
    module = nn.Sequential(nn.Conv2d(nIn, nOut * 4, 3, 1, 1), nn.PixelShuffle(2))
    return module


def SpatialFullConvolution(nIn, nOut):
    return nn.ConvTranspose2d(nIn, nOut, 4, 2, 1)


class netG(nn.Module):
    def __init__(self, opt):
        super(netG, self).__init__()
        self.__name__ = "netG"
        #        bsz = 1 if test else opt.bsz
        #        self.noise = Variable(torch.FloatTensor(bsz, opt.noiseDim, 1, 1))
        self.li = opt.nc_in * opt.input_len
        self.lo = opt.nc_out * opt.target_len
        self.frame_width, self.frame_height = opt.frame_width, opt.frame_height
        if opt.UpConv == "SpatialFullConvolution":
            UpConvolution = SpatialFullConvolution
        elif opt.UpConv == "UpSamplingConvolution":
            UpConvolution = UpSamplingConvolution
        elif opt.UpConv == "SubPixelConvolution":
            UpConvolution = SubPixelConvolution
        else:
            print("Unknown opt.UpConv")

        if opt.instanceNorm:
            Norm = nn.InstanceNorm2d
        else:
            Norm = nn.BatchNorm2d

        def enc(dim_in, nf, outputDim):
            outputDim = outputDim or latentDim
            # input is (nc) x 64 x 64
            model = nn.Sequential(
                nn.Conv2d(dim_in, nf, 4, 2, 1),
                Norm(nf),
                nn.ReLU(True),  # state size: (nf) x 32 x 32
                nn.Conv2d(nf, nf * 2, 4, 2, 1),
                Norm(nf * 2),
                nn.ReLU(True),  # state size: (nf * 2) x 16 x 16
                nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
                Norm(nf * 4),
                nn.ReLU(True),  # state size: (nf * 4) x 8 x 8
                nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
                Norm(nf * 8),
                nn.ReLU(True),  # state size: (nf * 8) x 4 x 4
                nn.Conv2d(nf * 8, outputDim, 4, 4),
                Norm(outputDim),
                nn.ReLU(),  # state size: (latentDim) x 1 x 1
            )
            return model

        def dec(dim_in, dim_out, nf):
            if opt.finalNL == "sigmoid":
                finalNL = nn.Sigmoid()
            elif opt.finalNL == "tanh":
                finalNL = nn.TanH()

            # state size: (dim_in) x 1 x 1
            model = nn.Sequential(
                nn.ConvTranspose2d(dim_in, nf * 8, 4),
                nn.ReLU(True),  # state size: (nf * 8) x 4 x 4
                UpConvolution(nf * 8, nf * 4),
                Norm(nf * 4),
                nn.ReLU(True),  # state size: (nf*4) x 8 x 8
                UpConvolution(nf * 4, nf * 2),
                Norm(nf * 2),
                nn.ReLU(True),  # state size: (nf*2) x 16 x 16
                UpConvolution(nf * 2, nf),
                Norm(nf),
                nn.ReLU(),  # state size: (nf) x 32 x 32
                UpConvolution(nf, dim_out),  # state size: (dim_out) x 64 x 64
                finalNL,
            )
            return model

        self.encoder = enc(self.li, opt.nf, opt.latentDim)
        self.decoder = dec(opt.latentDim + opt.noiseDim, self.lo, opt.nf)

    def forward(self, x):
        a = x[0].view(-1, self.li, self.frame_height, self.frame_width)
        a = a.contiguous()
        a = self.encoder.forward(a)
        x = torch.cat([a, x[1]], 1)
        x = self.decoder.forward(x)
        x = x.view(-1, self.lo, self.frame_height, self.frame_width)
        return x


class netD(nn.Module):
    def __init__(self, opt):
        super(netD, self).__init__()
        self.__name__ = "netD"
        self.gen_bis, self.two_heads = opt.gen_bis, opt.two_heads
        self.li = opt.nc_in * opt.input_len
        self.lo = opt.nc_out * opt.target_len
        self.frame_width, self.frame_height = opt.frame_width, opt.frame_height
        if opt.instanceNorm:
            Norm = nn.InstanceNorm2d
        else:
            Norm = nn.BatchNorm2d

        def path(nc, nf):
            # 64x64
            net = nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1),
                Norm(nf),
                nn.LeakyReLU(0.2, True),  # 32x32
                nn.Conv2d(nf, nf * 2, 4, 2, 1),
                Norm(nf * 2),
                nn.LeakyReLU(0.2, True),  # 16x16
                nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
                Norm(nf * 4),
                nn.LeakyReLU(0.2),  # 8x8
            )
            return net

        finalNL0 = nn.Sigmoid()
        finalNL1 = nn.Sigmoid()
        self.path0 = path(opt.nc_in * opt.input_len, opt.nf)
        r = opt.gen_bis + 1
        self.path1 = path(opt.nc_out * opt.target_len * r, opt.nf)
        self.head0 = nn.Sequential(
            nn.Conv2d(opt.nf * 8, opt.nf * 8, 4, 2, 1),
            Norm(opt.nf * 8),
            nn.LeakyReLU(0.2, True),  # 4x4
            nn.Conv2d(opt.nf * 8, 1, 4, 4),
            finalNL0,
        )
        if opt.two_heads:
            self.head1 = nn.Sequential(
                nn.Conv2d(opt.nf * 4, opt.nf * 8, 4, 2, 1),
                Norm(opt.nf * 8),
                nn.LeakyReLU(0.2, True),  # 4x4
                nn.Conv2d(opt.nf * 8, 1, 4, 4),
                finalNL1,
            )

        self.criterionD0 = nn.BCELoss()
        if opt.two_heads:
            self.criterionD1 = nn.BCELoss()
        self.weight_head0 = opt.weight_head0

    def forward(self, x):
        a = x[0].view(-1, self.li, self.frame_height, self.frame_width)
        b = x[1].view(-1, self.lo, self.frame_height, self.frame_width)
        if self.gen_bis:
            c = x[2].view(-1, self.lo, self.frame_height, self.frame_width)
            b = torch.cat([b, c], 1)
        a = self.path0.forward(a)
        b = self.path1.forward(b)
        x = torch.cat([a, b], 1)
        h0 = self.head0.forward(x)
        if self.two_heads:
            h1 = self.head1.forward(b)
            return [h0.view(-1), h1.view(-1)]
        else:
            return [h0.view(-1)]


class Gan(Model):
    def __init__(self, opt, test=False):
        self.__name__ = "gan"
        # define variables
        self.gen_bis = opt.gen_bis
        self.p_red = opt.p_red
        bsz = 1 if test else opt.bsz
        li, lo = bsz * opt.input_len, bsz * opt.target_len
        self.target_len = opt.target_len
        self.input = Variable(torch.FloatTensor(li, opt.nc_in, 64, 64))
        self.target = Variable(torch.FloatTensor(lo, opt.nc_out, 64, 64))
        self.noise = Variable(torch.FloatTensor(bsz, opt.noiseDim, 1, 1))
        self.r = Variable(torch.FloatTensor(bsz).fill_(opt.target_real))
        self.f = Variable(torch.FloatTensor(bsz).fill_(opt.target_fake))
        self.MSE = nn.MSELoss()

        # define model
        self.netG = netG(opt)
        self.netD = netD(opt)

        # custom weights initialization called on netG and netD
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # define maskPredictor
        if opt.maskPredictor:
            optmp = {
                "frame_width": opt.frame_width,
                "frame_height": opt.frame_height,
                "input_len": 1,
                "target_len": 1,
                "nc_in": opt.nc_in,
                "nc_out": opt.nc_out,
                "nf": opt.nf,
                "latentDim": 128,
                "instanceNorm": False,
                "middleNL": opt.middleNL,
                "bsz": None,
                "maskPredictor": None,
                "lr": None,
                "beta1": None,
            }
            optmp = utils.to_namespace(optmp)
            self.inputMaskPredictor = Resnet_ae(
                optmp, False, self.input, self.input
            ).eval()
            self.inputMaskPredictor.load(opt.maskPredictor)
            self.targetMaskPredictor = Resnet_ae(
                optmp, False, self.target, self.target
            ).eval()
            self.targetMaskPredictor.load(opt.maskPredictor)
        else:
            self.targetMaskPredictor = None
            self.inputMaskPredictor = None

        # define optimizer
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=opt.lrD, betas=(opt.beta1D, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=opt.lrG, betas=(opt.beta1G, 0.999)
        )

    def gpu(self):
        self.input = self.input.cuda()
        self.target = self.target.cuda()
        self.noise = self.noise.cuda()
        self.r = self.r.cuda()
        self.f = self.f.cuda()
        self.netG.cuda()
        self.netD.cuda()
        self.MSE.cuda()
        if self.inputMaskPredictor:
            # use cuda() and not gpu() here
            self.inputMaskPredictor.cuda()
        if self.targetMaskPredictor:
            # use cuda() and not gpu() here
            self.targetMaskPredictor.cuda()

    def step(self, batch, set_):
        if random.random() < self.p_red:
            self.input.data.fill_(0)
            self.input.data[0].fill_(1)
            self.target.data.fill_(0)
            self.target.data[0].fill_(1)
        else:
            self.input.data.copy_(batch[0])
            self.target.data.copy_(batch[1])
            if self.inputMaskPredictor:
                self.input = self.inputMaskPredictor(self.input).detach()
            if self.targetMaskPredictor:
                self.target = self.targetMaskPredictor(self.target).detach()
        input_G = [self.input, self.noise]
        if (
            self.gen_bis
        ):  # note this forward must come first, because we backprop Gforward_1
            self.noise.data.normal_(0, 1)
            gen1 = self.netG.forward(input_G).detach()
        else:
            gen1 = None

        def Dstep(x, y):
            out = self.netD.forward(x)
            res = self.netD.criterionD0(out[0], y)
            if self.netD.two_heads:
                res *= self.netD.weight_head0
                res += (1 - self.netD.weight_head0) * self.netD.criterionD1(out[1], y)
            return res, out

        # train D on real
        err_r, out = Dstep([self.input, self.target, gen1], self.r)
        p_real = out[0].mean()
        p_real_std = out[0].std()
        if set_ == "train":
            self.netD.zero_grad()
            err_r.backward()
        # train D on fake
        self.noise.data.normal_(0, 1)
        self.gen0 = self.netG.forward(input_G)
        self.netD.apply(utils.disableBNRunningMeanStd)
        err_f, out = Dstep([self.input, self.gen0.detach(), gen1], self.f)
        self.netD.apply(utils.enableBNRunningMeanStd)
        p_fake = out[0].mean()
        p_fake_std = out[0].std()
        if set_ == "train":
            err_f.backward()
            self.optimizerD.step()
            # train G on fake
            self.netG.zero_grad()
            self.noise.data.normal_(0, 1)
            err, out = Dstep([self.input, self.gen0, gen1], self.r)
            err.backward()
            self.optimizerG.step()

        mse = self.MSE.forward(self.gen0, self.target)
        out = {
            "err_r": err_r.data,
            "err_f": err_f[0],
            "mse": mse,
            "p_real": p_real,
            "p_fake": p_fake,
            "p_real_std": p_real_std,
            "p_fake_std": p_fake_std,
        }
        return utils.to_number(out)

    def output(self):
        d1, d2, d3 = self.gen0.size(1), self.gen0.size(2), self.gen0.size(3)
        return self.gen0.view(-1, self.target_len, d1, d2, d3).data

    def save(self, path, epoch):
        # save netG
        f = open(os.path.join(path, "netG.txt"), "w")
        f.write(str(self.netG))
        f.close()
        torch.save(self.netG.state_dict(), os.path.join(path, "netG_%s.pth" % epoch))
        # save netD
        f = open(os.path.join(path, "netD.txt"), "w")
        f.write(str(self.netD))
        f.close()
        torch.save(self.netD.state_dict(), os.path.join(path, "netD_%s.pth" % epoch))

    def load(self, d):
        for e in d:
            if e[0] == "netG":
                path = e[1]
                print("loading netG: %s" % path)
                self.netG.load_state_dict(torch.load(e[1]))
            if e[0] == "netD":
                path = e[1]
                print("loading netD: %s" % path)
                self.netD.load_state_dict(torch.load(e[1]))

    def score(self, batch):
        self.input.data.copy_(batch[0])
        self.target.data.copy_(batch[1])
        if self.inputMaskPredictor:
            self.input = self.inputMaskPredictor(self.input).detach()
        if self.targetMaskPredictor:
            self.target = self.targetMaskPredictor(self.target).detach()
        input_G = [self.input, self.noise]
        if (
            self.gen_bis
        ):  # note this forward must come first, because we backprop Gforward_1
            self.noise.data.normal_(0, 1)
            gen1 = self.netG.forward(input_G).detach()
        else:
            gen1 = None
        out = self.netD.forward([self.input, self.target, gen1])

        self.noise.data.normal_(0, 1)
        # this is need to plot generations at test time (debugging)
        self.gen0 = self.netG.forward(input_G).detach()
        return out[0].data[0]

    def eval(self):
        self.netG.eval()
        self.netD.eval()

    def parameters(self):
        params = []
        for m in self.netD.parameters():
            params.append(m)
        for m in self.netG.parameters():
            params.append(m)
        return params
