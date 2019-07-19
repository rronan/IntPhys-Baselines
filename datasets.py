import torch.utils.data
import random
import scipy.misc
import numpy as np
import os
import math

import utils


class IntPhys(torch.utils.data.Dataset):
    def __init__(self, opt, split):
        self.opt = opt
        self.index = 0
        self.test = split == "test"
        if opt.list:
            self.file = os.path.join(opt.list, "paths_%s.npy" % split)
            self.paths = np.load(self.file).tolist()
            count = min(opt.count, len(self.paths)) * self.opt.m
        else:
            self.pattern = opt.pattern
            count = opt.count * opt.m
            count = count * 0.9 if split == "train" else count * 0.1
            count = int(count)
            self.i0 = 1 if split == "train" else int(0.9 * opt.count + 1)
        vars(opt)["n_sample_%s" % split] = count
        self.count = count - (count % opt.bsz)
        vars(opt)["nbatch_%s" % split] = int(count / opt.bsz)
        print("n_sample_%s: %s" % (split, self.count))

    def load(self, x, nc, start, seq, interp, c):
        res = []
        for j, f in enumerate(seq):
            v = os.path.join(video_path, c)
            mode = "L" if nc == 1 else "RGB"
            img = scipy.misc.imread(
                "%s/%s/%s_%03d.png" % (v, x, x, start + f), mode=mode
            )
            out = scipy.misc.imresize(
                img, (self.opt.frame_height, self.opt.frame_width), interp
            )
            res.append(out)
        return np.array(res)

    def loadDiff(x, nc, start, seq, interp, c):
        if self.opt.residual == 0:
            return load(x, nc, start, seq, interp, c)
        else:
            out0 = load(x, nc, start, seq, interp, c)
            out1 = load(x, nc, start + self.opt.residual, seq, interp, c)
            return out1 - out0

    def make_output(self, x, start, seq, c="."):
        if x == "depth":
            return loadDiff("depth", 1, start, seq, "bilinear", c)
        elif x == "mask":
            mask_value = utils.get_mask_index(
                os.path.join(video_path, str(c), "status.json"), self.opt.mask_object
            )
            raw_mask = loadDiff("masks", 1, start, seq, "nearest", c)
            mask = raw_mask.astype(int)
            out = [np.ones(mask.shape, dtype=bool)]
            for o in self.opt.mask_object:
                m = np.zeros(mask.shape, dtype=bool)
                for v in mask_value[o]:
                    m[mask == v] = True
                    out[0][mask == v] = False
                out.append(m)
            return np.transpose(np.array(out, dtype=int), (1, 0, 2, 3))
        elif x == "scene":
            out = (
                loadDiff(
                    "scene", self.opt.num_channels, start, seq, "bilinear", c
                ).astype(float)
                / 255
            )
            return np.transpose(out, (0, 3, 1, 2))
        else:
            print("Unknown opt.input or opt.target: " + x)
            return None

    def __getitem__(self, index):
        video_idx = math.floor(index / self.opt.m)
        video_path = self._getpath(video_idx)
        frame_idx = index % self.opt.m
        if self.test:
            input_, target = [], []
            for c in range(1, 5):
                input_.append(
                    make_output(self.opt.input, frame_idx, self.opt.input_seq, str(c))
                )
                target.append(
                    make_output(self.opt.target, frame_idx, self.opt.target_seq, str(c))
                )
            input_ = np.array(input_)
            target = np.array(target)
        else:
            input_ = make_output(self.opt.input, frame_idx, self.opt.input_seq)
            target = make_output(self.opt.target, frame_idx, self.opt.target_seq)
        return input_, target

    def __len__(self):
        return self.count

    def _getpath(self, video_idx):
        if hasattr(self, "paths"):
            try:
                video_path = self.paths[video_idx].decode("UTF-8")
            except AttributeError:
                video_path = self.paths[video_idx]
        else:
            video_path = self.pattern % (self.i0 + video_idx)
        return video_path
