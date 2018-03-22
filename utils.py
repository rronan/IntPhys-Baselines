import json
import random
import os
import torch
import visdom
import time
import sys
import numpy as np
import scipy.misc
import math

def get_mask_index(filename, mask_object):
    """Get the index of an object from a metadata file

    filename: name of the metadata file
    mask_object: name of the object to get the mask index of
    """
    with open(filename) as file:
        masks_grayscale = json.load(file)["masks_grayscale"]
    out = {}
    for mo in mask_object:
        out[mo] = []
    if type(masks_grayscale) is list:
        for m in masks_grayscale:
            id, o = m[0], m[1]
            for mo in mask_object:
                if mo in o:
                    out[mo].append(id)
    else:
        print('wrong format to masks_grayscale')
    return out

def checkpoint(epoch, model, log, opt):
    """Saves a checkpoint

    epoch: epoch
    model: model
    log: a list gathering statistics about optimization through time
    opt: the opt variable
    """
    if not os.path.isdir(opt.checkpoint):
        os.mkdir(opt.checkpoint)
    with open(os.path.join(opt.checkpoint, 'opt.txt'), 'w') as f:
        json.dump(vars(opt), f)
    with open(os.path.join(opt.checkpoint, 'log.txt'), 'w') as f:
        json.dump(log, f)
    model.save(opt.checkpoint, epoch)

def stack(input_, n, seq):
    """Creates a patch of images for visualization

    input_: the tensor of images
    n: number of sample to include
    seq: sequence length of the input
    """
    cat1 = []
    for i in range(n):
        cat2 = []
        for j in range(seq):
            cat2.append(input_[i][j])
        cat1.append(torch.cat(cat2, 2))
    return torch.cat(cat1, 1).cpu().numpy()

def Viz(opt):
    """Visualization"""
    if opt.visdom:
        vis = visdom.Visdom()
        visdom_id = 'id_' + str(time.time())
        time.sleep(1e-3)
    if not os.path.isdir(opt.checkpoint):
        os.mkdir(opt.checkpoint)
    def inner(img, curve, epoch, batch_idx, nbatch, set_):
        if opt.image_save and batch_idx % opt.image_save_interval == 0:
            scipy.misc.imsave(
                '%s/img_%04d_%s_%06d.png' \
                    %(opt.checkpoint, epoch, set_, batch_idx),
                np.transpose(img, (1, 2, 0))
            )
        if opt.visdom and batch_idx % opt.visdom_interval == 0:
            options = {'title': 'Epoch %02d - %s - Batch %06d/%06d' %(epoch, set_, batch_idx, nbatch)}
            vis.image(
                img = img,
                win = visdom_id + set_,
                env = opt.name
            )
            for i,c in curve.items():
                if len(c) > 1:
                    vis.line(
                        Y=np.array(c),
                        X=np.arange(len(c)),
                        win=visdom_id + str(i),
                        env=opt.name,
                        opts=dict(title=i),
                    )
    return inner

def to_number(d):
    """Convert values of dict d to numbers"""
    out = {}
    for key, value in d.items():
        try:
            out[key] = value.data[0]
        except AttributeError:
            try:
                out[key] = value[0]
            except TypeError:
                out[key] = value
    return out

def filter(d, patterns):
    """Returns a copy of d, containing only keys with given patterns

    d: dict
    patterns: list of patterns
    """
    out = {}
    for p in patterns:
        for name, params in d.items():
            if name.split('.')[0] == p:
                out[name] = params
    return out

class to_namespace:
    def __init__(self, d):
        """Constructs a namespace from a dict

        d: dict
        """
        vars(self).update(dict([(key, value) for key, value in d.items()]))

def slice_epoch(l, n):
    """Creates n slices of range(l), with (approx.) same length"""
    slice_size = math.ceil(l / n)
    result = []
    for i in range(n):
        result.append([])
        for j in range(slice_size):
            if i * slice_size + j >= l:
                return result
            else:
                result[i].append(i * slice_size + j)
    return result

def disableBNRunningMeanStd(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        setattr(m, '_saved_momentum', m.momentum)
        setattr(m, 'momentum', 0)


def enableBNRunningMeanStd(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        setattr(m, 'momentum', m._saved_momentum)
        delattr(m, '_saved_momentum')
