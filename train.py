import argparse
import random
import time
from pydoc import locate

import numpy as np
import torch.utils.data

import datasets
import option
import utils

opt = option.make(argparse.ArgumentParser())
trainLoader = torch.utils.data.DataLoader(
    datasets.IntPhys(opt, "train"), opt.bsz, num_workers=opt.nThreads, shuffle=True
)
trainLoader.count = 0
valLoader = torch.utils.data.DataLoader(
    datasets.IntPhys(opt, "val"), opt.bsz, num_workers=opt.nThreads, shuffle=True
)
print(opt)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.gpu:
    torch.cuda.manual_seed_all(opt.manualSeed)

model = locate("models.%s" % opt.model)(opt)
if opt.load:
    model.load(opt.load)
if opt.gpu:
    model.gpu()

print("n parameters: %d" % sum([m.numel() for m in model.parameters()]))

viz = utils.Viz(opt)


def process_batch(batch, loss, i, k, set_, t0):
    """Optimization step.

    batch = [input, target]: contains data for optim step [input, target]
    loss: dict containing statistics about optimization
    i: epoch
    k: index of the current batch
    set_: type of batch (\"train\" or \"dev\")
    t0: time of the beginning of epoch
    """

    nbatch = vars(opt)["nbatch_" + set_]
    res = model.step(batch, set_)
    for key, value in res.items():
        try:
            loss[key].append(value)
        except KeyError:
            loss[key] = [value]
    if opt.verbose:
        batch_time = (time.time() - t0) / (k + 1)
        eta = nbatch * batch_time
        out = " %s %d: batch %.5d/%.5d |" % (set_, i, k, nbatch - 1)
        for key, value in res.items():
            out += " %s: %.2e |" % (key, value)
        out += " batch time: %.2fs | %s eta: %.2dH%.2dm%.2ds" % (
            batch_time,
            set_,
            eta / (60 * 60),
            (eta / 60) % 60,
            eta % 60,
        )
        print(out, end="\r")
    if opt.image_save or opt.visdom:
        to_plot = []
        nviz = min(10, opt.bsz)
        to_plot.append(utils.stack(batch[0], nviz, opt.input_len))
        to_plot.append(utils.stack(batch[1], nviz, opt.target_len))
        to_plot.append(utils.stack(model.output(), nviz, opt.target_len))
        img = np.concatenate(to_plot, 2)
        viz(img, loss, i, k, nbatch, set_)
    return loss


loss_train, loss_val, log = {}, {}, []

try:
    for i in range(opt.n_epochs):
        log.append([])
        t_optim = 0
        t0 = time.time()
        train_slices = utils.slice_epoch(opt.nbatch_train, opt.n_slices)
        val_slices = utils.slice_epoch(opt.nbatch_val, opt.n_slices)
        for ts, vs, j in zip(train_slices, val_slices, range(opt.n_slices)):
            log[i].append({})
            for k, batch in zip(ts, trainLoader):
                t = time.time()
                loss_train = process_batch(batch, loss_train, i, k, "train", t0)
                t_optim += time.time() - t
            for key, value in loss_train.items():
                log[i][j]["train_" + key] = np.mean(value[-opt.nbatch_train :])
            for k, batch in zip(vs, valLoader):
                loss_val = process_batch(batch, loss_val, i, k, "val", t0)
                t_optim += time.time() - t
            for key, value in loss_val.items():
                log[i][j]["val_" + key] = np.mean(value[-opt.nbatch_val :])
            utils.checkpoint("%d_%d" % (i, j), model, log, opt)
            log[i][j]["time(optim)"] = "%.2f(%.2f)" % (time.time() - t0, t_optim)
            print(log[i][j])

except KeyboardInterrupt:
    time.sleep(2)  # waiting for all threads to stop
    print("-" * 89)
    save = input("Exiting early, save the last model?[y/n]")
    if save == "y":
        print("Saving...")
        utils.checkpoint("final", model, log, opt)
