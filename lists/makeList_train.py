import argparse
import os
import numpy as np
import math

def makeList(opt):
    video_paths = []
    for datadir in opt.datadir:
        for subfolder in sorted(os.listdir(datadir)):
            path = os.path.join(datadir, subfolder)
            if opt.pattern in path and os.path.isdir(path):
                print(path)
                video_paths.append(path)
    ntotal = len(video_paths)
    print(ntotal)
    idx = np.random.permutation(ntotal)
    ntrain = math.floor(opt.prop_train * ntotal)
    nval = math.floor(opt.prop_val * ntotal)
    ntest = math.floor(opt.prop_test * ntotal)
    train, val, test = [], [], []
    for i in idx:
        x = video_paths.pop()
        if i < ntrain:
            train.append(x)
        elif i < ntrain + nval:
            val.append(x)
        else:
            test.append(x)
    if not os.path.isdir(opt.outdir):
        os.mkdir(opt.outdir)
    np.save(os.path.join(opt.outdir, opt.train_file), np.array(train))
    np.save(os.path.join(opt.outdir, opt.val_file), np.array(val))
    np.save(os.path.join(opt.outdir, opt.test_file), np.array(test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='.')
    parser.add_argument('--prop_train', type=float, default=0.9)
    parser.add_argument('--prop_val', type=float, default=0.1)
    parser.add_argument('--prop_test', type=float, default=0.0)
    parser.add_argument('--train_file', default='paths_train.npy')
    parser.add_argument('--val_file', default='paths_val.npy')
    parser.add_argument('--test_file', default='paths_test.npy')
    parser.add_argument('--datadir', nargs='+', default=[])
    parser.add_argument('--pattern', default='')
    opt = parser.parse_args()
    print(opt)

    makeList(opt)

