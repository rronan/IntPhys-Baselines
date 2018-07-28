import argparse
import os
import numpy as np
import math

from path import Path

def make_list(args):
    video_paths = []
    for datadir in args.datadir:
        for subfolder in sorted(os.listdir(datadir)):
            path = os.path.join(datadir, subfolder)
            if args.pattern in path and os.path.isdir(path):
                print(path)
                video_paths.append(path)
    ntotal = len(video_paths)
    print(ntotal)
    idx = np.random.permutation(ntotal)
    ntrain = math.floor(args.prop_train * ntotal)
    nval = math.floor(args.prop_val * ntotal)
    ntest = math.floor(args.prop_test * ntotal)
    train, val, test = [], [], []
    for i in idx:
        x = video_paths.pop()
        if i < ntrain:
            train.append(x)
        elif i < ntrain + nval:
            val.append(x)
        else:
            test.append(x)
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    np.save(os.path.join(args.outdir, args.train_file), np.array(train))
    np.save(os.path.join(args.outdir, args.val_file), np.array(val))
    np.save(os.path.join(args.outdir, args.test_file), np.array(test))

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
    args = parser.parse_args()
    print(args)

    make_list(args)

