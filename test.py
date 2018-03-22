import argparse
import torch.utils.data
import random
import time as dt
import numpy as np
from pydoc import locate
import json
import time
import os
import sklearn.metrics

import option
import models
import datasets
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='append',
                    type=lambda kv: kv.split("="), dest='load')
parser.add_argument('--opt_file', type=str)
parser.add_argument('--list_path', type=str, default='lists/3d_test')
parser.add_argument('--load_scores', nargs='+',  default=None)
parser.add_argument('--list_name', nargs='+',  default=['visible_dynamic_1_nObj1', 'visible_dynamic_1_nObj2', 'visible_dynamic_1_nObj3', 'visible_dynamic_2_nObj1', 'visible_dynamic_2_nObj2', 'visible_dynamic_2_nObj3', 'visible_static_nObj1', 'visible_static_nObj2', 'visible_static_nObj3', 'occluded_dynamic_1_nObj1', 'occluded_dynamic_1_nObj2', 'occluded_dynamic_1_nObj3', 'occluded_dynamic_2_nObj1', 'occluded_dynamic_2_nObj2', 'occluded_dynamic_2_nObj3', 'occluded_static_nObj1', 'occluded_static_nObj2', 'occluded_static_nObj3'])
parser.add_argument('--name', default='test')
parser.add_argument('--checkpoint', type=str, default='checkpoints/tests')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--image_save', action='store_true')
parser.add_argument('--image_save_interval', type=int, default=1)
parser.add_argument('--visdom', action='store_true')
parser.add_argument('--visdom_interval', type=int, default=1)
parser.add_argument('--manualSeed', type=int, default=1)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--nThreads', type=int, default=20)
parser.add_argument('--count', type=int, default=25)
parser.add_argument('--mask_object', nargs='+', default=['object', 'occluder'])
opt_test = parser.parse_args()
opt_test.name += '_' + time.strftime('%y%m%d_%H%M%S')
print(opt_test)

random.seed(opt_test.manualSeed)
torch.manual_seed(opt_test.manualSeed)
if opt_test.gpu:
    torch.cuda.manual_seed_all(opt_test.manualSeed)

if opt_test.load:
    with open (opt_test.opt_file, "r") as f:
        data=f.readlines()
    opt = json.loads(data[0])
    opt['p_red'] = 0
    opt['mask_object'] = opt_test.mask_object
    print(opt)
    opt = utils.to_namespace(opt)
    opt.bsz = opt.m
    opt.count = opt_test.count
    model = locate('models.%s' %opt.model)(opt, test=True)
    model.load(opt_test.load)
    if opt_test.gpu:
        model.gpu()
    if opt_test.eval:
        model.eval()
    else:
        print('WARNING: no call to eval()')
    viz = utils.Viz(opt_test)
    viz_output = utils.Viz(opt_test)
else:
    assert opt_test.load_scores

def process_batch(batch, j, t0):
    """Compute score for every frames in a video (which is also a batch).

    batch = [input, target]: all frames in the video
    j: index of the video
    t0: time when the test started
    """

    nbatch = vars(opt)['nbatch_test']
    frame_scores = np.zeros((opt.m, 4))
    d3, d4 = batch[0].size(3), batch[0].size(4)
    for i in range(opt.bsz):
        for c in range(4):
            data = [batch[0][i][c], batch[1][i][c]]
            frame_scores[i][c] = model.score(data)
            if opt_test.image_save or opt_test.visdom:
                to_plot = []
                nviz = 1
                to_plot.append(utils.stack(data[0].unsqueeze(0), nviz, opt.input_len))
                to_plot.append(utils.stack(data[1].unsqueeze(0), nviz, opt.target_len))
                to_plot.append(utils.stack(model.output(), nviz, opt.target_len))
                img = np.concatenate(to_plot, 2)
                viz_output(img, {}, i, j, nbatch, 'output')

    if opt_test.verbose:
        batch_time = (dt.time() - t0) / (j + 1)
        eta = nbatch * batch_time
        out = ' test: batch %.5d/%.5d |' %(j, nbatch - 1)
        mean = frame_scores.mean(0)
        out += ' Mean: %.2e - %.2e - %.2e - %.2e |' \
                %(mean[0], mean[1], mean[2], mean[3])
        out += ' batch time: %.2fs | test eta: %.2dH%.2dm%.2ds' \
            %(batch_time, eta / (60 * 60), (eta / 60) % 60, eta % 60)
        print(out, end='\r')
    if opt_test.image_save or opt_test.visdom:
        for c in range(4):
            to_plot = []
            nviz = opt.m
            to_plot.append(utils.stack(batch[0].select(1, c), nviz, opt.input_len))
            to_plot.append(utils.stack(batch[1].select(1, c), nviz, opt.target_len))
            img = np.concatenate(to_plot, 2)
            viz(img, {'c' : frame_scores}, i, j, nbatch, str(c))
    return frame_scores

def _acc(scores, labels, k=2):
    r = np.random.random(scores.shape)
    # lexsort allows to randomly choose between ties
    idx = np.lexsort((r, scores), axis=scores.ndim-1)
    m = 1
    if idx.ndim > 1:
        for i in range(k):
            m -= idx[:, i].choose(labels.T).mean() / k
    else:
        m -= labels[idx[:k]].mean()
    return m

def absolute_acc(scores, labels):
    """Computes accuracy for absolute classification task."""
    k = scores.shape[0] * 2
    return _acc(scores.flatten(), labels.flatten(), k)

def auc(scores, labels):
    """Computes Area Under the Roc Curve."""
    return sklearn.metrics.roc_auc_score(labels.flatten(), scores.flatten())

def relative_acc(scores, labels):
    """Compute accuracy for relative classification task."""
    mean_pos = (scores * labels).mean(1)
    mean_imp = (scores * np.logical_not(labels)).mean(1)
    return (np.mean(mean_pos > mean_imp) + np.mean(mean_pos >= mean_imp)) / 2

def test(list_name):
    """Performs the test for a given set of videos.

    list_name: path to the subfolder (in opt_test.list_path) containing
        the videos to test
    """
    scores_mean, scores_min = [], []
    if opt_test.load_scores:
        for p in opt_test.load_scores:
            scores_mean.append(np.load(p + '/' + list_name + '_scores_mean.npy'))
            scores_min.append(np.load(p + '/' + list_name + '_scores_min.npy'))
        scores_mean = np.array(scores_mean).mean(0)
        scores_min = np.array(scores_min).mean(0)
    else:
        opt.list = os.path.join(opt_test.list_path, list_name)
        testLoader = torch.utils.data.DataLoader(
            datasets.IntPhys(opt, 'test'),
            opt.bsz,
            num_workers=opt_test.nThreads,
            shuffle=False
        )
        t0 = dt.time()
        for j, batch in enumerate(testLoader, 0):
            frame_scores = process_batch(batch, j, t0)
            scores_mean.append(frame_scores.mean(0))
            scores_min.append(frame_scores.min(0))
        scores_mean = np.array(scores_mean)
        scores_min = np.array(scores_min)
    np.save(os.path.join(cp_path, list_name + '_scores_min'), scores_min)
    np.save(os.path.join(cp_path, list_name + '_scores_mean'), scores_mean)
    labels = np.ones(scores_mean.shape, dtype=bool)
    labels[:,-2:] = False
    res = {}
    dscores = {
        'scores_mean': scores_mean,
        'scores_min': scores_min
    }
    dacc = {
        'absolute_acc': absolute_acc,
        'auc': auc,
        'relative_acc': relative_acc
    }
    for keysc, sc in dscores.items():
       for keyacc, acc in dacc.items():
           res['%s(%s)' %(keyacc, keysc)] =  acc(sc, labels)
    return res

cp_path = os.path.join('checkpoints/tests', opt_test.name)
if not os.path.isdir(cp_path):
    os.mkdir(cp_path)
if opt_test.load:
    with open(os.path.join(cp_path, 'opt.txt'), 'w') as f:
        json.dump(vars(opt), f)
with open(os.path.join(cp_path, 'opt_test.txt'), 'w') as f:
    json.dump(vars(opt_test), f)
results = {}
for list_name in opt_test.list_name:
    res = test(list_name)
    print(list_name)
    print(res)
    results[list_name] = res
with open(os.path.join(cp_path, 'results.txt'), 'w') as f:
    json.dump(results, f)

print('python format_table.py --results checkpoints/tests/%s/results.txt --score \'relative_acc(scores_mean)\'' %opt_test.name)
