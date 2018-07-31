import os
import json
from collections import defaultdict

import numpy as np
from path import Path
from argparse import ArgumentParser
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key, Attr
from sklearn.metrics import roc_auc_score


dynamodb = boto3.resource('dynamodb',
                          aws_access_key_id=os.environ['DDB_AWS_ACCESS_KEY_ID'],
                          aws_secret_access_key=os.environ['DDB_AWS_SECRET_ACCESS_KEY'],
                          region_name='us-east-2')

def parse_item(item):
    image = item['image']
    if image.endswith(('_1.gif', '_2.gif')):
        label = True
    elif image.endswith(('_3.gif', '_4.gif')):
        label = False
    else:
        raise ValueError('Could not parse image %s.' %image)
    pred = int(item['result'])
    return label, pred, image[5:-6]


def summary(d):
    dd = defaultdict(list)
    for k,v in d.items():
        O = k[:2]
        if 'visible' in k:
            vis = 'visible'
        elif 'occluded' in k:
            vis = 'occluded'
        else:
            raise ValueError('Could not parse video: %s' %k)
        dd['%s_%s' %(O, vis)].append(v)
    out = {}
    for k,v in dd.items():
        out[k] = np.mean(v)
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--write_json", action='store_true')
    parser.add_argument("--output", type=Path, default='.')
    args = parser.parse_args()

    mturk_logs = {}

    table = dynamodb.Table('results')
    response = table.scan(
        FilterExpression=(Key('env').eq('prod')
                          & Attr('image').begins_with('test')))
    items = response['Items']
    mturk_logs['n_videos'] = len(items)
    results = map(parse_item, items)

    d_by_quadruplet = defaultdict(list)
    d_by_category = defaultdict(list)
    for label, pred, quadruplet in results:
        d_by_quadruplet[quadruplet].append((label, pred))
        category = quadruplet[5:]
        d_by_category[category].append((label, pred))

    abs_score_acc, abs_score_auc = {}, {}
    for k,v in d_by_category.items():
        labels, preds = map(np.array, zip(*v))
        abs_score_acc[k] = np.mean(labels == (preds > 2.5))
        abs_score_auc[k] = roc_auc_score(labels, preds / 6.)

    dd = defaultdict(list)
    c = 0
    for k,v in d_by_quadruplet.items():
        if len(v) == 4:
            c += 1
            category = quadruplet[5:]
            sum_false, sum_true = 0, 0
            for label, pred in v:
                if label:
                    sum_true += pred
                else:
                    sum_false += pred
            if sum_true > sum_false:
                y = 1
            elif sum_true == sum_false:
                y = 0.5
            else:
                y = 0
            dd[category].append(y)
        elif len(v) not in [1,2,3]:
            raise ValueError('len(v) not in [1,2,3,4]: %d' %len(v))
    mturk_logs['n_full_quadruplets'] = c

    rel_score_acc = {}
    for k,v in dd.items():
        rel_score_acc[k] = np.mean(v)

    summary_rel_score_acc = summary(rel_score_acc)
    summary_abs_score_acc = summary(abs_score_acc)
    summary_abs_score_auc = summary(abs_score_auc)

    if args.verbose:
        print("rel_score_acc", rel_score_acc, sep='\n')
        print("abs_score_acc", abs_score_acc, sep='\n')
        print("abs_score_auc", abs_score_auc, sep='\n')
        print("summary_rel_score_acc", summary_rel_score_acc, sep='\n')
        print("summary_abs_score_acc", summary_abs_score_acc, sep='\n')
        print("summary_abs_score_auc", summary_abs_score_auc, sep='\n')
    print(mturk_logs)

    if args.write_json:
        with open(args.output / 'mturk_rel_acc.json', 'wt') as f:
            json.dump(rel_score_acc, f, indent=4, sort_keys=True)

        with open(args.output / 'mturk_abs_acc.json', 'wt') as f:
            json.dump(abs_score_acc, f, indent=4, sort_keys=True)

        with open(args.output / 'mturk_abs_auc.json', 'wt') as f:
            json.dump(abs_score_auc, f, indent=4, sort_keys=True)

        with open(args.output / 'summary_mturk_rel_acc.json', 'wt') as f:
            json.dump(summary_rel_score_acc, f, indent=4, sort_keys=True)

        with open(args.output / 'summary_mturk_abs_acc.json', 'wt') as f:
            json.dump(summary_abs_score_acc, f, indent=4, sort_keys=True)

        with open(args.output / 'summary_mturk_abs_auc.json', 'wt') as f:
            json.dump(summary_abs_score_auc, f, indent=4, sort_keys=True)

        with open(args.output / 'mturk_logs.json', 'wt') as f:
            json.dump(mturk_logs, f, indent=4, sort_keys=True)

