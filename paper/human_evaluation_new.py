import os
import json
from collections import defaultdict
import time

import numpy as np
from path import Path
from argparse import ArgumentParser
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key, Attr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plot


def full_scan(table_ref):
    retries = 0
    response = table_ref.scan()
    data = response["Items"]
    while response.get("LastEvaluatedKey"):
        try:
            response = table_ref.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            retries = 0
        except Exception:
            print("WHOA, too fast, slow it down retries={}".format(retries))
            time.sleep(2 ** retries)
            retries += 1
            if retries > 5:
                raise Exception("Exceed retries limit of 5")
        data.extend(response["Items"])
    return data


def scan_db():
    dynamodb = boto3.resource(
        "dynamodb",
        aws_access_key_id=os.environ["DDB_AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["DDB_AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-2",
    )
    table = dynamodb.Table("results")
    db = full_scan(table)
    return db


def is_prod(item):
    return item["env"] == "prod" and item["image"].startswith("test")


def category_from_video(video):
    path = video.split(".")[0].replace("_", "/")
    with open(f"/sata/rriochet/intphys2019/metadata/test/{path}/status.json", "r") as f:
        status = json.load(f)
    # __import__("pdb").set_trace()
    category = None
    return category


def parse_item(item):
    image = item["image"]
    category = category_from_video(image)
    if image.endswith(("_1.gif", "_2.gif")):
        label = True
    elif image.endswith(("_3.gif", "_4.gif")):
        label = False
    else:
        raise ValueError("Could not parse image %s." % image)
    pred = int(item["result"])
    turker = item["turker"]
    return label, pred, category, image[:-4], turker


def inclusion_test(results):
    d, out = defaultdict(list), set()
    for label, pred, category, image, turker in results:
        if "visible" in image and "nobj1" in image and image[:2] == "O1":
            d[turker].append(label == (pred > 3.5))
    for k, v in d.items():
        if np.mean(v) == 0:
            out.add(k)
    return d.keys() - out, out


def apply_inclusion_test(results, mturk_logs):
    results = list(results)
    mturk_logs["n_videos_before_inclusion"] = len(results)
    in_, out_ = inclusion_test(results)
    mturk_logs["n_excluded_turkers"] = len(out_)
    mturk_logs["n_included_turkers"] = len(in_)
    results = filter(lambda x: x[-1] in in_, results)
    return results, mturk_logs


def turker_score(results):
    d = {"O1": defaultdict(list), "O2": defaultdict(list), "O3": defaultdict(list)}
    for label, pred, category, image, turker in results:
        if "visible" in image and "nobj1" in image:
            O = image[:2]
            d[O][turker].append(label == (pred > 3.5))
    out = defaultdict(list)
    for O, v in d.items():
        for _, results in v.items():
            out[O].append(np.mean(results))
    return out


def plot_all_answers(results):
    results = list(results)
    for k, v in turker_score(results).items():
        plot.hist(v, bins=100)
        plot.title(k)
        plot.show()
    all_a = all_answers(results)
    plot.hist(all_a, bins=100)
    plot.title("all_answers")
    plot.show()


def make_d_by(results, mturk_logs):
    d_by_quadruplet = defaultdict(list)
    d_by_category = defaultdict(list)
    b = 0
    for label, pred, category, quadruplet, _ in results:
        d_by_quadruplet[quadruplet].append((label, pred))
        video = quadruplet[:7]
        d_by_category[category].append((label, pred))
        b += 1
    mturk_logs["n_videos"] = b
    return d_by_quadruplet, d_by_category, mturk_logs


def make_abs_acc_auc(d_by_category):
    abs_score_acc, abs_score_auc = {}, {}
    for k, v in d_by_category.items():
        labels, preds = map(np.array, zip(*v))
        abs_score_acc[k] = np.mean(labels == (preds > 3.5))
        abs_score_auc[k] = roc_auc_score(labels, preds / 6.0)
    return abs_score_acc, abs_score_auc


def make_rel_acc(d_by_category, mturk_logs):
    d = defaultdict(list)
    mturk_logs["n_full_quadruplets"] = 0
    mturk_logs["n_full+_quadruplets"] = 0
    for k, v in d_by_quadruplet.items():
        if len(v) >= 4:
            mturk_logs["n_full_quadruplets"] += 1
            if len(v) > 4:
                mturk_logs["n_full+_quadruplets"] += 1
            category = k[5:]
            mean_true = np.mean([x[1] for x in filter(lambda x: x[0], v)])
            mean_false = np.mean([x[1] for x in filter(lambda x: not x[0], v)])
            if mean_true > mean_false:
                y = 1
            elif mean_true == mean_false:
                y = 0.5
            else:
                y = 0
            d[category].append(y)
    rel_score_acc = {}
    for k, v in d.items():
        rel_score_acc[k] = np.mean(v)
    return rel_score_acc, mturk_logs


def all_answers(results):
    out = []
    for _, pred, *_ in results:
        out.append(pred)
    return out


def summary(d):
    dd = defaultdict(list)
    for k, v in d.items():
        O = k[:2]
        if "visible" in k:
            vis = "visible"
        elif "occluded" in k:
            vis = "occluded"
        else:
            raise ValueError("Could not parse video: %s" % k)
        big_category = "%s_%s" % (O, vis)
        dd[big_category].append(v)
    out = {}
    for k, v in dd.items():
        out[k] = 1 - np.mean(v)
    return out


def make_turker_answers(results):
    label_list = []
    answer_list = []
    name_list = []
    turker_list = []
    for label, answer, category, name, turker in results:
        label_list.append(label)
        answer_list.append(answer)
        name_list.append(name)
        turker_list.append(turker)
    out = {
        "labels": label_list,
        "answers": answer_list,
        "names": name_list,
        "turkers": turker_list,
    }
    return out


def output(processed_results, outdir, write_json, verbose):
    for key, value in processed_results.items():
        if verbose:
            print(key, value, sep="\n")
        if write_json:
            with open(outdir / key + ".json", "wt") as f:
                json.dump(value, f, indent=4, sort_keys=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--write_json", action="store_true")
    parser.add_argument("--save_turker_answers", action="store_true")
    parser.add_argument("--inclusion_test", action="store_true")
    parser.add_argument("--outdir", type=Path, default=".")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    mturk_logs = {}
    db = scan_db()
    results = list(map(parse_item, db))
    if args.inclusion_test:
        results, mturk_logs = apply_inclusion_test(results, mturk_logs)
    if args.plot:
        plot_all_answers(results)
    d_by_quadruplet, d_by_category, mturk_logs = make_d_by(results, mturk_logs)
    __import__("pdb").set_trace()
    abs_score_acc, abs_score_auc = make_abs_acc_auc(d_by_category)
    rel_score_acc, mturk_logs = make_rel_acc(d_by_category, mturk_logs)
    processed_results = {
        "rel_score_acc": rel_score_acc,
        "abs_score_acc": abs_score_acc,
        "abs_score_auc": abs_score_auc,
        "summary_rel_score_acc": summary(rel_score_acc),
        "summary_abs_score_acc": summary(abs_score_acc),
        "summary_abs_score_auc": summary(abs_score_auc),
    }
    output(processed_results, args.outdir, args.write_json, args.verbose)
    with open(args.outdir / "mturk_logs.json", "wt") as f:
        json.dump(mturk_logs, f, indent=4, sort_keys=True)

    if args.save_turker_answers:
        with open(args.outdir / "full_results.json", "wt") as f:
            json.dump(make_turker_answers(results), f, indent=4, sort_keys=True)
