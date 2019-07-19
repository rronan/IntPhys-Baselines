import os
from subprocess import call
import argparse
from copy import deepcopy

import make_list_train

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default=".")
parser.add_argument(
    "--blocks",
    nargs="+",
    default=[
        "O1_test_occluded_dynamic_1_nobj1",
        "O1_test_occluded_dynamic_1_nobj2",
        "O1_test_occluded_dynamic_1_nobj3",
        "O1_test_occluded_dynamic_2_nobj1",
        "O1_test_occluded_dynamic_2_nobj2",
        "O1_test_occluded_dynamic_2_nobj3",
        "O1_test_occluded_static_nobj1",
        "O1_test_occluded_static_nobj2",
        "O1_test_occluded_static_nobj3",
        "O1_test_visible_dynamic_1_nobj1",
        "O1_test_visible_dynamic_1_nobj2",
        "O1_test_visible_dynamic_1_nobj3",
        "O1_test_visible_dynamic_2_nobj1",
        "O1_test_visible_dynamic_2_nobj2",
        "O1_test_visible_dynamic_2_nobj3",
        "O1_test_visible_static_nobj1",
        "O1_test_visible_static_nobj2",
        "O1_test_visible_static_nobj3",
        "O2_test_occluded_dynamic_1_nobj1",
        "O2_test_occluded_dynamic_1_nobj2",
        "O2_test_occluded_dynamic_1_nobj3",
        "O2_test_occluded_dynamic_2_nobj1",
        "O2_test_occluded_dynamic_2_nobj2",
        "O2_test_occluded_dynamic_2_nobj3",
        "O2_test_occluded_static_nobj1",
        "O2_test_occluded_static_nobj2",
        "O2_test_occluded_static_nobj3",
        "O2_test_visible_dynamic_1_nobj1",
        "O2_test_visible_dynamic_1_nobj2",
        "O2_test_visible_dynamic_1_nobj3",
        "O2_test_visible_dynamic_2_nobj1",
        "O2_test_visible_dynamic_2_nobj2",
        "O2_test_visible_dynamic_2_nobj3",
        "O2_test_visible_static_nobj1",
        "O2_test_visible_static_nobj2",
        "O2_test_visible_static_nobj3",
        "O3_test_occluded_dynamic_1_nobj1",
        "O3_test_occluded_dynamic_1_nobj2",
        "O3_test_occluded_dynamic_1_nobj3",
        "O3_test_occluded_dynamic_2_nobj1",
        "O3_test_occluded_dynamic_2_nobj2",
        "O3_test_occluded_dynamic_2_nobj3",
        "O3_test_occluded_static_nobj1",
        "O3_test_occluded_static_nobj2",
        "O3_test_occluded_static_nobj3",
        "O3_test_visible_dynamic_1_nobj1",
        "O3_test_visible_dynamic_1_nobj2",
        "O3_test_visible_dynamic_1_nobj3",
        "O3_test_visible_dynamic_2_nobj1",
        "O3_test_visible_dynamic_2_nobj2",
        "O3_test_visible_dynamic_2_nobj3",
        "O3_test_visible_static_nobj1",
        "O3_test_visible_static_nobj2",
        "O3_test_visible_static_nobj3",
    ],
)
parser.add_argument("--prop_train", type=float, default=0)
parser.add_argument("--prop_val", type=float, default=0)
parser.add_argument("--prop_test", type=float, default=1)
parser.add_argument("--train_file", default="paths_train.npy")
parser.add_argument("--val_file", default="paths_val.npy")
parser.add_argument("--test_file", default="paths_test.npy")
parser.add_argument("--datadir", nargs="+", default=[])
args = parser.parse_args()
for b in args.blocks:
    args_c = deepcopy(args)
    args_c.outdir = os.path.join(args.outdir, b)
    args_c.pattern = b
    make_list_train.make_list(args_c)
