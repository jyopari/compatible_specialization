import math
import merge_methods.slerp
import torch
import numpy as np

from tqdm import tqdm

from data import load_dataset
from architecture import load_model
from optim import get_optimizer
import log_utils
import distributed as D
import time
import misc

from collections import deque
from tqdm import tqdm
from copy import deepcopy

from merge_methods.slerp import slerp
from merge_methods.lerp import lerp

import os
from arguments import parse_args

args = parse_args()
args.arch = "gpt2"
args.context = 1024
args.name = "gpt2_base"


def coefs(n):
    l = [0 for i in range(n - 1)]
    l[n - 2] = (n - 1) / n
    i = n - 3
    prod = 1
    while i >= 0:
        prod *= l[i + 1]
        l[i] = 1 - (1 / n) * (1 / prod)
        i -= 1
    return l


def merge(t, path_1, path_2, name, merge_type="slerp"):

    model_1 = load_model(args)
    model_1.load_state_dict(torch.load(path_1)["model"])

    model_2 = load_model(args)
    model_2.load_state_dict(torch.load(path_2)["model"])
    
    model_1_dict = model_1.state_dict()
    model_2_dict = model_2.state_dict()

    # create a copy of the model_1_dict to interpolate
    model_merge_dict = deepcopy(model_1_dict)

    # interpolate the weights
    for key in tqdm(model_1_dict):
        if merge_type == "lerp":
            model_merge_dict[key] = lerp(t, model_1_dict[key], model_2_dict[key])
        elif merge_type == "slerp":
            model_merge_dict[key] = slerp(t, model_1_dict[key], model_2_dict[key])
        else:
            raise ValueError("Invalid type of interpolation")

    # save the new model
    merged_model = deepcopy(model_1)
    merged_model.load_state_dict(model_merge_dict)

    torch.save(
        {
            "model": merged_model.state_dict(),
        },
        name,
    )


def multi_merge(n, paths, name):
    l = coefs(n)
    print(l)
    model_curr = load_model(args)
    model_curr.load_state_dict(torch.load(paths[0])["model"])
    for i in range(1, n):
        model2 = load_model(args)
        chkpt2 = torch.load(paths[i])
        model2.load_state_dict(chkpt2["model"])
        model_curr = merge((1 - l[i - 1]), model_curr, model2)

    torch.save(
        {
            "model": model_curr.state_dict(),
        },
        name,
    )


# paths = [
#     "/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt",
#     "/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-orca-math-gpt2-0.0008-1-orca-math-problems_finetuned.pt"
# ]

paths = [
    "/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-code_instructions-gpt2-0.0008-1-code_instructions_finetuned.pt",
    "/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/gpt2-tinycodes-gpt2-0.0008-1-tinycodes_finetuned.pt"
]

merge_type = "lerp"

#multi_merge(len(paths), paths, "out/merged_model_tinycodes_miniproofpile_slerp.pt")

for i in range(11):
    t = i / 10
    merge(t, paths[0], paths[1], f"interpolation_results_final/interpolated_models/{merge_type}:{t}_[{os.path.basename(paths[0])}][{os.path.basename(paths[1])}].pt", merge_type=merge_type)