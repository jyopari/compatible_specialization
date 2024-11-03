"""
Interpolates between two models and computes the loss, and then launches ipdb
"""

import math
import torch
import numpy as np

from tqdm import tqdm

from data import load_dataset
import architecture
from architecture import load_model
from optim import get_optimizer
import log_utils
import distributed as D
import time
import misc

from collections import deque

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def cosine_lr_decay(curr_step, max_step, min_lr_factor):
    current_progress = curr_step / max_step
    gain = (
        min_lr_factor
        + (1 - min_lr_factor) * (1 + math.cos(math.pi * current_progress)) / 2
    )
    return gain


@torch.no_grad()
def evaluate(model, dataloader, eval_iter, args):

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    losses = []
    router_stats = []
    model = model.eval()

    for i, (data, target) in (
        pbar := tqdm(
            enumerate(dataloader), total=eval_iter, disable=not D.is_main_process()
        )
    ):
        data, target = data.cuda(D.local_rank()), target.cuda(D.local_rank())

        if i == eval_iter:
            break

        with torch.autocast(device_type="cuda", dtype=args.dtype, enabled=args.amp):
            output = model(data)

            output = output.reshape(-1, output.size(-1))
            target = target.reshape(-1)
            error = criterion(output, target.to(output.device))
            if args.router_method is not None:
                _, stats = model.get_router_loss(error)
                router_stats.append(stats)
            losses.append(error)

        if math.isnan(error.mean().item()):
            raise Exception("Found loss to be nan. Raising exception and exiting.")

    loss = torch.cat(losses).mean()

    # average dictionaries in router_stats
    if args.router_method is not None:
        router_stats = {
            k: np.mean([x[k] for x in router_stats]) for k in router_stats[0]
        }

    if args.distributed:
        loss = D.reduce(loss, reduction="mean")
    model.train()
    return loss, router_stats


def main(args):
    D.init_distributed_mode(args)

    assert len(args.chkpt_base_name) == 2

    checkpoints_1 = [
        f"/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/{args.chkpt_base_name[0]}{step}_finetuned.pt" for step in range(0, args.max_step+1, args.step_size)
    ]

    checkpoints_2 = [
        f"/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/{args.chkpt_base_name[1]}{step}_finetuned.pt" for step in range(0, args.max_step+1, args.step_size)
    ]

    if args.eval_sep:
        train_loader, test_loader, sep_test_loader = load_dataset(args)
    else:
        train_loader, test_loader = load_dataset(args)
        sep_test_loader = None

    if args.distributed:
        train_loader.sampler.set_epoch(0)
    args.dtype = misc.auto_determine_dtype()[0]
    args.amp = True if args.dtype == torch.float16 else False

    final_results = {}
    for c_i in range(len(checkpoints_1)):
        
        args.router_ckpts = [checkpoints_1[c_i], checkpoints_2[c_i]]

        if args.analyze:
            model = load_model(args)
            model.load_weights(args)
            model = model.to(args.dtype)
        else:
            assert False

        results = {}
        #interpolation_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        interpolation_values = [0.4,0.5,0.6] # for first 100 exp
        for interpolate in interpolation_values:
            model.weights.weights = torch.ones(
                (len(model.transformer.h), len(args.router_ckpts))
            )
            model.weights.weights[:, 0] *= interpolate
            model.weights.weights[:, 1] *= 1 - interpolate
            print("interpolate value: ", interpolate)
            with torch.no_grad():
                if args.eval_sep:
                    for i, sep_loader in enumerate(sep_test_loader):
                        test_loss, _ = evaluate(
                            model,
                            sep_loader,
                            eval_iter=args.eval_iter * args.grad_accum,
                            args=args,
                        )
                        print(f"Test loss {args.eval_sep[i]}: \t", test_loss.item())
                        results[interpolate, args.eval_sep[i]] = test_loss.item()

                else:
                    assert False # for now viz notebook not formated for this 

        final_results[c_i] = results

    # loop through all models and make the file name
    import os
    import pickle

    with open(f"interpolation_results_final/{args.file_name}.pkl", "wb") as f:
        pickle.dump(final_results, f)

   
    return


if __name__ == "__main__":
    """
    # interpolation analysis - here we interpolate between two models using a hardcoded router 
    >>> CUDA_VISIBLE_DEVICES=1 python analyze_batch.py --arch gpt2 --pretrained --batch_size 32 --context 1024 \
        --dataset gsm-hard --eval_sep gsm-hard \
        --chkpt_base_name gpt2-tinycodes-gpt2-0.0008-1-tinycodes_560000_step- gpt2-orca-math-gpt2-0.0008-1-orca-math-problems_560000_step- \
        --max_step 100 --step_size 10 --file_name tinycodes-orca-gsm-hard-step-interpolation-first-100 --analyze 
    """

    from arguments import parse_args

    args = parse_args()
    main(args)
