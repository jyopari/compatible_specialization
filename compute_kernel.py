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


def main(args):
    D.init_distributed_mode(args)

    if args.eval_sep:
        train_loader, test_loader, sep_test_loader = load_dataset(args)
    else:
        assert False
        train_loader, test_loader = load_dataset(args)
        sep_test_loader = None

    if args.distributed:
        train_loader.sampler.set_epoch(0)
    args.dtype = misc.auto_determine_dtype()[0]
    args.amp = True if args.dtype == torch.float16 else False

    if args.router_method is not None:
        router_configs = {
            "standard": architecture.StandardBlockConfig,
        }
        assert len(args.router_ckpts) > 1, "Please specify the router checkpoints."
        assert (
            args.router_method in router_configs
        ), f"Unknown router method {args.router_method}"
        router_config = router_configs[args.router_method](
            in_dim=args.width, expert_dim=args.width, top_k=args.topk
        )

        model = load_model(args)
        model.load_weights(args, router_config)
        model = model.to(args.dtype)
        if args.chkpt is not None:
            print("loaded router model from ...", args.chkpt)
            model.load_state_dict(torch.load(args.chkpt)["model"], strict=True)
    else:
        model = load_model(args)

    optim = get_optimizer(model, args)
    args.step = 0

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"num params: \t{num_params}")
    print(f"num tokens: \t{args.train_tokens * args.context}")
    print(f"num steps:  \t{args.train_tokens // args.batch_size}")
    print(f"dtype:      \t{args.dtype}")
    print(f"amp:        \t{args.amp}")
    # model = torch.compile(model)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    logger = None
    if args.wandb and D.is_main_process():
        if args.train_mlp:
            logger = log_utils.wandb_logger("gpt-merge-ft", args)
        else:
            logger = log_utils.wandb_logger("gpt-merge-route", args)

    checkpoints = []

    for i in range(len(args.chkpt_base_name)):
        checkpoints += [
        f"/data/scratch-oc40/pulkitag/jyop/gpt-merge-out/{args.chkpt_base_name[i]}{step}_finetuned.pt" for step in range(0, args.max_step+1, args.step_size)
    ]

    results = {}

    # assert that the sep_test_loader has only one dataset
    if args.eval_sep:
        assert len(sep_test_loader) == 1

    # get only one batch
    data_in, _ = next(iter(sep_test_loader[0]))

    for i, checkpoint in enumerate(checkpoints):
        model.load_state_dict(torch.load(checkpoint)["model"], strict=True)
        model.eval()
        print(f"Loaded model from {checkpoint}")

        # evaluate only
        with torch.no_grad():
            output = model(data_in.cuda())
            #results[checkpoint] = model.hidden_states.mean(dim=1).cpu()
            results[checkpoint] = model.hidden_states_list[6]

    # save the results
    torch.save(results, f"representation_analysis_final/{args.file_name}.pt")


if __name__ == "__main__":
    """
    >>> CUDA_VISIBLE_DEVICES=0 python compute_kernel.py --arch gpt2 --context 1024 --batch_size 128 \
        --datasets gsm-hard --eval_sep gsm-hard \
        --chkpt_base_name gpt2-tinycodes-gpt2-0.0008-1-tinycodes_560000_step- gpt2-orca-math-gpt2-0.0008-1-orca-math-problems_560000_step- \
        --max_step 8500 --step_size 500 --file_name tinycodes-orca-gsm-hard-step-representation-layer-6
    """

    from arguments import parse_args

    args = parse_args()
    main(args)
