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
                _, stats = model.get_router_loss(error, args)
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
    return loss, router_stats


def train(
    model,
    optim,
    scaler,
    train_loader,
    test_loader,
    sep_test_loader,
    logger,
    args,
):
    grad_accum = 0
    t_misc = time.time()

    runtimes = {k: deque(maxlen=10) for k in ["model", "misc", "util"]}
    gain = 1.0
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    best_test_loss = float("inf")

    for i, (data, target) in (
        pbar := tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            disable=not D.is_main_process(),
        )
    ):

        if grad_accum == 0:
            # update learning rate only decay after stepping
            gain = cosine_lr_decay(i, len(train_loader), min_lr_factor=0.1)
            for g in optim.param_groups:
                g["lr"] = args.lr * gain

        if args.step % args.log_freq == 0 and grad_accum == 0 and D.is_main_process():
            test_loss, _ = evaluate(
                model,
                test_loader,
                eval_iter=max((args.log_freq // 10) * args.grad_accum, 1),
                args=args,
            )
            print("Test loss: \t", test_loss.item())

            checkpoint = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "args": args,
                "step": args.step,
            }

            if test_loss.item() < best_test_loss:
                print("Saving checkpoint ...")
                if args.router_method is not None:
                    torch.save(
                        checkpoint,
                        f"{args.out_dir}/{args.name}-{args.arch}-{args.lr}-{args.topk}-{args.single_router}-{'-'.join(args.datasets)}_routed.pt",
                    )
                else:
                    torch.save(
                        checkpoint,
                        f"{args.out_dir}/{args.name}-{args.arch}-{args.lr}-{args.topk}-{'-'.join(args.datasets)}_{args.train_tokens}_finetuned.pt",
                    )
                best_test_loss = test_loss.item()

            if args.router_method is None:  # only for non routed models (finetuning)
                # torch.save(
                #     checkpoint,
                #     f"{args.out_dir}/{args.name}-{args.arch}-{args.lr}-{args.topk}-{'-'.join(args.datasets)}_{args.train_tokens}_step-{args.step}_finetuned.pt",
                # )
                pass

            if args.router_method is not None:
                torch.save(
                    checkpoint,
                    f"{args.out_dir}/{args.name}-{args.arch}-{args.lr}-{args.topk}-{args.single_router}-{'-'.join(args.datasets)}_routed_latest.pt",
                )

            if logger is not None and D.is_main_process():
                if args.eval_sep:
                    for i, sep_loader in enumerate(sep_test_loader):
                        test_loss_i, router_stats_i = evaluate(
                            model,
                            sep_loader,
                            eval_iter=args.eval_iter * args.grad_accum,
                            args=args,
                        )
                        if args.router_method is not None:
                            router_stats_i = {
                                f"{args.eval_sep[i]}/{k}": v
                                for k, v in router_stats_i.items()
                            }
                            logger.log(
                                {"step": args.step, **router_stats_i}, train=False
                            )

                        print(f"Test loss {args.eval_sep[i]}: \t", test_loss_i.item())
                        logger.log(
                            {
                                "step": args.step,
                                f"{args.eval_sep[i]}_test_loss": test_loss_i,
                            },
                            train=False,
                        )
                else:
                    logger.log({"step": args.step, "test_loss": test_loss}, train=False)

        # load datset
        data, target = data.cuda(D.local_rank()), target.cuda(D.local_rank())

        t_st = time.time()
        with torch.autocast(device_type="cuda", dtype=args.dtype, enabled=args.amp):
            output = model(data)

            output = output.reshape(-1, output.size(-1))
            target = target.reshape(-1)

            error = criterion(output, target.to(output.device))
            loss = error.mean()  # for non routed models
            if args.router_method is not None:
                loss, _ = model.get_router_loss(error, args)
            error = error.mean()

        # check if distributed mode
        if args.distributed:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

        grad_accum += 1

        if args.distributed:
            loss = D.reduce(loss, reduction="mean")

        if math.isnan(loss):
            raise Exception("Found loss to be nan. Raising exception and exiting.")

        if grad_accum % args.grad_accum == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            model.zero_grad()
            args.step += 1
            grad_accum = 0

        t_en = time.time()

        if (
            logger is not None
            and args.step % 1 == 0
            and grad_accum == 0
            and D.is_main_process()
        ):
            logger.increment(data.shape[0] * D.world_size())
            logger.log(
                {"loss": error.item(), "epoch": args.step, "gain": gain}, train=True
            )
            logger.step()

        runtimes["model"].append(t_en - t_st)  # model forward and backward
        runtimes["misc"].append(t_st - t_misc)  # data loading, logging, etc.
        runtimes["util"].append((t_en - t_st) / (t_en - t_misc) * 100)

        pbar.set_description(
            f"[train] loss: {error.item():.3f} gain: {gain:.3f} "
            + f"time-util: {np.mean(runtimes['util']):.1f}% "
            + f"(model: {np.mean(runtimes['model']):.2f}s | misc: {np.mean(runtimes['misc']):.2f}s)"
        )
        t_misc = time.time()
    return


def main(args):
    D.init_distributed_mode(args)

    if args.eval_sep:
        train_loader, test_loader, sep_test_loader = load_dataset(args)
    else:
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
            logger = log_utils.wandb_logger("gpt-merge-ft-final", args)
        else:
            logger = log_utils.wandb_logger("gpt-merge-route-final", args)

    # evaluate only
    if args.eval:
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

            else:
                test_loss, _ = evaluate(
                    model,
                    test_loader,
                    eval_iter=args.eval_iter * args.grad_accum,
                    args=args,
                )
                print("Test loss: \t", test_loss.item())
        return

    # train
    train(
        model,
        optim,
        scaler,
        train_loader,
        test_loader,
        sep_test_loader,
        logger,
        args,
    )
    print("finished training ...")

    # final test loss
    with torch.no_grad():
        test_loss, _ = evaluate(
            model, test_loader, eval_iter=args.eval_iter * args.grad_accum, args=args
        )

    print("Test loss: \t", test_loss.item())

    if D.is_main_process() and logger is not None:
        logger.log({"step": args.step, "test_loss": test_loss}, train=False)
    return


if __name__ == "__main__":

    from arguments import parse_args

    args = parse_args()
    main(args)
