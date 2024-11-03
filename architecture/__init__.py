import torch
import distributed as D
from easydict import EasyDict as edict
from .router import StandardBlockConfig, StandardBlock


def load_model(args):
    from architecture.transformer import GPT, GPT2config, GPT2checkpoints
    from architecture.routed_transformer import RoutedGPT
    from architecture.routed_transformer_analyze import RoutedGPT as AnalyzeRoutedGPT
    from architecture.multi_layer_routed_transformer_cleaned import MultiLayerRoutedGPT

    args = edict(vars(args))

    if args.arch is not None:
        if args.arch == "gpt2":
            args.update(GPT2config.base())
        elif args.arch == "gpt2-medium":
            args.update(GPT2config.medium())
        elif args.arch == "gpt2-large":
            args.update(GPT2config.large())
        elif args.arch == "gpt2-xl":
            args.update(GPT2config.xl())
        else:
            raise ValueError(f"Unknown architecture {args.arch}")

    if args.router_method is None and args.analyze is False:
        net = GPT(
            heads=args.heads,
            width=args.width,
            depth=args.depth,
            block_size=args.context,
            vocab_size=args.vocab_size,
            scale=args.get("scale", True),
            bias=args.get("bias", False),
            mlp=args.get("mlp", "default"),
        )
    if args.router_method is not None and args.analyze is False and args.multi_layer_router is False:
        net = RoutedGPT(
            heads=args.heads,
            width=args.width,
            depth=args.depth,
            block_size=args.context,
            vocab_size=args.vocab_size,
            scale=args.get("scale", True),
            bias=args.get("bias", False),
            mlp=args.get("mlp", "default"),
        )

    if args.analyze and args.multi_layer_router is False:
        net = AnalyzeRoutedGPT(
            heads=args.heads,
            width=args.width,
            depth=args.depth,
            block_size=args.context,
            vocab_size=args.vocab_size,
            scale=args.get("scale", True),
            bias=args.get("bias", False),
            mlp=args.get("mlp", "default"),
        )

    if args.multi_layer_router:
        net = MultiLayerRoutedGPT(
            heads=args.heads,
            width=args.width,
            depth=args.depth,
            block_size=args.context,
            vocab_size=args.vocab_size,
            scale=args.get("scale", True),
            bias=args.get("bias", False),
            mlp=args.get("mlp", "default")
        )

    if args.pretrained:
        if args.arch == "gpt2":
            print(f"loading pretrained model from ... {GPT2checkpoints.base()}")
            net.load_state_dict(torch.load(GPT2checkpoints.base()))
        elif args.arch == "gpt2-medium":
            print(f"loading pretrained model from ... {GPT2checkpoints.medium()}")
            net.load_state_dict(torch.load(GPT2checkpoints.medium()))
        elif args.arch == "gpt2-large":
            print(f"loading pretrained model from ... {GPT2checkpoints.large()}")
            net.load_state_dict(torch.load(GPT2checkpoints.large()))
        elif args.arch == "gpt2-xl":
            print(f"loading pretrained model from ... {GPT2checkpoints.xl()}")
            net.load_state_dict(torch.load(GPT2checkpoints.xl()))
        else:
            raise ValueError(f"Unknown architecture {args.arch}")

    if args.chkpt is not None and args.router_method is None:
        print(f"loading model from ... {args.chkpt}")
        net.load_state_dict(torch.load(args.chkpt)["model"])

    net = net.cuda(D.local_rank())

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[D.local_rank()]
        )

    # print("{: <37} {: <37} {}".format("\nLayer", " Shape", " Initial norm"))
    # print("{: <37} {: <37} {}".format(*["============================="]*3))
    # for name, p in net.named_parameters():
    #     print("{: <37} {: <37} {}".format(name, str(list(p.shape)), p.norm().item()))
    return net
