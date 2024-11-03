import torch
import torch.nn as nn
from types import SimpleNamespace
from easydict import EasyDict as edict

import distributed as D

from .library import Library
from .router import StandardBlock
from .transformer import GPT, GPT2config


class WeightedMultiMLP(nn.Module):
    def __init__(self, mlps, weights, layer_id):
        super().__init__()
        self.mlps = mlps
        self.weights = weights
        self.layer_id = layer_id

    def forward(self, x):
        out = 0
        for i, k in enumerate(self.mlps):
            out += self.weights.weights[self.layer_id][i].item() * self.mlps[k](x)
        return out


class RoutedGPT(GPT):
    def __init__(
        self,
        heads,
        width,
        depth,
        block_size,
        vocab_size,
        mlp="default",
        tie_weights=False,
        scale=False,
        bias=False,
    ):
        super().__init__(
            heads, width, depth, block_size, vocab_size, mlp, tie_weights, scale, bias
        )
        self.library = Library()
        self.router_type = None
        return

    def _add_experts(self, args):
        for i, layer in enumerate(self.transformer.h):
            self.library.add_model(f"model_0_mlp_{i}", layer.mlp)

        data = {
            "weights": torch.ones((len(self.transformer.h), len(args.router_ckpts)))
            * (1 / len(args.router_ckpts))
        }
        weights = SimpleNamespace(**data)
        self.weights = weights

        args = edict(vars(args))
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

        # populate library with additional models' mlps
        for route_id, model_name in enumerate(args.router_ckpts[1:]):
            model = GPT(
                heads=args.heads,
                width=args.width,
                depth=args.depth,
                block_size=args.context,
                vocab_size=args.vocab_size,
                scale=args.scale,
                bias=args.bias,
                mlp=args.mlp,
            )
            model.load_state_dict(torch.load(model_name)["model"])

            for i, layer in enumerate(model.transformer.h):
                self.library.add_model(f"model_{route_id+1}_mlp_{i}", layer.mlp)
            del model

        for i, layer in enumerate(self.transformer.h):
            experts = nn.ModuleDict(
                {
                    f"mlp_{j}": self.library.get_model(f"model_{j}_mlp_{i}")
                    for j in range(len(args.router_ckpts))
                }
            )
            device = torch.device(f"cuda:{D.local_rank()}")
            self.transformer.h[i].mlp = WeightedMultiMLP(experts, weights, i).to(
                device=device, dtype=args.dtype
            )
        print("models in the library", self.library.list_models())

    def load_weights(self, args):
        ckpts = args.router_ckpts
        state_dict = torch.load(ckpts[0])
        self.load_state_dict(state_dict["model"])
        self._add_experts(args)
