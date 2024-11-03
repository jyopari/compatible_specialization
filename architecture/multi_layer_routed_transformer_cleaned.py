import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

import distributed as D

from .library import Library
from .router import StandardBlock
from .transformer import GPT, GPT2config


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if key in ["self", "__class__"]:
                continue
            setattr(self, key, value)


class GPT2checkpoints:
    @staticmethod
    def base():
        # source huggingface converter
        return "./pretrained/gpt2/gpt2.pt"

    def medium():
        return "./pretrained/gpt2/gpt2-medium.pt"

    def large():
        return "./pretrained/gpt2/gpt2-large.pt"

    def xl():
        return "./pretrained/gpt2/gpt2-xl.pt"


class LayerNorm(nn.Module):
    def __init__(self, ndim, scale=True, bias=False):
        super().__init__()
        self.ndim = (ndim,)
        self.weight = nn.Parameter(torch.ones(ndim)) if scale else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.ndim, self.weight, self.bias, 1e-5)


class GatedMLP(nn.Module):
    def __init__(self, width, input_dim, output_dim, act_fn=nn.GELU, bias=False):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, width, bias=bias)
        self.out_proj = nn.Linear(width, output_dim, bias=bias)
        self.gate_proj = nn.Linear(input_dim, width, bias=bias)
        self.act = act_fn()

    def forward(self, x):
        x = self.out_proj(self.act(self.in_proj(x)) * self.gate_proj(x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, width, input_dim, output_dim, bias=False):
        super().__init__()
        self.width = width
        self.in_proj = nn.Linear(input_dim, width, bias=bias)
        self.out_proj = nn.Linear(width, output_dim, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.out_proj(self.act(self.in_proj(x)))


class MultiHeadSelfAttention(nn.Module):
    """MultiHead Attention using PyTorch's scaled_dot_product_attention"""

    def __init__(self, width, heads, causal=False, bias=False, split_qkv=False):
        super().__init__()
        self.causal = causal
        self.heads = heads
        self.split_qkv = split_qkv

        if self.split_qkv:
            self.q = nn.Linear(width, width, bias=bias)
            self.k = nn.Linear(width, width, bias=bias)
            self.v = nn.Linear(width, width, bias=bias)
        else:
            self.in_proj = nn.Linear(width, width * 3, bias=bias)

        self.out_proj = nn.Linear(width, width, bias=bias)
        self.init_weights()
        return

    def init_weights(self):
        """
        Using same initialization protocol for PyTorch's MultiheadAttention
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L1041
        """
        if self.split_qkv:
            torch.nn.init.xavier_uniform_(self.q.weight)
            torch.nn.init.xavier_uniform_(self.k.weight)
            torch.nn.init.xavier_uniform_(self.v.weight)
            if self.q.bias is not None:
                torch.nn.init.constant_(self.q.bias, 0.0)
                torch.nn.init.constant_(self.k.bias, 0.0)
                torch.nn.init.constant_(self.v.bias, 0.0)
        else:
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            if self.in_proj.bias is not None:
                torch.nn.init.constant_(self.in_proj.bias, 0.0)
                torch.nn.init.constant_(self.out_proj.bias, 0.0)
        return

    def in_projection(self, x):
        """
        Args:
            q, k, v: torch.Tensor of shape (B, S, D)
        Returns:
            q, k, v: torch.Tensor of shape (B, H, S, D_head)
        """
        if self.split_qkv:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
        else:
            q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k, v = (
            q.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
            k.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
            v.unflatten(-1, (self.heads, -1)).swapaxes(1, 2),
        )
        return q, k, v

    def forward(self, x):
        q, k, v = self.in_projection(x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.permute(0, 2, 1, 3).flatten(-2, -1)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, heads, width, causal, scale=True, bias=False, mlp="default"):
        super().__init__()
        self.ln_1 = LayerNorm(width, scale=scale, bias=bias)
        self.attn = MultiHeadSelfAttention(width, heads, causal, bias=bias)
        self.ln_2 = LayerNorm(width, scale=scale, bias=bias)
        if mlp == "default":
            self.mlp = MLP(
                width=4 * width, input_dim=width, output_dim=width, bias=bias
            )
        elif mlp == "geglu":
            self.mlp = GatedMLP(
                width=4 * width,
                input_dim=width,
                output_dim=width,
                act_fn=nn.GELU,
                bias=bias,
            )
        elif mlp == "swiglu":
            self.mlp = GatedMLP(
                width=4 * width,
                input_dim=width,
                output_dim=width,
                act_fn=nn.SiLU,
                bias=bias,
            )
        else:
            raise ValueError(f"MLP architecture {mlp} not supported")

    def forward(self, x, probs=None):
        x = x + self.attn(self.ln_1(x))
        if probs is not None:
            res, probs = self.mlp(self.ln_2(x), probs=probs)
            x = x + res
        else:
            res, probs = self.mlp(self.ln_2(x))
            x = x + res
        return x, probs


class MultiLayerRoutedGPT(GPT):
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

        super().__init__(heads, width, depth, block_size, vocab_size)
        self.config = Config(locals())
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, width),
                wpe=nn.Embedding(block_size, width),
                h=nn.ModuleList(
                    [
                        Block(
                            heads, width, causal=True, scale=scale, bias=bias, mlp=mlp
                        )
                        for _ in range(depth)
                    ]
                ),
                ln_f=LayerNorm(width, scale=scale, bias=bias),
            )
        )
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        if tie_weights:
            self.transformer.wte.weight = self.lm_head.weight
        self.init_weights()

        self.library = Library()
        self.router_type = None
        self.single_router = False
        self.num_multi_layer_experts = 0
        return

    def _add_experts(self, args, router_config):
        router_blocks = {"standard": StandardBlock}
        router_block = router_blocks[args.router_method]

        for i, layer in enumerate(self.transformer.h):
            self.library.add_model(f"model_0_mlp_{i}", layer.mlp)

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
        router_config.in_dim = args.width
        router_config.expert_dim = args.width

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

            print(f"Loading expert from model {model_name}")
            if model_name == "pretrained":
                if args.arch == "gpt2":
                    print(f"loading pretrained model from ... {GPT2checkpoints.base()}")
                    model.load_state_dict(torch.load(GPT2checkpoints.base()))
                elif args.arch == "gpt2-medium":
                    print(
                        f"loading pretrained model from ... {GPT2checkpoints.medium()}"
                    )
                    model.load_state_dict(torch.load(GPT2checkpoints.medium()))
                elif args.arch == "gpt2-large":
                    print(
                        f"loading pretrained model from ... {GPT2checkpoints.large()}"
                    )
                    model.load_state_dict(torch.load(GPT2checkpoints.large()))
                elif args.arch == "gpt2-xl":
                    print(f"loading pretrained model from ... {GPT2checkpoints.xl()}")
                    model.load_state_dict(torch.load(GPT2checkpoints.xl()))
                else:
                    raise ValueError(f"Unknown architecture {args.arch}")
            else:
                model.load_state_dict(torch.load(model_name)["model"])

            for i, layer in enumerate(model.transformer.h):
                self.library.add_model(f"model_{route_id+1}_mlp_{i}", layer.mlp)
            del model

        '''
            when self.num_multi_layer_experts = 1, we only have one layer of experts - should replicate routed_transformer
        '''
        for i, layer in enumerate(self.transformer.h):
            experts = nn.ModuleDict(
                    {
                        f"mlp_{j}": self.library.get_model(f"model_{j}_mlp_{i}")
                        for j in range(len(args.router_ckpts))
                    }
                )
            
            final_c = 0
            # go up till you reach limit
            for c in range(1, self.num_multi_layer_experts):
                if i + c >= len(self.transformer.h):
                    break
            
                final_c = c
                experts.update(
                    {
                        f"mlp_{j + c * len(args.router_ckpts)}": self.library.get_model(f"model_{j}_mlp_{i+c}")
                        for j in range(len(args.router_ckpts))
                    }
                )

            n_down = self.num_multi_layer_experts - 1 - final_c

            # then go down on the remaining 
            for c in range(1, n_down+1):
                experts.update(
                    {
                        f"mlp_{j + (final_c + c) * len(args.router_ckpts)}": self.library.get_model(f"model_{j}_mlp_{i-c}")
                        for j in range(len(args.router_ckpts))
                    }
                )

            device = torch.device(f"cuda:{D.local_rank()}")
            self.transformer.h[i].mlp = router_block(
                router_config, experts, passthrough=(i > 0 and self.single_router)
            ).to(device=device, dtype=args.dtype)

    def load_weights(self, args, router_config):
        ckpts = args.router_ckpts
        state_dict = torch.load(ckpts[0])
        print(f"Loading expert from model {args.router_ckpts[0]}")
        self.load_state_dict(state_dict["model"])
        self.router_type = args.router_method
        self.single_router = args.single_router
        self.num_multi_layer_experts = args.num_multi_layer_experts
        assert self.num_multi_layer_experts > 0 # just to be sure 
        self._add_experts(args, router_config)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()

        e_msg = f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert t <= self.config.block_size, e_msg
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        c = 0
        for block in self.transformer.h:
            if c == 0 or self.single_router is False:
                x, probs = block(x)
                c += 1
            else:
                x, probs = block(x, probs=probs)

        x = self.transformer.ln_f(x)
        self.hidden_states = x  # track hidden_states
        return self.lm_head(x)

    def get_router_loss(self, loss, args):
        # if self.router_type == "standard_Block":
        #     return loss
        # elif self.router_type == "rpo_Block":
        # loop through all modules
        router_loss = 0
        router_log = []
        n = 0
        for name, module in self.named_modules():
            if name[-4:] == ".mlp":
                # get the router loss
                router_loss_i, stats = module.get_router_loss(loss, args)
                stats = stats.tolist()  # list of average weighting / probs per expert
                router_log.append(stats)
                router_loss += router_loss_i
                n += 1

        # average over all layers
        # router_log = torch.tensor(router_log).mean(0) # average over all layers
        # wandb_log = {}
        # for i, log in enumerate(router_log):
        #     wandb_log[f'expert_{i}'] = log

        # record each layer
        wandb_log = {}
        for i, log in enumerate(router_log):
            for j, log_j in enumerate(log):
                wandb_log[f"expert_{j}_layer_{i}"] = log_j

        router_loss /= n
        return router_loss, wandb_log
