import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if key in ["self", "__class__"]:
                continue
            setattr(self, key, value)


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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

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

        super().__init__()
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

        # track hidden_states
        self.hidden_states = None
        self.hidden_states_list = []
        return

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                if "out_proj" in n:
                    torch.nn.init.normal_(
                        m.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.depth)
                    )

            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        return

    def get_hidden_states(self):
        return self.hidden_states

    def forward(self, idx):
        self.hidden_states_list = []
        self.hidden_states = None

        device = idx.device
        b, t = idx.size()

        e_msg = f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert t <= self.config.block_size, e_msg
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
            self.hidden_states_list.append(x.mean(dim=1).cpu())
        x = self.transformer.ln_f(x)
        #self.hidden_states_list.append(x)
        self.hidden_states = x  # track hidden_states
        return self.lm_head(x)


class GPT2config:
    @staticmethod
    def base():
        args = edict()
        args.vocab_size = 50257
        args.depth = 12
        args.width = 768
        args.heads = 12
        args.block_size = 1024
        args.scale = True
        args.bias = True
        args.mlp = "default"
        return args

    def medium():
        args = edict()
        args.vocab_size = 50257
        args.depth = 24
        args.width = 1024
        args.heads = 16
        args.block_size = 1024
        args.scale = True
        args.bias = True
        args.mlp = "default"
        return args

    def large():
        args = edict()
        args.vocab_size = 50257
        args.depth = 36
        args.width = 1280
        args.heads = 20
        args.block_size = 1024
        args.scale = True
        args.bias = True
        args.mlp = "default"
        return args

    def xl():
        args = edict()
        args.vocab_size = 50257
        args.depth = 48
        args.width = 1600
        args.heads = 25
        args.block_size = 1024
        args.scale = True
        args.bias = True
        args.mlp = "default"
        return args


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
