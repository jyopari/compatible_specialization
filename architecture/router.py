import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RouterBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        return


class StandardBlockConfig:
    def __init__(
        self,
        in_dim,
        expert_dim,
        n_layers=1,
        top_k=1,
        normalize=False,
        noise_scale=0.0,
        weight_init="kaiming",
        bias=False,
        use_softmax=True,
    ):
        self.name = "StandardBlock"
        self.in_dim = in_dim
        self.expert_dim = expert_dim
        self.n_layers = n_layers
        self.top_k = top_k
        self.normalize = normalize
        self.noise_scale = noise_scale
        self.weight_init = weight_init
        self.bias = bias
        self.use_softmax = use_softmax


class StandardBlock(RouterBlock):
    def __init__(self, config, experts, passthrough=False):
        super(StandardBlock, self).__init__()
        self.experts = experts
        self.num_experts = len(self.experts.keys())
        self.top_k = config.top_k
        self.normalize = config.normalize
        self.noise_scale = config.noise_scale
        self.weight_init = config.weight_init
        self.use_softmax = config.use_softmax

        if passthrough is False:
            self.router_policy = []
            for _ in range(config.n_layers - 1):
                self.router_policy += [
                    nn.Linear(config.in_dim, config.in_dim, bias=config.bias)
                ]
                self.router_policy += [nn.GELU()]
            self.router_policy += [
                nn.Linear(config.in_dim, self.num_experts, bias=config.bias)
            ]
            self.router_policy = nn.Sequential(*self.router_policy)
            self._init_weights()
        self.probs = None

    def _init_weights(self):
        for m in self.router_policy.modules():
            if isinstance(m, nn.Linear):
                if self.weight_init == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                if self.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(m.weight) * 0.1
                if self.weight_init == "zero":
                    nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        input_shape = x.size()
        if kwargs.get("probs", None) is None:
            x = x.reshape(-1, x.size(-1))
            logits = self.router_policy(x)

            probs = torch.softmax(logits, dim=-1)
            
            top_k_indices = probs.topk(self.top_k).indices

            router_mask = torch.zeros_like(probs).scatter_(-1, top_k_indices, 1)
            router_probs = torch.zeros_like(probs).scatter_(
                1, top_k_indices, probs.gather(-1, top_k_indices)
            )

            top_k_indices = top_k_indices.view(*input_shape[:-1], -1)
            router_mask = router_mask.view(*input_shape[:-1], -1)
            router_probs = router_probs.view(*input_shape[:-1], -1)
            probs = probs.view(*input_shape[:-1], -1)
        else:
            router_dict = kwargs["probs"]
            top_k_indices = router_dict["top_k_indices"]
            router_mask = router_dict["router_mask"]
            router_probs = router_dict["router_probs"]
            probs = router_dict["probs"]

        if self.normalize:  # allow gradients to flow
            if self.top_k == 1:
                router_probs = (
                    router_probs / router_probs.sum(dim=-1, keepdim=True).detach()
                )
            else:
                router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)

        self.probs = {
            "top_k_indices": top_k_indices,
            "router_mask": router_mask,
            "router_probs": router_probs,
            "probs": probs,
        }

        x = x.view(input_shape)
        next_states = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[..., idx].bool()
            probs = router_probs[..., [idx]][token_indices]
            next_states[token_indices] += probs * expert(x[token_indices])
        return next_states, self.probs

    def get_router_loss(self, loss, args):
        if self.probs is None:
            raise ValueError("No probs available. Run forward pass first.")
        stats = self.probs["router_probs"].mean(dim=(0, 1)).detach()
        loss = loss.mean()
        return loss, stats
