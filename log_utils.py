import wandb

import torch
import distributed as D

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


@torch.no_grad()
def get_kernel(model, data, pooling):
    representations = []
    for d in data:
        model(d[0].cuda(D.local_rank()))  # forward pass to aggreagate representations
        repr = model.get_hidden_states()
        if pooling == "avg":
            repr = repr.mean(dim=1)
        elif pooling == "last":
            repr = repr[:, -1]
        representations.append(repr.cpu())

    representations = torch.cat(representations)
    norms = torch.norm(representations, dim=1, keepdim=True)
    representations = representations / norms
    kernel = torch.einsum(
        "ij,kj->ik", representations, representations
    )  # cosine similarity
    return kernel


class wandb_logger:
    def __init__(self, project, args):
        wandb.init(
            project=project,
            config=args,
            name=f"{args.name}-{args.arch}-{args.lr}-{args.topk}-{args.datasets}",
        )
        wandb.define_metric("train/samples")
        wandb.define_metric("*", step_metric="train/samples")
        self.log_freq = args.log_freq
        self._step = 0
        self._train = True
        self._samples = 0
        return

    def increment(self, n):
        self._samples += n
        return

    def train(self, train=True):
        self._train = train

    def eval(self):
        return self.train(False)

    def log(self, log_dict, train=None):
        # allow for overriding for single line log calls
        if train is not None:
            self.train(train)

        # if mode is train and step is not a multiple of log_freq, skip logging
        if self._train and (self._step % self.log_freq) != 0:
            return

        # log to wandb and add prefix based on train/eval
        prefix = "train/" if self._train else "test/"
        log_dict["samples"] = self._samples
        if self._step % self.log_freq == 0 or not self._train:
            wandb.log({prefix + k: v for k, v in log_dict.items()}, step=self._step)
        return

    def step(self):
        self._step += 1
        return
