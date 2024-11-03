import os
import torch
import importlib
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .sampler import DistributedEvalSampler
import distributed as D


def load_dataset(args):
    ############################################################################################
    ######################################### Get data #########################################
    ############################################################################################

    print("\nGetting data...")
    print("===================================" * 3)

    train_data = []
    val_data = []
    for dataset_name in args.datasets:
        train_data.append(
            np.memmap(
                os.path.join("data", dataset_name, "train.bin"),
                dtype=np.uint16,
                mode="r",
            )
        )
        val_data.append(
            np.memmap(
                os.path.join("data", dataset_name, "val.bin"), dtype=np.uint16, mode="r"
            )
        )

    trainset = SimpleLLMDataset(train_data, args.context, args.train_tokens)
    testset = SimpleLLMDataset(val_data, args.context, args.train_tokens)

    if args.eval_sep is not None:
        # create seperate SimpleLLMDataset for each dataset
        sep_val_data = []
        for dataset_name in args.eval_sep:
            sep_val_data.append(
                np.memmap(
                    os.path.join("data", dataset_name, "val.bin"),
                    dtype=np.uint16,
                    mode="r",
                )
            )
        sep_testset = [
            SimpleLLMDataset([data], args.context, args.train_tokens)
            for data in sep_val_data
        ]

    if args.distributed:
        assert (
            args.batch_size % (D.world_size() * args.grad_accum) == 0
        ), "batch size must be divisible by grad_accum"
        print(f"cumulative batch_size:\t{args.batch_size}")
        print(f"world_size:           \t{D.world_size()}")
        print(f"grad_accum:           \t{args.grad_accum}")
        print(
            f"per-device batch_size:\t{int(args.batch_size/(D.world_size() * args.grad_accum))}"
        )

        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedEvalSampler(testset, shuffle=False)

        train_loader = DataLoader(
            trainset,
            batch_size=int(args.batch_size / (D.world_size() * args.grad_accum)),
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        test_loader = DataLoader(
            testset,
            batch_size=int(args.batch_size / (D.world_size() * args.grad_accum)),
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=test_sampler,
        )
    else:
        assert (
            args.batch_size % args.grad_accum == 0
        ), "batch size must be divisible by grad_accum"
        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size // args.grad_accum,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=True,
        )
        test_loader = DataLoader(
            testset,
            batch_size=args.batch_size // args.grad_accum,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
        )

        if args.eval_sep:
            # create seperate test dataloader for each dataset
            sep_test_loader = [
                DataLoader(
                    dataset,
                    batch_size=args.batch_size // args.grad_accum,
                    num_workers=args.workers,
                    shuffle=False,
                    pin_memory=True,
                )
                for dataset in sep_testset
            ]
            return train_loader, test_loader, sep_test_loader

    return train_loader, test_loader


class SimpleLLMDataset(torch.utils.data.Dataset):
    def __init__(self, data, context, train_tokens):
        self.data = data
        self.num_datasets = len(data)
        self.context = context
        self.train_tokens = train_tokens

        # index to start reading from (choose randomly and make train_tokens) for each dataset
        self.idx_mappings = [
            torch.randint(
                0, len(data) - context, size=(train_tokens // self.num_datasets,)
            )
            for data in self.data
        ]
        # dataset index for each token
        self.dataset_mappings = [
            torch.full_like(idx, i) for i, idx in enumerate(self.idx_mappings)
        ]
        self.dataset_mappings = torch.cat(self.dataset_mappings)
        self.data_size = sum([len(idx) for idx in self.idx_mappings])

    def __getitem__(self, index):
        dataset_idx = self.dataset_mappings[index]  # choose dataset
        st_idx = self.idx_mappings[dataset_idx][
            index % (self.train_tokens // self.num_datasets)
        ]  # choose starting index
        return (
            torch.tensor(
                self.data[dataset_idx][st_idx : st_idx + self.context].astype(int)
            ).long(),
            torch.tensor(
                self.data[dataset_idx][st_idx + 1 : st_idx + 1 + self.context].astype(
                    int
                )
            ).long(),
        )

    def __len__(self):
        return self.data_size
