import os
import torch
import numpy as np


def auto_determine_dtype():
    compute_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float16
    torch_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float16
    print(f"compute_dtype:\t{compute_dtype}")
    print(f"torch_dtype:\t{torch_dtype}")
    return compute_dtype, torch_dtype


def check_bfloat16_support():
    # check if cuda version supports bfloat16
    device_capable = torch.cuda.is_bf16_supported()
    supported_devices = ["H100", "A100", "Ada", "3090", "3080", "4080", "4090"]
    device_can_utilize = np.any(
        [device in torch.cuda.get_device_name() for device in supported_devices]
    )
    return device_capable and device_can_utilize


def create_directory(dir):
    try:
        os.mkdir(dir)
        print(f"Directory '{dir}' created successfully.")
    except FileExistsError:
        raise FileExistsError(f"Directory '{dir}' already exists.")
