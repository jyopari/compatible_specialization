'''
Code taken from: https://github.com/Digitous/LLM-SLERP-Merge/blob/main/slerpmergelm.py
'''

import torch
import numpy as np


def lerp(t: float,
         v0: np.ndarray,
         v1: np.ndarray):
    return (1 - t) * v0 + t * v1


def slerp(
        t: float,
        v0: torch.Tensor,
        v1: torch.Tensor,
        DOT_THRESHOLD: float = 0.9995,
        eps: float = 1e-8,
    ):

    # Convert tensors to a common format, float32
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)

    # Convert tensors to numpy arrays and normalize them
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)

    if norm_v0 > eps:
        v0 = v0 / norm_v0
    if norm_v1 > eps:
        v1 = v1 / norm_v1

    dot = np.sum(v0 * v1)
    if np.abs(dot) > DOT_THRESHOLD:
        return torch.from_numpy(lerp(t, v0_copy, v1_copy))

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    del v0_copy, v1_copy
    del v1
    return torch.from_numpy(v2)