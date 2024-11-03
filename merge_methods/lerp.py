'''
Code taken from: https://github.com/Digitous/LLM-SLERP-Merge/blob/main/slerpmergelm.py
'''

import torch

def lerp(t: float,
         v0: torch.Tensor,
         v1: torch.Tensor):
    return (1 - t) * v0 + t * v1