"""
Class that stores torch Modules and provides methods to add, get, list and remove models.
"""

import torch.nn as nn


class Library:
    def __init__(self):
        self._models = {}

    def add_model(self, key, model):
        if not isinstance(model, nn.Module):
            raise ValueError("The model should be an instance of torch.nn.Module")
        self._models[key] = model

    def get_model(self, key):
        if key not in self._models:
            raise KeyError(f"No model found with the key: {key}")
        return self._models[key]

    def list_models(self):
        return list(self._models.keys())

    def remove_model(self, key):
        if key in self._models:
            del self._models[key]
        else:
            raise KeyError(f"No model found with the key: {key}")
