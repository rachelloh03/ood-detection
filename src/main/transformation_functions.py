"""
Different transformation functions for the input data
that can be used to create a Transformations object.
"""

from constants.model_constants import DEVICE
from extract_layers.extract_layers_main import extract_representations
from extract_layers.pooling_functions import pool_mean_std
import torch
import numpy as np


def extract_layer_with_mean_std_pooling(model, layer_idxs):
    """
    Extract the layer with mean and std pooling.

    The returned function takes in a DataLoader or a tensor
    and returns a tensor of shape (len(layer_idxs), _, 2*D)
    """
    model.eval()
    model.to(DEVICE)

    def extract_layer_function(data):
        buffers = extract_representations(
            model,
            data,
            pool_mean_std,
            layer_idxs,
        )
        # sort buffers by layer_idx
        buffers = sorted(buffers.items(), key=lambda x: x[0])
        return torch.cat([torch.tensor(buffer) for _, buffer in buffers], dim=0)

    return extract_layer_function


class Subsample:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.feature_indices = None
        torch.manual_seed(42)
        np.random.seed(42)

    def fit(self, x):
        if isinstance(x, torch.Tensor):
            n_total_features = x.shape[1]
            n_select = min(self.n_features, n_total_features)
            self.feature_indices = torch.randperm(n_total_features)[:n_select]
        else:
            n_total_features = x.shape[1]
            n_select = min(self.n_features, n_total_features)
            self.feature_indices = np.random.choice(
                n_total_features, n_select, replace=False
            )

    def transform(self, x):
        if self.feature_indices is None:
            raise ValueError("Subsample must be fitted before transform")
        if isinstance(x, torch.Tensor):
            return x[:, self.feature_indices]
        else:
            return x[:, self.feature_indices]

    def __call__(self, x):
        return self.transform(x)
