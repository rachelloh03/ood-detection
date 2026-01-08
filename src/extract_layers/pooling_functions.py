"""
Various ways to reduce the input hidden states to a single vector.
"""

import torch


def pool_mean_std(h):
    """
    Pool using mean and std across sequence dimension.
    Args:
        h: (B, L, D) tensor of hidden states
    Returns:
        (B, 2*D) tensor of pooled hidden states
    """
    mean = h.mean(dim=1)
    std = h.std(dim=1)
    pooled = torch.cat([mean, std], dim=-1)
    return pooled


# TODO: add other pooling functions. Maybe taking the last few D-dimensional vectors instead?
# or concatenating some of them together?
