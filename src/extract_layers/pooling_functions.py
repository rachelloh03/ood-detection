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


def pool_last_k_tokens(h, k=10):
    """
    Pool using the last k tokens.
    Args:
        h: (B, L, D) tensor of hidden states
        k: number of last tokens to pool
    Returns:
        (B, 2*D) tensor of pooled hidden states
    """
    if k is None:
        k = h.size(1)
    mean = h[:, -k:].mean(dim=1)
    std = h[:, -k:].std(dim=1)
    return torch.cat([mean, std], dim=-1)
