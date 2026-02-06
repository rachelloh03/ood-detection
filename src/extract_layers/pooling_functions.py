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
    # Check for NaN in input
    if torch.isnan(h).any():
        raise ValueError(
            f"pool_mean_std: Input contains NaN values. "
            f"Shape: {h.shape}, NaN count: {torch.isnan(h).sum().item()}"
        )

    mean = h.mean(dim=1)
    # Use unbiased=False to avoid NaN when L=1 (dividing by n-1 = 0)
    # When L=1, std should be 0 (single value has no variance)
    std = h.std(dim=1, unbiased=False)
    pooled = torch.cat([mean, std], dim=-1)

    # Check for NaN in output
    if torch.isnan(pooled).any():
        raise ValueError(
            f"pool_mean_std: Output contains NaN values. "
            f"Input shape: {h.shape}, Output shape: {pooled.shape}, "
            f"NaN in mean: {torch.isnan(mean).any().item()}, "
            f"NaN in std: {torch.isnan(std).any().item()}"
        )

    return pooled


def pool_last_k_tokens_base(h, k=10):
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

    # Check for NaN in input
    if torch.isnan(h).any():
        raise ValueError(
            f"pool_last_k_tokens_base: Input contains NaN values. "
            f"Shape: {h.shape}, k: {k}, NaN count: {torch.isnan(h).sum().item()}"
        )

    h_slice = h[:, -k:]
    mean = h_slice.mean(dim=1)
    std = h_slice.std(dim=1, unbiased=False)
    pooled = torch.cat([mean, std], dim=-1)

    # Check for NaN in output
    if torch.isnan(pooled).any():
        raise ValueError(
            f"pool_last_k_tokens_base: Output contains NaN values. "
            f"Input shape: {h.shape}, k: {k}, Output shape: {pooled.shape}, "
            f"NaN in mean: {torch.isnan(mean).any().item()}, "
            f"NaN in std: {torch.isnan(std).any().item()}"
        )

    return pooled


def pool_last_k_tokens(k=10):
    def pool_last_k_tokens_function(h):
        return pool_last_k_tokens_base(h, k)

    pool_last_k_tokens_function.__name__ = f"pool_last_k_tokens(k={k})"
    return pool_last_k_tokens_function


def norm_weighted_mean_std(h, eps=1e-8):
    """
    Pool using a norm-weighted mean and std across sequence dimension.

    Args:
        h: (B, L, D)
    Returns:
        pooled: (B, 2D)
    """
    norms = torch.norm(h, dim=-1)  # (B, L)
    weights = norms / (norms.sum(dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(-1)  # (B, L, 1)

    mean = (h * weights).sum(dim=1)  # (B, D)

    var = (weights * (h - mean.unsqueeze(1)) ** 2).sum(dim=1)
    std = torch.sqrt(var + eps)  # (B, D)

    return torch.cat([mean, std], dim=-1)  # (B, 2*D)
