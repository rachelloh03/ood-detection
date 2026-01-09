"""
Scoring functions for OOD detection.

All scoring functions take in:

    - ID embeddings: (N, D)
    - New input embeddings: (M, D)

And return:

    - OOD score: (M,)
"""

import torch


def mahalanobis_distance(
    id_embeddings: torch.Tensor, new_input_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Mahalanobis distance between the ID embeddings and the new input embeddings.
    """
    assert (
        id_embeddings.shape[1] == new_input_embeddings.shape[1]
    ), "ID embeddings and new input embeddings must have the same number of dimensions"
    mu = id_embeddings.mean(dim=0)  # (D,)

    X_centered = id_embeddings - mu
    cov = X_centered.T @ X_centered / (id_embeddings.shape[0] - 1)  # (D, D)

    cov_inv = torch.linalg.pinv(cov)

    diff = new_input_embeddings - mu  # (M, D)
    left = diff @ cov_inv  # (M, D)
    distances_squared = (left * diff).sum(dim=1)  # (M,)
    distances = torch.sqrt(distances_squared + 1e-12)  # (M,)

    return distances
