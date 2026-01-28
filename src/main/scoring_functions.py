"""
Scoring functions for OOD detection.

All scoring functions take in:

    - ID embeddings: (N, D)
    - New input embeddings: (M, D)

And return:

    - OOD score: (M,)
"""

import torch
from utils.misc import mahalanobis_from_stats


def mahalanobis_distance(
    id_embeddings: torch.Tensor,
    new_input_embeddings: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mahalanobis OOD score.

    Args:
        id_embeddings: (N, D)
        new_input_embeddings: (M, D)
        eps: float

    Returns:
        scores: (M,)
        Higher scores = more OOD.
    """
    if id_embeddings.ndim == 1:
        id_embeddings = id_embeddings.unsqueeze(0)
    if new_input_embeddings.ndim == 1:
        new_input_embeddings = new_input_embeddings.unsqueeze(0)
    assert (
        id_embeddings.shape[1] == new_input_embeddings.shape[1]
    ), "ID embeddings and new input embeddings must have the same number of dimensions"

    mean = id_embeddings.mean(dim=0)  # (D,)
    X_centered = id_embeddings - mean
    covariance = X_centered.T @ X_centered / (id_embeddings.shape[0] - 1)  # (D, D)

    scores = mahalanobis_from_stats(
        new_input_embeddings,
        mean,
        covariance,
        eps=eps,
    )

    return scores


def k_nearest_neighbors(k: int):
    """
    Return a function that computes the mean of the k-nearest neighbors distances
    between the ID embeddings and the new input embeddings.
    """

    def k_nearest_neighbors_function(id_embeddings, new_input_embeddings):
        return k_nearest_neighbors_distances(
            id_embeddings, new_input_embeddings, k
        ).mean(dim=1)

    return k_nearest_neighbors_function


################################################################################
#                               Helper functions                               #
################################################################################


def k_nearest_neighbors_distances(
    id_embeddings: torch.Tensor, new_input_embeddings: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Compute the k-nearest neighbors distance between the ID embeddings and the new input embeddings.
    """
    MAX_ID_SAMPLES = 1_000

    if id_embeddings.ndim > 1:
        assert (
            id_embeddings.shape[1] == new_input_embeddings.shape[1]
        ), "ID embeddings and new input embeddings must have the same number of dimensions"
    else:
        id_embeddings = id_embeddings.unsqueeze(-1)
        new_input_embeddings = new_input_embeddings.unsqueeze(-1)

    num_id = id_embeddings.shape[0]
    assert k > 0 and k <= num_id

    if num_id > MAX_ID_SAMPLES:
        generator = torch.Generator(device=id_embeddings.device)
        generator.manual_seed(42)
        perm = torch.randperm(num_id, device=id_embeddings.device, generator=generator)
        keep = perm[:MAX_ID_SAMPLES]
        id_embeddings = id_embeddings[keep]
        num_id = id_embeddings.shape[0]
        if k > num_id:
            k = num_id

    x2 = (new_input_embeddings**2).sum(dim=1, keepdim=True)  # (M, 1)
    y2 = (id_embeddings**2).sum(dim=1).unsqueeze(0)  # (1, N)
    xy = new_input_embeddings @ id_embeddings.T  # (M, N)

    dists = x2 + y2 - 2 * xy
    dists = torch.clamp(dists, min=0.0)
    dists = torch.sqrt(dists)

    knn_dists, _ = torch.topk(dists, k=k, largest=False, dim=1)

    return knn_dists  # (M, k)


def identity(id_embeddings, new_input_embeddings):
    return new_input_embeddings
