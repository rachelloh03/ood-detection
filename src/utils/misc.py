"""Miscellaneous utility functions."""

import torch
import math


def mahalanobis_from_stats(
    x: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor | None = None,
    inv_covariance: torch.Tensor | None = None,
    squared: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes the Mahalanobis distance between the samples and the mean.

    Args:
    x: (N, d)
    mean: (d,)
    covariance: (d, d)
    inv_covariance: (d, d)

    Either covariance or inv_covariance must be provided.

    Returns:
        distances (N,)
    """

    diff = x - mean

    if inv_covariance is not None:
        dist_sq = torch.einsum("...d,dd,...d->...", diff, inv_covariance, diff)

    elif covariance is not None:
        d = covariance.shape[0]
        eye = torch.eye(d, device=covariance.device, dtype=covariance.dtype)
        cov = covariance + eps * eye

        sol = torch.linalg.solve(cov, diff.unsqueeze(-1)).squeeze(-1)
        dist_sq = torch.sum(diff * sol, dim=-1)

    else:
        raise ValueError("Either covariance or inv_covariance must be provided")

    return dist_sq if squared else torch.sqrt(dist_sq)


def gaussian_log_prob(x, mean, cov):
    """
    x: (N, D)
    mean: (D,)
    cov: (D, D)


    returns: (N,)
    """
    D = x.shape[1]
    diff = x - mean  # (N, D)

    L = torch.linalg.cholesky(cov)  # (D, D)
    solve = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)  # (N, D)

    maha = (diff * solve).sum(dim=1)  # (N,)
    logdet = 2 * torch.log(torch.diagonal(L)).sum()

    return -0.5 * (maha + logdet + D * math.log(2 * math.pi))
