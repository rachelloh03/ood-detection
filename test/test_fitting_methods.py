"""Test the fitting methods."""

import itertools

import torch
from src.transformations.transformation_functions import GaussianMixture


def _params_match_up_to_permutation(
    params: torch.Tensor,
    target_params: torch.Tensor,
    atol: float = 1e-2,
) -> bool:
    """Check if params match target_params up to a permutation of COLUMNS."""
    assert (
        params.shape == target_params.shape
    ), "Params and target_params must have the same shape"
    k = params.shape[0]
    for perm in itertools.permutations(range(k)):
        perm = list(perm)
        if torch.allclose(params[perm], target_params, atol=atol):
            return True
    return False


def test_gaussian_mixture():
    """Test GaussianMixture."""
    torch.manual_seed(42)
    gm = GaussianMixture(n_components=3, dim=4)
    mean1, std1 = (
        torch.tensor([5.0, 5.0, 5.0, 5.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0]) + 1e-6,
    )
    data1 = torch.randn(100, 4) * std1 + mean1
    mean2, std2 = (
        torch.tensor([10.0, 0.0, 0.0, 0.0]),
        torch.tensor([2.0, 1.0, 2.0, 2.0]) + 1e-6,
    )
    data2 = torch.randn(100, 4) * std2 + mean2
    mean3, std3 = (
        torch.tensor([-10.0, -10.0, -10.0, -10.0]),
        torch.tensor([2.0, 2.0, 1.0, 2.0]) + 1e-6,
    )
    data3 = torch.randn(100, 4) * std3 + mean3
    data = torch.cat([data1, data2, data3], dim=0)
    assert data.shape == (300, 4), f"Expected shape (300, 4), got {data.shape}"
    gm.fit(data, num_epochs=100)
    assert gm.is_fitted
    assert gm.mean_.shape == (3, 4)
    assert gm.covariance_.shape == (3, 4)
    assert gm.weights_.shape == (3,)

    means, covariances, weights = gm.get_params()
    target_means = torch.stack([mean1, mean2, mean3])  # (3, 4)
    target_covs = torch.stack([std1**2, std2**2, std3**2])  # (3, 4)
    target_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    assert _params_match_up_to_permutation(means, target_means, atol=1)
    assert _params_match_up_to_permutation(covariances, target_covs, atol=1)
    assert torch.allclose(weights, target_weights, atol=1)


if __name__ == "__main__":
    test_gaussian_mixture()
