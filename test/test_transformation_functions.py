import math
import torch
import pytest
from src.main.transformation_functions import GaussianMixtureWithMD, KMeansDistance


def test_kmeans_distance():
    """Test KMeansDistance with concrete 2D tensor inputs, shape validation, and error handling."""
    X_train = torch.tensor(
        [
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [5.0, 6.0],
            [5.5, 6.5],
            [6.0, 7.0],
            [10.0, 11.0],
            [10.5, 11.5],
        ],
        dtype=torch.float32,
    )
    X_test = torch.tensor([[1.2, 2.3], [5.3, 6.4], [10.2, 11.3]], dtype=torch.float32)

    kmeans = KMeansDistance(n_clusters=3, random_state=42)

    with pytest.raises(ValueError, match="must be fitted before transform"):
        kmeans.transform(X_test)

    kmeans.fit(X_train)

    distances = kmeans.transform(X_test)
    assert distances.shape == (3, 3), f"Expected shape (3, 3), got {distances.shape}"

    assert isinstance(
        distances, torch.Tensor
    ), "Should return torch.Tensor for torch input"

    assert torch.all(distances >= 0), "Distances should be non-negative"

    distances_train = kmeans.transform(X_train)
    assert distances_train.shape == (
        8,
        3,
    ), f"Expected shape (8, 3), got {distances_train.shape}"

    expected_distances = torch.tensor(
        [
            [0.360555, 12.728119, 6.010823],
            [5.445181, 6.930007, 0.223607],
            [12.374571, 0.070711, 6.717886],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(
        distances, expected_distances, atol=1e-5
    ), f"Distances don't match expected values. Got {distances}, expected {expected_distances}"

    print("Cluster centers:", kmeans.get_cluster_centers())


def test_gaussian_mixture_with_md():
    """Test GaussianMixtureWithMD with concrete 2D tensor inputs, shape validation, and error handling."""
    # bimodal data, one cluster around (0, 0) and one cluster around (10, 10)
    X_train = torch.cat(
        [
            torch.randn(100, 2) * 0.5 + torch.tensor([0.0, 0.0]),
            torch.randn(100, 2) * 0.5 + torch.tensor([10.0, 10.0]),
        ],
        dim=0,
    )

    gmm = GaussianMixtureWithMD(n_components=2, dim=2)
    gmm.fit(X_train, num_epochs=1000)

    means, covariances, weights = gmm.gm.get_params()
    print("means:", means)
    print("covariances:", covariances)
    print("weights:", weights)

    X_test = torch.tensor(
        [
            [-1, 1],
            [0.5, 0.5],
            [5, 5],
            [10, 10],
            [12, 7],
        ]
    )
    test_distances = gmm.transform(X_test)
    # expect distances to be the euclidean distance to the nearest point / 0.5**2
    expected_test_distances = torch.tensor(
        [
            math.sqrt(2) / 0.5,
            math.sqrt(0.5) / 0.5,
            math.sqrt(50) / 0.5,
            math.sqrt(0) / 0.5,
            math.sqrt(13) / 0.5,
        ]
    )
    print("expected_test_distances:", expected_test_distances)
    print("test_distances:", test_distances)

    assert torch.allclose(test_distances, expected_test_distances, atol=1)


if __name__ == "__main__":
    test_kmeans_distance()
    test_gaussian_mixture_with_md()
