import torch
import pytest
from src.main.transformation_functions import KMeansDistance


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


if __name__ == "__main__":
    test_kmeans_distance()
