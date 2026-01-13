from src.main.scoring_functions import mahalanobis_distance
import torch


def test_mahalanobis_distance():
    """
    Test the mahalanobis_distance function with a simple example.
    """

    id_embeddings = torch.tensor([[1.0, 1.0], [1.1, 0.9], [0.9, 1.2]])
    new_inputs = torch.tensor([[1.05, 1.0], [3.0, 3.0], [5.0, 5.0]])

    distances = mahalanobis_distance(id_embeddings, new_inputs)

    print("Distances:", distances)

    assert (
        distances.shape[0] == new_inputs.shape[0]
    ), "Output shape should match number of new inputs"

    assert (distances >= 0).all(), "All distances should be non-negative"

    assert (
        distances[0] < distances[1] and distances[1] < distances[2]
    ), "Close points should have smaller Mahalanobis distance than far points"


if __name__ == "__main__":
    test_mahalanobis_distance()
