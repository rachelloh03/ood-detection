import torch
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformations.ensemble_transformations import EnsembleTransformations
from src.transformations.transformations import Transformations


def test_transformations():
    torch.manual_seed(42)
    random_matrix = torch.randn(2, 5)

    def multiply_by_random_matrix(x: torch.Tensor) -> torch.Tensor:
        return x @ random_matrix

    def add_tensor(x: torch.Tensor) -> torch.Tensor:
        return x + torch.ones_like(x)

    print("random_matrix", random_matrix)

    transformations = Transformations(
        [
            multiply_by_random_matrix,
            PCA(n_components=2),
            StandardScaler(),
            add_tensor,
        ]
    )

    id_train = torch.tensor([[1.0, 1.0], [1.1, 1.0], [0.9, 1.2]])

    transformations.fit(id_train)

    test = torch.tensor([[1.0, 1.0], [1.1, 1.0], [0.9, 1.2], [2.0, 3.0], [3.0, 5.0]])
    transformed = transformations(test)

    assert transformed[:3, :].sum() == pytest.approx(6.0)
    result = transformed[0, :] + transformed[4, :] - 2 * transformed[3, :]
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)


def test_ensemble_transformations():
    x = torch.randn(8, 20)
    base_transforms = [PCA(n_components=5), StandardScaler()]

    def mean_pooling(z):
        return z.mean(dim=0)

    ensemble = EnsembleTransformations(base_transforms, 3, 10, mean_pooling, seed=42)
    ensemble.fit(x)
    z = ensemble(x)
    assert z.shape == (8, 5)

    pca_1 = ensemble.transformations[0].transformations[0]
    pca_2 = ensemble.transformations[1].transformations[0]

    # checks that the PCA transformations are different
    reduced_x_1 = x[:, ensemble.feature_indices[0]]
    reduced_x_2 = x[:, ensemble.feature_indices[1]]
    assert reduced_x_1.shape == (8, 10)
    assert reduced_x_2.shape == (8, 10)
    post_pca_1 = pca_1.transform(reduced_x_1)
    post_pca_2 = pca_2.transform(reduced_x_2)
    assert post_pca_1.shape == (8, 5)
    assert post_pca_2.shape == (8, 5)
    assert not np.allclose(post_pca_1, post_pca_2, atol=1e-5)


if __name__ == "__main__":
    test_transformations()
    test_ensemble_transformations()
