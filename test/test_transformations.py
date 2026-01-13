import torch
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.main.transformations import Transformations


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


if __name__ == "__main__":
    test_transformations()
