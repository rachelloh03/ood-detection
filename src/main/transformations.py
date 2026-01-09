from collections.abc import Callable, Sequence
import torch


class Transformations:
    """
    Transformations that applies a list of transformations to the input.

    A transformation is either a function or an object with a __call__ method that can be fit to input data, e.g. PCA.

    Example:
        transformations = [PCA(n_components=10), StandardScaler(), lambda x: x + 1]
    """

    def __init__(
        self,
        transformations: Sequence[Callable],
    ):
        """
        Args:
            transformations: A list of transformations to apply to the input.
        """
        self.transformations = list(transformations)

    def fit(self, x: torch.Tensor) -> None:
        """
        Fit the embedding function on the input data.

        Args:
            x: Input data to fit the embedding function on.
        """
        for t in self.transformations:
            if hasattr(t, "fit"):
                t.fit(x)
            if hasattr(t, "transform"):
                x = t.transform(x)
            else:
                x = t(x)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the embedding function to the input.

        Args:
            x: Input data to apply the embedding function to.

        Returns:
            x: The embedded input data.
        """
        for t in self.transformations:
            if hasattr(t, "transform"):
                x = t.transform(x)
            else:
                x = t(x)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
        return x
