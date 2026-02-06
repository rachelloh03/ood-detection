"""Ensemble Transformations."""

import torch
from collections.abc import Callable, Sequence
from copy import deepcopy
from transformations.transformations import Transformations


class EnsembleTransformations:
    """
    Ensemble of Transformations trained on random feature subspaces.

    This class allows you to create an ensemble of `Transformations` instances,
    each trained on a random subset of the input features. The outputs of the ensemble
    can then be combined using a pooling function.

    Args:
        base_transformations (Sequence[Callable]): Transformations applied to each subspace.
        num_ensembles (int): Number of ensemble members.
        num_dimensions (int): Number of input dimensions per ensemble member.
        ensemble_pooling (Callable[[torch.Tensor], torch.Tensor]): Function that pools ensemble outputs.
            Expected input shape: (E, B, D_out), returns shape (B, D_final)
        replace (bool, optional): Whether to sample dimensions with replacement. Default: False
        seed (int | None, optional): RNG seed for reproducibility. Default: None

    Example:
        >>> import torch
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.preprocessing import StandardScaler
        >>>
        >>> # Sample input: 100 samples, 20 features
        >>> x = torch.randn(100, 20)
        >>>
        >>> # Define a base set of transformations
        >>> base_transforms = [PCA(n_components=5), StandardScaler()]
        >>>
        >>> # Define a mean pooling function
        >>> def mean_pooling(z):
        ...     return z.mean(dim=0)
        >>>
        >>> # Create the ensemble
        >>> ensemble = EnsembleTransformations(
        ...     base_transformations=base_transforms,
        ...     num_ensembles=3,
        ...     num_dimensions=10,
        ...     ensemble_pooling=mean_pooling,
        ...     seed=42
        ... )
        >>>
        >>> # Fit ensemble on data
        >>> ensemble.fit(x)
        >>>
        >>> # Apply ensemble to new data
        >>> z = ensemble(x)
        >>> print(z.shape)
        torch.Size([100, 5])
    """

    def __init__(
        self,
        base_transformations: Sequence[Callable],
        num_ensembles: int,
        num_dimensions: int,
        ensemble_pooling: Callable[[torch.Tensor], torch.Tensor],
        replace: bool = False,
        seed: int | None = None,
    ):
        self.num_ensembles = num_ensembles
        self.num_dimensions = num_dimensions
        self.ensemble_pooling = ensemble_pooling
        self.replace = replace

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

        self.transformations = [
            Transformations(deepcopy(base_transformations))
            for _ in range(num_ensembles)
        ]

        self.feature_indices: list[torch.Tensor] = []

    def _sample_features(self, d: int) -> torch.Tensor:
        return (
            torch.randperm(d, generator=self._rng)[: self.num_dimensions]
            if not self.replace
            else torch.randint(0, d, (self.num_dimensions,), generator=self._rng)
        )

    def fit(self, x: torch.Tensor) -> None:
        _, d = x.shape
        if self.num_dimensions > d:
            raise ValueError("num_dimensions cannot exceed input dimension")

        self.feature_indices = []

        for t in self.transformations:
            idx = self._sample_features(d)
            self.feature_indices.append(idx)

            x_sub = x[:, idx]
            t.fit(x_sub)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for t, idx in zip(self.transformations, self.feature_indices):
            x_sub = x[:, idx]
            z = t(x_sub)
            outputs.append(z)

        stacked = torch.stack(outputs, dim=0)  # shape: (E, B, D_out)
        return self.ensemble_pooling(stacked)
