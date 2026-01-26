"""
Different transformation functions for the input data
that can be used to create a Transformations object.
"""

from constants.model_constants import DEVICE
from extract_layers.extract_layers_main import extract_representations
import torch
import numpy as np
from sklearn.cluster import KMeans


def extract_layer_transformation(model, pooling_function, layer_idxs):
    """
    Extract the layer with mean and std pooling.

    The returned function takes in a DataLoader or a tensor
    and returns a tensor of shape (len(layer_idxs), _, 2*D)
    """
    model.eval()
    model.to(DEVICE)

    def extract_layer_function(data):
        buffers = extract_representations(
            model,
            data,
            pooling_function,
            layer_idxs,
        )
        # sort buffers by layer_idx
        buffers = sorted(buffers.items(), key=lambda x: x[0])
        return torch.cat([torch.tensor(buffer) for _, buffer in buffers], dim=0)

    return extract_layer_function


class Subsample:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.feature_indices = None
        torch.manual_seed(42)
        np.random.seed(42)

    def fit(self, x):
        if isinstance(x, torch.Tensor):
            n_total_features = x.shape[1]
            n_select = min(self.n_features, n_total_features)
            self.feature_indices = torch.randperm(n_total_features)[:n_select]
        else:
            n_total_features = x.shape[1]
            n_select = min(self.n_features, n_total_features)
            self.feature_indices = np.random.choice(
                n_total_features, n_select, replace=False
            )

    def transform(self, x):
        if self.feature_indices is None:
            raise ValueError("Subsample must be fitted before transform")
        if isinstance(x, torch.Tensor):
            return x[:, self.feature_indices]
        else:
            return x[:, self.feature_indices]

    def __call__(self, x):
        return self.transform(x)


class KMeansDistance:
    """
    Transform data by computing distances to k-means cluster centers.

    During fit, finds k cluster centers using k-means clustering.
    During transform, returns distances from each point to each cluster center.
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
        max_samples_for_fit: int | None = 2000,
    ):
        """
        Args:
            n_clusters: Number of clusters (k) for k-means
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_samples_for_fit = max_samples_for_fit
        self.kmeans = None
        self.cluster_centers_ = None
        self.__name__ = f"KMeansDistance(n_clusters={n_clusters})"

    def fit(self, x):
        """
        Fit k-means clustering on the data.

        Args:
            x: Input data of shape (n_samples, n_features)
               Can be torch.Tensor or numpy array
        """
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = np.array(x)

        # Optionally subsample rows to speed up KMeans on very large datasets
        if (
            self.max_samples_for_fit is not None
            and x_np.shape[0] > self.max_samples_for_fit
        ):
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(x_np.shape[0], self.max_samples_for_fit, replace=False)
            x_fit = x_np[indices]
        else:
            x_fit = x_np

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.kmeans.fit(x_fit)
        self.cluster_centers_ = self.kmeans.cluster_centers_

    def transform(self, x):
        """
        Transform data by computing distances to cluster centers.

        Args:
            x: Input data of shape (n_samples, n_features)
               Can be torch.Tensor or numpy array

        Returns:
            Distances to each cluster center, shape (n_samples, n_clusters)
            Returns torch.Tensor if input is torch.Tensor, numpy array otherwise
        """
        if self.cluster_centers_ is None:
            raise ValueError("KMeansDistance must be fitted before transform")

        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            x_np = x.cpu().numpy()
        else:
            x_np = np.array(x)

        distances = np.sqrt(
            (
                (x_np[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2
            ).sum(axis=2)
        )

        if is_torch:
            return torch.tensor(distances, dtype=x.dtype, device=x.device)
        else:
            return distances

    def __call__(self, x):
        return self.transform(x)  # (n_samples, n_clusters)

    def get_cluster_centers(self):
        if self.cluster_centers_ is None:
            raise ValueError("KMeansDistance must be fitted before get_cluster_centers")
        return self.cluster_centers_
