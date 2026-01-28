"""
Different transformation functions for the input data
that can be used to create a Transformations object.

These are either functions or objects with a __call__ method that can be fit to input data, e.g. PCA.

Transformations:
1. extract_layer_transformation: Extract the l-th hidden layer of a model.
2. GaussianMixtureWithMD: Fit a Gaussian mixture model to the data,
finds the minimum Mahalanobis distance to the cluster centers.
3. KMeansDistance: Compute distances to k-means cluster centers.
4. Subsample: Subsample the features (pick a random subset of features).
"""

from constants.model_constants import DEVICE
from extract_layers.extract_layers_main import extract_representations
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as SklearnGMM
from utils.misc import gaussian_log_prob, mahalanobis_from_stats


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


class GaussianMixture(nn.Module):
    """
    Gaussian Mixture Model with EM training.
    """

    def __init__(
        self,
        n_components: int,
        dim: int,
        diagonal_cov: bool = True,
        random_state: int | None = None,
    ):
        super().__init__()

        self.n_components = n_components
        self.dim = dim
        self.diagonal_cov = diagonal_cov
        self.random_state = random_state
        self.gmm = None

        self.mean_ = None
        self.covariance_ = None
        self.weights_ = None
        self.is_fitted = False

    def fit(
        self,
        samples: torch.Tensor,
        num_epochs: int | None = None,
        tolerance: float | None = None,
        save_filepath: str | None = None,
    ) -> None:
        """
        Fit the Gaussian Mixture Model using EM.
        """
        if num_epochs is None and tolerance is None:
            raise ValueError("Either num_epochs or tolerance must be provided")

        if samples.ndim != 2:
            raise ValueError(f"Expected (N, D), got {samples.shape}")

        if samples.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {samples.shape[1]}")

        if samples.shape[0] < self.n_components:
            raise ValueError(f"Need at least {self.n_components} samples")

        X = samples.detach().cpu().numpy()

        covariance_type = "diag" if self.diagonal_cov else "full"

        tol = tolerance if tolerance is not None else 1e-3
        max_iter = num_epochs if num_epochs is not None else 100

        gmm = SklearnGMM(
            n_components=self.n_components,
            covariance_type=covariance_type,
            tol=tol,
            max_iter=max_iter,
            init_params="kmeans",
            n_init=1,
            random_state=self.random_state,
        )

        self.gmm = gmm
        self.gmm.fit(X)

        device = samples.device
        dtype = samples.dtype

        self.mean_ = torch.tensor(self.gmm.means_, device=device, dtype=dtype)
        self.weights_ = torch.tensor(self.gmm.weights_, device=device, dtype=dtype)

        if self.diagonal_cov:
            self.covariance_ = torch.tensor(
                self.gmm.covariances_, device=device, dtype=dtype
            )  # (k, D)
        else:
            self.covariance_ = torch.tensor(
                self.gmm.covariances_, device=device, dtype=dtype
            )  # (k, D, D)

        if save_filepath is not None:
            torch.save(
                {
                    "mean_": self.mean_,
                    "covariance_": self.covariance_,
                    "weights_": self.weights_,
                },
                f"{save_filepath}.pth",
            )

        self.is_fitted = True

    def get_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return GMM parameters.

        Returns:
            means: (k, D)
            covariances: (k, D) if diagonal, (k, D, D) if full
            mix_weights: (k,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling get_params()")

        return self.mean_, self.covariance_, self.weights_


class GaussianMixtureWithScore:
    """
    Fit a Gaussian mixture model to the data,
    finds the minimum Mahalanobis distance to the cluster centers.
    """

    def __init__(
        self,
        n_components: int,
        dim: int,
        diagonal_cov: bool = True,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.dim = dim
        self.random_state = random_state
        self.diagonal_cov = diagonal_cov
        self.gm = GaussianMixture(
            n_components=n_components,
            dim=dim,
            diagonal_cov=diagonal_cov,
            random_state=random_state,
        )
        self.__name__ = f"GaussianMixtureWithMD(n_components={n_components}, dim={dim}, diagonal_cov={diagonal_cov})"

    def fit(self, x, tolerance: float = 1e-4):
        """
        Fit the Gaussian mixture model to the data.
        """
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {x.shape[1]}")
        self.gm.fit(
            x,
            tolerance=tolerance,
            save_filepath=f"gaussian_mixture_with_md_{self.n_components}_{self.dim}_{self.diagonal_cov}",
        )  # x: (N, D)
        print("Mixture weights:", self.gm.get_params()[2])
        print("ELBO", self.gm.gmm.lower_bound_)
        return self.transform(x)  # (N,)

    def transform(self, x):
        """
        Transform the data by computing the minimum Mahalanobis distance to the cluster centers.

        Args:
            x: (N, D)

        Returns:
            min_distances: (N,)
        """
        return self.density(x)  # (N,)

    def min_md(self, x):
        """
        Compute the minimum Mahalanobis distance to the cluster centers.
        """
        means, covariances, _ = self.gm.get_params()
        if self.diagonal_cov:
            covariances = torch.diag_embed(covariances)
        distances = []
        for mean, covariance in zip(means, covariances):
            distance = mahalanobis_from_stats(x, mean, covariance, eps=1e-6)  # (N,)
            distances.append(distance)
        all_distances = torch.stack(distances)  # (k, N)
        min_distances = torch.min(all_distances, dim=0)[0]  # (N,)
        return min_distances

    def density(self, x):
        """
        Compute -log p(x) under a Gaussian Mixture Model.

        Args:
            x: (N, D)

        Returns:
            neg_log_density: (N,)
        """
        means, covariances, weights = self.gm.get_params()  # (K,D), (K,D or D,D), (K,)

        if self.diagonal_cov:
            covariances = torch.diag_embed(covariances)  # (K,D,D)

        log_probs = []

        for mean, covariance, weight in zip(means, covariances, weights):
            log_prob = gaussian_log_prob(x, mean, covariance)
            log_prob = log_prob + torch.log(weight)
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=0)  # (K, N)
        log_px = torch.logsumexp(log_probs, dim=0)  # (N,)

        return -log_px


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
