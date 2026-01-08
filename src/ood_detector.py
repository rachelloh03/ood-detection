"""Out-of-distribution detector using Mahalanobis distance."""

import os
import numpy as np
import torch
import pickle
from scipy import stats
from sklearn.covariance import EmpiricalCovariance


class OODDetector:
    """Out-of-distribution detector using Mahalanobis distance"""

    def __init__(self, method="mahalanobis"):
        self.method = method
        self.mean = None
        self.cov_inv = None
        self.fitted = False

        # For percentile-based threshold
        self.train_scores = None
        self.threshold = None

    def fit(self, X, contamination=0.05):
        """
        Fit the OOD detector on in-distribution training data

        Args:
            X: Training representations (n_samples, n_features)
            contamination: Expected proportion of outliers (for threshold)
        """
        print(f"Fitting OOD detector on {X.shape[0]} samples...")

        if self.method == "mahalanobis":
            # Compute mean and covariance
            self.mean = np.mean(X, axis=0)

            # Use empirical covariance with regularization
            cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(X)

            cov = cov_estimator.covariance_

            # Add small regularization for numerical stability
            reg = 1e-6 * np.trace(cov) / cov.shape[0]
            cov_reg = cov + reg * np.eye(cov.shape[0])

            self.cov_inv = np.linalg.inv(cov_reg)

            # Compute scores on training data for threshold
            self.train_scores = self._mahalanobis_distance(X)

            # Set threshold at (1-contamination) percentile
            self.threshold = np.percentile(self.train_scores, (1 - contamination) * 100)

        elif self.method == "euclidean":
            # Simple Euclidean distance from mean
            self.mean = np.mean(X, axis=0)
            self.train_scores = np.linalg.norm(X - self.mean, axis=1)
            self.threshold = np.percentile(self.train_scores, (1 - contamination) * 100)

        self.fitted = True

        print(f"Threshold set at {self.threshold:.4f}")
        print(
            f"Training score stats: mean={np.mean(self.train_scores):.4f}, "
            f"std={np.std(self.train_scores):.4f}"
        )

    def _mahalanobis_distance(self, X):
        """Compute Mahalanobis distance for samples"""
        diff = X - self.mean
        return np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))

    def score(self, X):
        """
        Compute OOD score for samples

        Args:
            X: Representations to score (n_samples, n_features)

        Returns:
            scores: OOD scores (higher = more out-of-distribution)
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before scoring")

        if self.method == "mahalanobis":
            return self._mahalanobis_distance(X)
        elif self.method == "euclidean":
            return np.linalg.norm(X - self.mean, axis=1)

    def predict(self, X):
        """
        Predict whether samples are OOD

        Args:
            X: Representations (n_samples, n_features)

        Returns:
            is_ood: Boolean array (True = OOD, False = in-distribution)
        """
        scores = self.score(X)
        return scores > self.threshold

    def save(self, path):
        """Save detector to file"""
        state = {
            "method": self.method,
            "mean": self.mean,
            "cov_inv": self.cov_inv,
            "threshold": self.threshold,
            "train_scores": self.train_scores,
            "fitted": self.fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Detector saved to {path}")

    @classmethod
    def load(cls, path):
        """Load detector from file"""
        with open(path, "rb") as f:
            state = pickle.load(f)

        detector = cls(method=state["method"])
        detector.mean = state["mean"]
        detector.cov_inv = state["cov_inv"]
        detector.threshold = state["threshold"]
        detector.train_scores = state["train_scores"]
        detector.fitted = state["fitted"]

        print(f"Detector loaded from {path}")
        return detector


def fit_ood_detector(layer_file, save_path, contamination=0.05, method="mahalanobis"):
    """Fit and save an OOD detector for a specific layer"""

    print(f"Loading representations from {layer_file}...")
    X = np.load(layer_file)

    detector = OODDetector(method=method)
    detector.fit(X, contamination=contamination)
    detector.save(save_path)

    return detector


if __name__ == "__main__":
    # Read best layer
    with open("analysis/best_layer.txt", "r") as f:
        best_layer = f.read().strip()

    print(f"Fitting OOD detector for {best_layer}...")

    layer_file = f"representations/{best_layer}.npy"
    save_path = f"ood_detector_{best_layer}.pkl"

    detector = fit_ood_detector(
        layer_file=layer_file,
        save_path=save_path,
        contamination=0.05,
        method="mahalanobis",
    )

    print("\nOOD detector ready!")
    print(f"To use: detector = OODDetector.load('{save_path}')")
