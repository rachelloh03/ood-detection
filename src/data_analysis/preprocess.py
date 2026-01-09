"""
Preprocessing steps before passing into multivariate tests.
"""

import numpy as np

MAX_SAMPLES = 5000
MAX_FEATURES = 100


def subsample(
    X: np.ndarray, num_samples: int = MAX_SAMPLES, num_features: int = MAX_FEATURES
) -> np.ndarray:
    """
    Subsample the data to a maximum number of samples and features.
    """
    n_samples_test = min(num_samples, X.shape[0])
    n_features_test = min(num_features, X.shape[1])

    if X.shape[0] > n_samples_test or X.shape[1] > n_features_test:
        print(
            f"  Subsampling to {n_samples_test} samples and {n_features_test} features for testing"
        )
        idx_samples = np.random.choice(X.shape[0], n_samples_test, replace=False)
        idx_features = np.random.choice(X.shape[1], n_features_test, replace=False)
        return X[np.ix_(idx_samples, idx_features)]
    else:
        return X
