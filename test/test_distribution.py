"""
Test the multivariate normality tests.
"""

from src.data_analysis.royston import ROYSTON_P_KEY, royston
from src.data_analysis.hz import HZ_P_VALUE_KEY
import numpy as np
from src.data_analysis.hz import hz
from src.data_analysis.mardia import (
    mardia,
    SKEWNESS_KEY,
    KURTOSIS_KEY,
    SKEWNESS_P_KEY,
    KURTOSIS_P_KEY,
)


def test_mardia():
    """Test the mardia function."""
    np.random.seed(42)
    X_nearly_normal = np.concatenate(
        [np.zeros((1000, 10)), np.ones((1000, 10))], axis=0
    ) + 100 * np.random.randn(2000, 10)
    assert X_nearly_normal.shape[0] == 2000
    assert X_nearly_normal.shape[1] == 10
    metrics = mardia(X_nearly_normal)
    print("metrics", metrics)
    assert metrics[SKEWNESS_KEY] is not None
    assert metrics[SKEWNESS_P_KEY] > 0.05
    assert metrics[KURTOSIS_KEY] is not None
    assert metrics[KURTOSIS_P_KEY] > 0.05

    np.random.seed(42)
    X_not_normal = np.concatenate(
        [np.zeros((1000, 10)), np.ones((1000, 10))], axis=0
    ) + 0.001 * np.random.randn(2000, 10)
    assert X_not_normal.shape[0] == 2000
    assert X_not_normal.shape[1] == 10
    metrics = mardia(X_not_normal)
    print("metrics", metrics)
    assert metrics[SKEWNESS_KEY] is not None
    assert metrics[KURTOSIS_KEY] is not None
    assert metrics[KURTOSIS_P_KEY] < 0.05


def test_hz():
    """Test the hz function."""
    np.random.seed(42)
    X_nearly_normal = np.concatenate(
        [np.zeros((500, 20)), np.ones((500, 20))], axis=0
    ) + 100 * np.random.randn(1000, 20)
    assert X_nearly_normal.shape[0] == 1000
    assert X_nearly_normal.shape[1] == 20
    metrics = hz(X_nearly_normal)
    print("metrics", metrics)
    assert metrics[HZ_P_VALUE_KEY] is not None
    assert metrics[HZ_P_VALUE_KEY] > 0.05

    np.random.seed(42)
    X_not_normal = np.concatenate(
        [np.zeros((500, 20)), np.ones((500, 20))], axis=0
    ) + 0.001 * np.random.randn(1000, 20)
    assert X_not_normal.shape[0] == 1000
    assert X_not_normal.shape[1] == 20
    metrics = hz(X_not_normal)
    print("metrics", metrics)
    assert metrics[HZ_P_VALUE_KEY] is not None
    assert metrics[HZ_P_VALUE_KEY] < 0.05


def test_royston():
    """Test the royston function."""
    np.random.seed(42)
    X_nearly_normal = np.concatenate(
        [np.zeros((500, 20)), np.ones((500, 20))], axis=0
    ) + 100 * np.random.randn(1000, 20)
    assert X_nearly_normal.shape[0] == 1000
    assert X_nearly_normal.shape[1] == 20
    metrics = royston(X_nearly_normal)
    print("metrics", metrics)
    assert metrics[ROYSTON_P_KEY] is not None
    assert metrics[ROYSTON_P_KEY] > 0.05

    np.random.seed(42)
    X_not_normal = np.concatenate(
        [np.zeros((500, 20)), np.ones((500, 20))], axis=0
    ) + 0.001 * np.random.randn(1000, 20)
    assert X_not_normal.shape[0] == 1000
    assert X_not_normal.shape[1] == 20
    metrics = royston(X_not_normal)
    print("metrics", metrics)
    assert metrics[ROYSTON_P_KEY] is not None
    assert metrics[ROYSTON_P_KEY] < 0.05


if __name__ == "__main__":
    test_mardia()
    test_hz()
    test_royston()
