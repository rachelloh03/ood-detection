"""
Shapiro-Wilk test for _univariate_ normality.

Weakest test, only use if you have to.

Source: https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
"""

from data_analysis.preprocess import subsample
import numpy as np
from scipy import stats

SHAPIRO_P_KEY = "shapiro_p"


def shapiro(X: np.ndarray) -> dict:
    metrics = {}
    X_test = subsample(X, num_features=1)
    shapiro_p = stats.shapiro(X_test[:, 0])
    metrics[SHAPIRO_P_KEY] = float(shapiro_p.pvalue)

    # Explicitly clean up numpy objects to prevent interference between calls
    del X_test, shapiro_p

    return metrics
