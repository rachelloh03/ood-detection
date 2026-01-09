"""
Royston test for multivariate normality.

Source: https://en.wikipedia.org/wiki/Royston_test
"""

from data_analysis.preprocess import subsample
from rpy2.robjects.packages import importr
from rpy2.robjects import ListVector, numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import numpy as np

mvn = importr("MVN")
ROYSTON_P_KEY = "royston_p"


def royston(X: np.ndarray) -> dict:
    metrics = {}
    X_test = subsample(X, num_samples=1000, num_features=100)

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_data = X_test  # auto-converts to R matrix

        royston_result = mvn.mvn(r_data, mvn_test="royston")
        royston_result = ListVector(royston_result)
        result_table = royston_result.rx2("multivariate_normality")
        p_value = result_table["p.value"][0]
        if p_value == "<0.001":
            p_value = 0.001
        try:
            p_value = float(p_value)
        except ValueError:
            print("p_value is not a float")
            p_value = None
        metrics[ROYSTON_P_KEY] = p_value

        # THIS IS NECESSARY!!! Explicitly clean up R objects to prevent interference between calls
        del r_data, royston_result, result_table
        ro.r("gc()")  # Force R garbage collection

        return metrics
