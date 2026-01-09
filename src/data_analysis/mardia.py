"""
Mardia test for multivariate normality.

Computes multivariate extensions of the skewness and kurtosis statistics
and compares it against the null hypothesis of multivariate normality.

Mardia's test potentially causes issues if the number of samples is too small.

Source: https://en.wikipedia.org/wiki/Multivariate_normality#Tests_for_multivariate_normality
"""

from analyze_distribution.preprocess import subsample
from rpy2.robjects.packages import importr
from rpy2.robjects import ListVector, numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import numpy as np

mvn = importr("MVN")

SKEWNESS_KEY = "skewness"
KURTOSIS_KEY = "kurtosis"
SKEWNESS_P_KEY = "skewness_p"
KURTOSIS_P_KEY = "kurtosis_p"


def mardia(X: np.ndarray) -> dict:
    metrics = {}
    X_test = subsample(X)

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_data = X_test  # auto-converts to R matrix

        mardia_result = mvn.mvn(r_data, mvn_test="mardia")
        mardia_result = ListVector(mardia_result)
        result_table = mardia_result.rx2("multivariate_normality")

        metrics[SKEWNESS_KEY] = result_table[0][1]
        metrics[KURTOSIS_KEY] = result_table[1][1]
        skewness_p = result_table[0][2]
        if skewness_p == "<0.001":
            skewness_p = 0.001
        try:
            skewness_p = float(skewness_p)
        except ValueError:
            print("skewness_p is not a float")
            skewness_p = None
        metrics[SKEWNESS_P_KEY] = skewness_p
        kurtosis_p = result_table[1][2]
        if kurtosis_p == "<0.001":
            kurtosis_p = 0.001
        try:
            kurtosis_p = float(kurtosis_p)
        except ValueError:
            print("kurtosis_p is not a float")
            kurtosis_p = None
        metrics[KURTOSIS_P_KEY] = kurtosis_p

        # THIS IS NECESSARY!!! Explicitly clean up R objects to prevent interference between calls
        del r_data, mardia_result, result_table
        ro.r("gc()")  # Force R garbage collection

        return metrics
