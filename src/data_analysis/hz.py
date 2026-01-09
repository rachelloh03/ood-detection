"""
Henze-Zirkler test for multivariate normality.

Also called BHEP test in the wiki page on multivariate normality tests.
"""

from data_analysis.preprocess import subsample
from rpy2.robjects.packages import importr
from rpy2.robjects import ListVector, numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

mvn = importr("MVN")
HZ_P_VALUE_KEY = "hz_p_value"


def hz(X):
    metrics = {}
    X_test = subsample(X)

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_data = X_test  # auto-converts to R matrix

        hz_result = mvn.mvn(r_data, mvn_test="hz")
        hz_result = ListVector(hz_result)
        result_table = hz_result.rx2("multivariate_normality")
        p_value = result_table["p.value"][0]
        print("p_value", p_value)
        if p_value == "<0.001":
            p_value = 0.001
        try:
            p_value = float(p_value)
        except ValueError:
            print("p_value is not a float")
            p_value = None
        metrics[HZ_P_VALUE_KEY] = p_value

    # THIS IS NECESSARY!!! Explicitly clean up R objects to prevent interference between calls
    del r_data, hz_result, result_table
    ro.r("gc()")  # Force R garbage collection

    return metrics
