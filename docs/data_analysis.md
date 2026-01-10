# Data Analysis

Back to home: [README](../README.md)

TODO: clean up this doc

An example of usage in ```data_analysis/main.ipynb```

OOD detection (based on higher gaussianity, lower variance coefficient of variation, and higher eigenvalue entropy)

## Note on datasets
The Jordan dataset uses two instruments (0 and 1) and possibly starts with the AAR token while the current OOD Maestro dataset only uses instrument 0 and starts with the AR token. There is a preprocessing step that is already added to the jordan_dataset to make the instrument only 0.

## On multivariate normality tests
1. In the MVN tests there is a hidden state that carries over multiple runs, that if not cleared will result in the different runs interfering with each other. Garbage collection has to be done explicitly after every run:
```
# THIS IS NECESSARY!!! Explicitly clean up R objects to prevent interference between calls
del r_data, hz_result, result_table
ro.r("gc()")  # Force R garbage collection
```
2. In test/test_distribution.py the performance of the tests is tested on a dummy example: consider 
```
X = np.concatenate(
    [np.zeros((N, D)), np.ones((N, D))], axis=0
) + alpha * np.random.randn(2*N, D)
```
As alpha increases, this should appear more normal and the p-value should get higher.
Small alpha means the distribution is heavily bimodal.

- For Mardia, for alpha = 0.001 I tried D=10 and it needed N~1000 for it to detect that it's not normal
- Henze-Zirkler manages to detect this for alpha=0.001, D=20, N~500. So it seems more powerful, but it's still quite bad at normality detection for D>around 50. Probably need to implement PCA before passing it into the test.
