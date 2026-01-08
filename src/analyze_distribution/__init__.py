"""
All methods for analyzing distributions.

Contains:
- Mardia test
- Henze-Zirkler test
- Royston test

Preprocessing steps:
- Subsampling
- PCA

We may need more preprocessing steps if we aren't very sure it's Gaussian,
which might be possible because the normality tests are pretty weak
for high-dimensional data and our current number of samples (on the order of 1000).
"""
