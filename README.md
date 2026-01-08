# OOD Detection for JordanAI Disklavier Model

Out-of-distribution detection system for the JordanAI music generation model.

## Overview

This system:
1. Extracts hidden layer representations from your model on training data
2. Analyzes which layer has the best distribution properties for OOD detection (based on higher gaussianity, lower variance coefficient of variation, and higher eigenvalue entropy)
3. Fits a statistical model (Mahalanobis distance) on that layer
4. Detects when new prompts are out-of-distribution

## Installation

1. Install python dependencies, on a new conda environment if necessary.
```bash
pip install -r requirements.txt
```
Note: if you wish to do statistical tests using the R-Python bridge (rpy2), the python version in the environment must be <=3.11.
If python=3.11, you can install rpy2 version 3.6+.

2. If needed: huggingface setup.
- Set up huggingface token to allow downloading of model/dataset.
- Export token to HF_TOKEN (in ~/.bashrc):
```
export HF_TOKEN={your_token}
```

3. In src/constants.py, specify the filepath to store the dataset
```
if USER == "joeltjy1":
    JORDAN_DATASET_FILEPATH = "/scratch/joel/jordan_dataset"
elif USER == {your_username}:
    JORDAN_DATASET_FILEPATH = {your_filepath}
```

4. Run setup script to download dataset. Right now the setup script is just to download but if other setup needs to be done we can add it in there?
```
python -m src.setup
```

5. Set up proper multivariate normality testing (Mardia, Henze-Zirkler, Royston tests). To do this, install R integration:

```bash
# On hairesmobile (this should be sufficient):
conda install -c conda-forge gsl nlopt r-nloptr r-energy r-lme4 r-car r-mvn r-base

# Install rpy2 for Python-R bridge:
pip install rpy2
```

<!-- I tried doing the installation via this method and it doesn't work for me: install the MVN package in R:
```r
# Start R
R

# Inside R console:
install.packages("MVN")
quit()
``` -->


**Note**: Without R/MVN, the code will fall back to univariate Shapiro-Wilk tests, which work but are less rigorous for multivariate data.

## File Structure

```
ood-detection/src/
├── extract_layers/            # All methods for extracting representations
├── analyze_distribution/      # All methods for analyzing distributions
├── ood_detector.py            # Fit detector
├── detect_ood.py              # Detection pipeline
├── eval/                      # Evaluation methods to test how good the OOD detector is
├── data/                      # Datasets for both inference and eval. OOD dataset can be Maestro/Lakh?
├── utils/                     # Project-wide utilities
├── representations/           # Cached layer representations
│   └── layer_*.npy
├── analysis/                  # Analysis results
│   ├── layer_metrics.json
│   ├── layer_comparison.png
│   └── best_layer.txt
└── ood_detector_layer_X.pkl   # Trained detector
```

## Usage

### Step 1: Extract Hidden Layer Representations

```bash
python -m extract_layers.extract_layers_main
```

This will:
- Load your model from HuggingFace
- Process all training data
- Extract pooled representations (mean + std) from each layer
- Save to `representations/layer_X.npy`

Currently, the pooling is done along the sequence dimension: given the hidden layer activations of shape 
(N=number of samples, L=sequence length, D=hidden dim), the pooled result is (N, 2D) = (N, D) for mean + (N, D) for std.

If this is not effective, then there will be other pooling methods coded. 

Source code for this step is in the folder extract_layers/

### Step 2: Analyze Layer Distributions

```bash
python analyze_distributions.py
```

This will:
- Analyze all layers for distribution quality
- Generate comparison plots
- Recommend the best layer for OOD detection
- Save results to `analysis/`

Metrics evaluated:
- **Multivariate Gaussianity** (Mardia/Henze-Zirkler/Royston tests): More Gaussian = better for Mahalanobis distance
  - Falls back to univariate Shapiro-Wilk if R/MVN not available
- **Variance balance**: More uniform feature variances = better
- **Eigenvalue entropy**: Higher = better distributed information
- **Effective dimensionality**: Intrinsic dimensionality of representations

#### Notes on the multivariate tests:
- In the MVN tests there seems to be some hidden state that carries over multiple runs, that if not cleared will result in the different
runs interfering with each other. Garbage collection has to be done explicitly after every run:
```
# THIS IS NECESSARY!!! Explicitly clean up R objects to prevent interference between calls
del r_data, hz_result, result_table
ro.r("gc()")  # Force R garbage collection
```
- In test/test_distribution.py I tested the performance of the tests on two dummy example: consider 
```
X = np.concatenate(
    [np.zeros((N, D)), np.ones((N, D))], axis=0
) + alpha * np.random.randn(2*N, D)
```
As alpha increases, this should appear more normal and the p-value should get higher.
Small alpha means the distribution is heavily bimodal.

1. For Mardia, for alpha = 0.001 I tried D=10 and it needed N~1000 for it to detect that it's not normal
2. Henze-Zirkler manages to detect this for alpha=0.001, D=20, N~500. So it seems more powerful, but it's still quite bad at normality detection for D>around 50. Probably need to implement PCA before passing it into the test.
I haven't played around with royston yet.

- I haven't run PCA yet but I hope the number of dimensions for 95% explained variance is on the order of tens.

This took longer than expected and this is where I've explored up to and I still haven't run the OOD detector :( I'll look at the OOD detector and below tomorrow!


### Step 3: Fit OOD Detector

```bash
python ood_detector.py
```

This will:
- Load the best layer's representations
- Fit a Mahalanobis distance model
- Set threshold at 95th percentile (5% contamination)
- Save detector to `ood_detector_layer_X.pkl`

### Step 4: Detect OOD on New Prompts

```python
from detect_ood import OODPipeline

# Initialize
pipeline = OODPipeline(
    model_name="mitmedialab/JordanAI-disklavier-v0.1-pytorch",
    detector_path="ood_detector_layer_X.pkl",
    layer_idx=X  # from best_layer.txt
)

# Check if input is OOD
result = pipeline.check_ood(input_ids, attention_mask)

print(f"Is OOD: {result['is_ood']}")
print(f"Score: {result['score']:.4f}")
print(f"Percentile: {result['percentile']:.1f}%")
```

## How It Works

### Representation Extraction
Each sequence is converted to a fixed-size representation by:
1. Extracting hidden states from a specific layer: `(batch, seq_len, hidden_dim)`
2. Computing mean and std across sequence: `(batch, hidden_dim)` each
3. Concatenating: `(batch, 2 * hidden_dim)`

### OOD Detection Method
Uses **Mahalanobis distance**:
- Measures how far a point is from the training distribution
- Accounts for correlations between features (better than Euclidean)
- Formula: `D(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))`
  - `μ`: mean of training data
  - `Σ`: covariance matrix of training data

### Threshold Setting
- Compute Mahalanobis distances for all training samples
- Set threshold at 95th percentile (assumes 5% outliers in training)
- Samples with distance > threshold are flagged as OOD

## Output Interpretation

```python
{
    'is_ood': True/False,           # Binary OOD decision
    'score': 15.32,                 # Mahalanobis distance
    'threshold': 12.45,             # Decision threshold
    'percentile': 97.3,             # Percentile vs training data
    'z_score': 2.15                 # Standard deviations from mean
}
```

**Rules of thumb:**
- `percentile > 95`: Likely OOD
- `percentile > 99`: Very likely OOD
- `z_score > 3`: Extreme outlier

## Customization

### Change contamination rate
```python
detector.fit(X, contamination=0.10)  # Expect 10% outliers
```

### Use different method
```python
detector = OODDetector(method='euclidean')  # Simpler, faster
```

### Manual layer selection
```python
# Override automatic selection
detector = fit_ood_detector(
    layer_file="representations/layer_15.npy",
    save_path="ood_detector_custom.pkl"
)
```

## Tips

1. **Middle layers often work best** for OOD detection (layers 8-16 in a 24-layer model)
2. **More training data = better detector** - use your full training set
3. **Validate on held-out data** to tune the contamination parameter
4. **Monitor percentiles** - they're more interpretable than raw scores


## Troubleshooting

**Memory issues?**
- Process data in smaller batches in `extract_layers.py`
- Reduce batch size in DataLoader

**Poor OOD detection?**
- Try different layers manually
- Adjust contamination parameter
- Check if training data is clean
- If R/MVN tests are unavailable, install them for better layer selection

**Slow inference?**
- Cache representations for validation sets
- Use simpler `euclidean` method instead of `mahalanobis`
