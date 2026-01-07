# OOD Detection for JordanAI Disklavier Model

Out-of-distribution detection system for the JordanAI music generation model.

## Overview

This system:
1. Extracts hidden layer representations from your model on training data
2. Analyzes which layer has the best distribution properties for OOD detection (based on higher gaussianity, lower variance coefficient of variation, and higher eigenvalue entropy)
3. Fits a statistical model (Mahalanobis distance) on that layer
4. Detects when new prompts are out-of-distribution

## Installation

### Python Dependencies
```bash
pip install torch transformers datasets numpy scipy scikit-learn matplotlib tqdm
```

### Optional: Multivariate Normality Tests (Recommended)

For proper multivariate normality testing (Mardia, Henze-Zirkler, Royston tests), install R integration:

```bash
# Install rpy2 for Python-R bridge
pip install rpy2

# Install R (if not already installed)
# On Ubuntu/Debian:
sudo apt-get install r-base

# On macOS with Homebrew:
brew install r

# On Windows: Download from https://cran.r-project.org/
```

Then install the MVN package in R:
```r
# Start R
R

# Inside R console:
install.packages("MVN")
quit()
```

**Note**: Without R/MVN, the code will fall back to univariate Shapiro-Wilk tests, which work but are less rigorous for multivariate data.

## Usage

### Step 1: Extract Hidden Layer Representations

```bash
python extract_layers.py
```

This will:
- Load your model from HuggingFace
- Process all training data
- Extract pooled representations (mean + std) from each layer
- Save to `representations/layer_X.npy`

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

## File Structure

```
ood-detection/
├── extract_layers.py          # Extract representations
├── analyze_distributions.py   # Analyze layers
├── ood_detector.py            # Fit detector
├── detect_ood.py              # Detection pipeline
├── model_utils.py             # Model loading utilities
├── representations/           # Cached layer representations
│   └── layer_*.npy
├── analysis/                  # Analysis results
│   ├── layer_metrics.json
│   ├── layer_comparison.png
│   └── best_layer.txt
└── ood_detector_layer_X.pkl   # Trained detector
```

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
