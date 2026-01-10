# OOD Detection for JordanAI Disklavier Model

Out-of-distribution detection system for the JordanAI music generation model.

## Things I did/thoughts(1/10):
- Added some preprocessing steps to jordan dataset. The Jordan dataset uses two instruments (0 and 1) and possibly starts with the AAR token while the current OOD Maestro dataset only uses instrument 0 and starts with the AR token. There is a preprocessing step that is already added to the jordan_dataset to make the instrument only 0.
- After this, the AUROC is more reasonable for middle layers. layer 12 -> 0.7ish
- Made the real time detection thing. I don't have a MIDI cable so I haven't tested the whole pipeline, but I tested the software side of things.
- The distributions of hidden layers are very un-Gaussian. Probably need to do some density estimation.


## Overview

There are two main functionalities to this system:
1. Data analysis. Extracts hidden layer representations from the model on training data, and analyze the distribution properties for OOD detection.
2. OOD detection. Evaluates an OOD detection model (which can be customized) using OOD data.

## Documentation
1. [Installation process](docs/installation.md)
2. [Data Analysis](docs/data_analysis.md)
3. [OOD Detection](docs/ood_detection.md)
4. [Extract Layers](docs/extract_layers.md)

## File Structure

```
ood-detection/src/
├── extract_layers/            # All methods for extracting representations
├── data_analysis/             # All methods for analyzing distributions
├── main/                      # All methods relating to the OOD detector.
├── eval/                      # Evaluation methods to test how good the OOD detector is
├── data/                      # Datasets for both inference and eval. OOD dataset will be Maestro.
├── utils/                     # Project-wide utilities
├── representations/           # Cached layer representations
│   └── layer_*.npy
└── analysis/                  # Analysis results
    ├── layer_metrics.json
    ├── layer_comparison.png
    └── best_layer.txt
```
For documentation on extract_layers/

## Usage (to be changed)

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


- I haven't run PCA yet but I hope the number of dimensions for 95% explained variance is on the order of tens.



