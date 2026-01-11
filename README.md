# OOD Detection for JordanAI Disklavier Model

Out-of-distribution detection system for the JordanAI music generation model.

## Things I did/thoughts (1/10):
1. Added some preprocessing steps to jordan dataset. The Jordan dataset uses two instruments (0 and 1) and possibly starts with the AAR token while the current OOD Maestro dataset only uses instrument 0 and starts with the AR token. There is a preprocessing step that is already added to the jordan_dataset to make the instrument only 0.
- After this, the AUROC seems more reasonable for middle layers. layer 12 -> 0.7ish
2. Made the real time detection thing. I don't have a MIDI cable so I haven't tested the whole pipeline.
3. The distributions of hidden layers are very un-Gaussian, all the p-values are <0.001 (data_analysis/main.ipynb). Probably need to do some density estimation.
- to me a first step to just check things out (haven't done it) is to take a random vector v and do $v \cdot x$ for each sample $x$ and then plot the 1D values to see where they cluster. I'm not sure how effective visualization tools like tSNE will be on this (since it distorts the data), but maybe it allows us to see the clusters?
- more formal methods: empirical KL divergence, maximum mean discrepancy (MMD)
4. Implementation wise to make things way less confusing it may be better to get the OOD detector to also extract layers. what do you think? Plan on how to do this:
- let the extract_representations function take in either a DataLoader or a tensor.
- when it's run for DataLoader it saves the output to a unique file name that somehow contains which dataloader it is like a cache. when it's re-run it checks if the filepath exists and doesn't run it again if it exists.
- when it's run for tensor it just passes it through the model as usual.

## Rachel thoughts (1/10):
1. awesome! this makes sense to map everything to instrument 0 only.
2. I noticed that the real time detection is only being run once after the entire prompt is generated (aka pedal is released). seems to me like that's no longer real-time because we actually want to see if the prompt is OOD before the musician finishes playing. otherwise, there isn't really a point in detecting OOD because they will just hear the "bad" generated output.
- so I think we need to edit this to use sliding window and call ood detection multiple times (perhaps every few midi notes received) while the musician is still playing.
- just to confirm, to run ood detection, the prompt does not need to be tokenized right? we can just directly pass the midi into that helper function for real time detection?
- is convert.py related to the real-time stuff? what did you use it for?
- did you arbitrarily choose 12 as the layer index?  just cuz it's like a middle layer?
3. beause the hidden layers are un-Gaussian, does this mean we will need to change the evaluation metrics? right now, I know we give a higher score if the distribution is more gaussian-like. so instead, we will find the new distribution and then give a higher score based on whether it fits that distribution. is my understanding correct?
- also, visualizing the data seems like a good first step
4. can you clarify your plan? also why is extracting layers and then running OOD detector confusing at the moment? i'm all for the plan if it'll make things less confusing, just not exactly sure what you mean.
5. what are special, non-anticipated, and anticipated tokens?

## Overview

There are two main functionalities to this system:
1. Data analysis. Extracts hidden layer representations from the model on training data, and analyze the distribution properties for OOD detection.
2. OOD detection. Evaluates an OOD detection model (which can be customized) using OOD data.

## Documentation
1. [Installation process](docs/installation.md)
2. [Extract Layers](docs/extract_layers.md) for documentation on extract_layers/.
3. [Data Analysis](docs/data_analysis.md) for documentation on data_analysis/.
4. [OOD Detection](docs/ood_detection.md) for documentation on all other folders.


## File Structure

```
ood-detection/src/
├── extract_layers/            # All methods for extracting representations
├── data_analysis/             # All methods for analyzing distributions
├── main/                      # All methods relating to the OOD detector.
├── real_time_detection/       # Real-time OOD detector.
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



