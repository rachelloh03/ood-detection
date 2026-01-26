"""
Model-related constants for the OOD detection project.
"""

import torch

# Use the first visible CUDA device if available; otherwise fall back to CPU.
# This respects CUDA_VISIBLE_DEVICES set by SLURM.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Model name
JORDAN_MODEL_NAME = "mitmedialab/JordanAI-disklavier-v0.1-pytorch"

# for anticipation if needed
DELTA = 5
