"""
Model-related constants for the OOD detection project.
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Model name
JORDAN_MODEL_NAME = "mitmedialab/JordanAI-disklavier-v0.1-pytorch"
