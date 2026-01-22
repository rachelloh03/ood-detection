"""
Data and file path constants for the OOD detection project.
"""

import os

# User and device
USER = os.environ.get("USER")
if USER is None:
    raise ValueError("USER environment variable is not set")

# File paths
SCRATCH_FILEPATH = "representations"
OOD_DATASET_FILEPATH = None  # Default, may be set per user
MAESTRO_DATASET_FILEPATH = None  # Default, may be set per user

if USER == "joeltjy1":
    JORDAN_DATASET_FILEPATH = "/scratch/joel/jordan_dataset"
    MAESTRO_DATASET_FILEPATH = "/scratch/joel/maestrodata"
    SCRATCH_FILEPATH = "/scratch/joel/representations"
    SOUNDFONT_FILEPATH = "/data/scratch/joel/soundfont.sf2"
elif USER == "rachelloh":
    JORDAN_DATASET_FILEPATH = "/Users/rachelloh/Desktop/ood-detection/jordan_dataset"
    SCRATCH_FILEPATH = "/Users/rachelloh/Desktop/ood-detection/representations"
    SOUNDFONT_FILEPATH = "/Users/rachelloh/Desktop/ood-detection/soundfont.sf2"
elif USER == "rjloh":
    JORDAN_DATASET_FILEPATH = "/scratch/rjloh/jordan_dataset"
    OOD_DATASET_FILEPATH = "/scratch/rjloh/ood_dataset"
    SCRATCH_FILEPATH = "/scratch/rjloh/representations"
    SOUNDFONT_FILEPATH = "/data/scratch/rjloh/soundfont.sf2"
else:
    raise ValueError(f"USER {USER} not supported")

# Create directories
os.makedirs(JORDAN_DATASET_FILEPATH, exist_ok=True)
if USER == "joeltjy1":
    os.makedirs(MAESTRO_DATASET_FILEPATH, exist_ok=True)
os.makedirs(SCRATCH_FILEPATH, exist_ok=True)

# Dataset constants
JORDAN_DATASET_NAME = (
    "mitmedialab/"
    + "jordan_rudess__disklavier__trading"
    + "_inst0_inst1__free_time__pedal__velocity__v1"
)
