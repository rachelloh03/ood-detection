"""Universal project constants."""

import os
import torch

USER = os.environ.get("USER")
if USER is None:
    raise ValueError("USER environment variable is not set")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset constants
JORDAN_DATASET_NAME = (
    "mitmedialab/"
    + "jordan_rudess__disklavier__trading"
    + "_inst0_inst1__free_time__pedal__velocity__v1"
)

if USER == "joeltjy1":
    JORDAN_DATASET_FILEPATH = "/scratch/joel/jordan_dataset"
elif USER == "rjloh":
    raise ValueError(f"USER {USER} not supported")
else:
    raise ValueError(f"USER {USER} not supported")
os.makedirs(JORDAN_DATASET_FILEPATH, exist_ok=True)

TRAIN_FILENAME = "data-00000-of-00001.arrow"
TEST_FILENAME = "data-00000-of-00001.arrow"

# Model constants
JORDAN_MODEL_NAME = "mitmedialab/JordanAI-disklavier-v0.1-pytorch"

# Token constants
REST = 27512
SEP = 55025
AR = 55026
AAR = 55027

# not anticipated
TIME_OFFSET = 0
DUR_OFFSET = 10000
NOTE_OFFSET = 11000

# anticipated
ATIME_OFFSET = 27513
ADUR_OFFSET = 37513
ANOTE_OFFSET = 38513

# velocity
VELOCITY_OFFSET = 55028

# notes
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
