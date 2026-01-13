"""
Standardized file paths for extract layers.
"""

import os
from constants.data_constants import SCRATCH_FILEPATH


def get_extract_layers_file_path(
    dataset_name: str,
    pooling_function_name: str,
    layer_idx: int,
) -> str:
    os.makedirs(
        get_extract_layers_dir(dataset_name, pooling_function_name), exist_ok=True
    )
    return os.path.join(
        get_extract_layers_dir(dataset_name, pooling_function_name),
        f"layer_{layer_idx}.npy",
    )


def get_extract_layers_dir(
    dataset_name: str,
    pooling_function_name: str,
) -> str:
    return os.path.join(SCRATCH_FILEPATH, dataset_name, pooling_function_name)
