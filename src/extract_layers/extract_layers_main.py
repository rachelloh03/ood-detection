"""
Script to extract hidden layer activations from the JordanAI model.

Iterates through the dataset and each layer, and saves the representations.

If there are N transformer blocks, there are N + 1 layers to extract from:
hidden_states[0] is the embeddings.
hidden_states[i] is the output of the i-th transformer block.
"""

import os
from typing import Callable
from constants.file_format import get_extract_layers_file_path, get_extract_layers_dir
from data.jordan_dataset import JordanDataset
from constants.data_constants import JORDAN_DATASET_FILEPATH
from constants.model_constants import JORDAN_MODEL_NAME
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from extract_layers.pooling_functions import pool_mean_std
from utils.data_loading import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES_LIMIT = 10000
BATCH_SIZE = 8


def extract_representations(
    model: torch.nn.Module,
    data: DataLoader | torch.Tensor,
    pooling_function: Callable[[torch.Tensor], torch.Tensor],
    layers: list[int],
) -> dict[int, list[torch.Tensor]]:
    """
    Extract pooled representations from specified layers.
    """
    if isinstance(data, DataLoader):
        return extract_representations_from_dataset(
            model, data, pooling_function, None, layers
        )
    elif isinstance(data, torch.Tensor):
        return extract_representations_from_tensor(
            model, data, pooling_function, layers
        )
    else:
        raise ValueError(
            f"Invalid data type: {type(data)}, expected DataLoader or torch.Tensor"
        )


@torch.no_grad()
def extract_representations_from_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    pooling_function: Callable[[torch.Tensor], torch.Tensor],
    save_dir: str,
    layers: list[int],
    debug: bool = False,
) -> dict[int, list[torch.Tensor]]:
    """
    Extract pooled representations from specified layers.

    Args:
        model: The model to extract representations from.
        dataloader: The dataloader to extract representations from.
        pooling_function: The function to pool the representations. Currently supported option:
        - "mean_std": Pool using mean and std across sequence dimension, (B, L, D) -> (B, 2*D).
        Not sure if will lose too much info.
        save_dir: The directory to save the representations.
        layers: The layers to extract representations from.
    """
    dataset_name = dataloader.dataset.name
    model.eval()

    if debug:
        print(f"Extracting representations from {dataset_name} dataset")
        print(model)

    buffers = {layer_idx: [] for layer_idx in layers}

    save_dir = get_extract_layers_dir(dataset_name, pooling_function.__name__)
    if all(
        os.path.exists(
            get_extract_layers_file_path(
                dataset_name, pooling_function.__name__, layer_idx
            )
        )
        for layer_idx in layers
    ):
        if debug:
            print(
                f"Representations already exist for {dataset_name} dataset with {pooling_function.__name__}"
                f"pooling function for layers: {layers}. Loading from {save_dir}."
            )
        return {
            layer_idx: np.load(
                get_extract_layers_file_path(
                    dataset_name, pooling_function.__name__, layer_idx
                )
            )
            for layer_idx in layers
        }

    for batch in tqdm(dataloader, desc="Extracting hidden states"):
        batch = {
            k: v.to(DEVICE) for k, v in batch.items()
        }  # (B, L) where L ~ 766 is the sequence length

        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states  # (n_layers + 1)-tuple of (B, L, D)

        for layer_idx in layers:
            h = hidden_states[layer_idx]  # (B, L, D)
            pooled = pooling_function(h)
            buffers[layer_idx].append(pooled.cpu())

    for layer_idx in layers:
        X = torch.cat(buffers[layer_idx], dim=0).numpy()  # (dataset size, 2*D)
        buffers[layer_idx] = X  # (dataset size, 2*D)
        if save_dir is not None:
            np.save(
                get_extract_layers_file_path(
                    dataset_name, pooling_function.__name__, layer_idx
                ),
                X,
            )
            if debug:
                print(f"Saved layer {layer_idx}: {X.shape}")

    return buffers  # {layer_idx: (dataset size, 2*D)}


@torch.no_grad()
def extract_representations_from_tensor(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    pooling_function: Callable[[torch.Tensor], torch.Tensor],
    layers: list[int],
) -> dict[int, list[torch.Tensor]]:
    """
    Extract pooled representations from specified layers.

    Args:
        model: The model to extract representations from.
        tensor: The tensor to extract representations from. (B, L)
        pooling_function: The function to pool the representations. Currently supported option:
        - "mean_std": Pool using mean and std across sequence dimension, (B, L, D) -> (B, 2*D).
        Not sure if will lose too much info.
        layers: The layers to extract representations from.
    """
    model.eval()
    buffers = {layer_idx: [] for layer_idx in layers}

    outputs = model(
        tensor,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states  # (n_layers + 1)-tuple of (B, L, D)

    for layer_idx in layers:
        h = hidden_states[layer_idx]  # (B, L, D)
        pooled = pooling_function(h)
        buffers[layer_idx] = pooled.cpu()

    return buffers  # {layer_idx: (B, 2*D)}


def example_extract_representations():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        JORDAN_MODEL_NAME,
        dtype=torch.float32,
    ).to(DEVICE)

    n_layers = (
        len(model.transformer.h)
        if hasattr(model, "transformer")
        else len(model.model.layers)
    )
    layers_to_extract = list(range(n_layers + 1))

    print(f"Model has {n_layers} layers, extracting from layers: {layers_to_extract}")

    print("Loading dataset...")

    dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="train",
        name="id_train_dataset",
        num_samples=NUM_SAMPLES_LIMIT,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    pooling_function = pool_mean_std

    extract_representations(
        model,
        dataloader,
        pooling_function,
        layers_to_extract,
    )


if __name__ == "__main__":
    example_extract_representations()
