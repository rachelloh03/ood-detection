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
from data.sliding_window import SlidingWindowDataset
from constants.data_constants import JORDAN_DATASET_FILEPATH
from constants.model_constants import JORDAN_MODEL_NAME, DEVICE
from constants.real_time_constants import SLIDING_WINDOW_LEN, STRIDE
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from extract_layers.pooling_functions import pool_mean_std
from utils.data_loading import collate_fn

NUM_SAMPLES_LIMIT = 10000
BATCH_SIZE = 100

# Cache for loaded representations to avoid reloading from disk
_representation_cache = {}


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
    chunk_size: int = 10_000,
) -> dict[int, np.ndarray]:
    """
    Extract pooled representations from specified layers using chunked GPU pre-allocation.
    """

    dataset = dataloader.dataset
    dataset_name = dataset.name
    model.eval()

    if debug:
        print(f"Extracting representations from {dataset_name} dataset")
        print(model)

    save_dir = get_extract_layers_dir(dataset_name, pooling_function.__name__)

    cache_key = (dataset_name, pooling_function.__name__, tuple(sorted(layers)))

    if cache_key in _representation_cache:
        return _representation_cache[cache_key]
    if all(
        os.path.exists(
            get_extract_layers_file_path(
                dataset_name, pooling_function.__name__, layer_idx
            )
        )
        for layer_idx in layers
    ):
        print("Representations already exist. Loading from disk.")
        return_dict = {}
        for layer_idx in tqdm(layers, desc="Loading layers from disk"):
            print(f"Loading layer {layer_idx} from disk.")
            return_dict[layer_idx] = np.load(
                get_extract_layers_file_path(
                    dataset_name, pooling_function.__name__, layer_idx
                )
            )
        _representation_cache[cache_key] = return_dict
        return return_dict

    use_amp = DEVICE.startswith("cuda") and torch.cuda.is_available()

    first_batch = next(iter(dataloader))
    first_batch = {k: v.to(DEVICE, non_blocking=True) for k, v in first_batch.items()}

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        outputs = model(
            **first_batch,
            output_hidden_states=True,
            use_cache=False,
        )

    example_pooled = pooling_function(outputs.hidden_states[layers[0]])
    pooled_dim = example_pooled.size(-1)
    dtype = example_pooled.dtype

    gpu_chunk = {
        layer_idx: torch.empty(
            (chunk_size, pooled_dim),
            device=DEVICE,
            dtype=dtype,
        )
        for layer_idx in layers
    }

    temp_files = {layer_idx: [] for layer_idx in layers}
    chunk_offset = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Extracting hidden states"):
        B = next(iter(batch.values())).size(0)

        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states

        for layer_idx in layers:
            pooled = pooling_function(hidden_states[layer_idx])
            gpu_chunk[layer_idx][chunk_offset : chunk_offset + B].copy_(pooled)

        chunk_offset += B
        total_samples += B

        if chunk_offset >= chunk_size:
            for layer_idx in layers:
                chunk_cpu = gpu_chunk[layer_idx][:chunk_offset].cpu().numpy()
                temp_file = os.path.join(
                    save_dir,
                    f"temp_layer_{layer_idx}_chunk_{len(temp_files[layer_idx])}.npy",
                )
                os.makedirs(save_dir, exist_ok=True)
                np.save(temp_file, chunk_cpu)
                temp_files[layer_idx].append(temp_file)
            chunk_offset = 0

    if chunk_offset > 0:
        for layer_idx in layers:
            chunk_cpu = gpu_chunk[layer_idx][:chunk_offset].cpu().numpy()
            temp_file = os.path.join(
                save_dir,
                f"temp_layer_{layer_idx}_chunk_{len(temp_files[layer_idx])}.npy",
            )
            np.save(temp_file, chunk_cpu)
            temp_files[layer_idx].append(temp_file)

    buffers: dict[int, np.ndarray] = {}

    for layer_idx in layers:
        if len(temp_files[layer_idx]) == 1:
            X = np.load(temp_files[layer_idx][0])
            final_path = get_extract_layers_file_path(
                dataset_name, pooling_function.__name__, layer_idx
            )
            np.save(final_path, X)
            os.remove(temp_files[layer_idx][0])
        else:
            chunks = []
            for temp_file in temp_files[layer_idx]:
                chunks.append(np.load(temp_file))
            X = np.concatenate(chunks, axis=0)

            if save_dir is not None:
                final_path = get_extract_layers_file_path(
                    dataset_name, pooling_function.__name__, layer_idx
                )
                np.save(final_path, X)
                if debug:
                    print(f"Saved layer {layer_idx}: {X.shape}")

            for temp_file in temp_files[layer_idx]:
                os.remove(temp_file)

        buffers[layer_idx] = X

    # Cache the result
    _representation_cache[cache_key] = buffers
    return buffers


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

    base_dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="train",
        name="id_train_dataset",
    )

    dataset = SlidingWindowDataset(
        base_dataset=base_dataset,
        name="id_train_dataset",
        k=SLIDING_WINDOW_LEN,
        stride=STRIDE,
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
