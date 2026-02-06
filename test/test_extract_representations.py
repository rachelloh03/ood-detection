import os
from src.constants.file_format import get_extract_layers_file_path
from src.extract_layers.extract_layers_main import extract_representations
from src.constants.model_constants import JORDAN_MODEL_NAME, DEVICE
from src.constants.data_constants import (
    JORDAN_DATASET_FILEPATH,
)
from src.data.jordan_dataset import JordanDataset
from src.utils.data_loading import collate_fn
from src.extract_layers.pooling_functions import pool_mean_std
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader

NUM_SAMPLES_LIMIT = 10000
BATCH_SIZE = 8


def main(layers_to_extract: list[int]):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        JORDAN_MODEL_NAME,
        torch_dtype=torch.float32,
    ).to(DEVICE)

    n_layers = (
        len(model.transformer.h)
        if hasattr(model, "transformer")
        else len(model.model.layers)
    )

    print(f"Model has {n_layers} layers, extracting from layers: {layers_to_extract}")

    print("Loading dataset...")

    dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="train",
        name="id_train_dataset",
        num_samples=NUM_SAMPLES_LIMIT,
        split_input_and_output_ids=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    pooling_function = pool_mean_std

    return extract_representations(
        model,
        dataloader,
        pooling_function,
        layers_to_extract,
    )


def test_extract_representations():
    layers_to_extract = [2, 3]
    test_buffers = main(layers_to_extract)
    file_path = get_extract_layers_file_path(
        "id_train_dataset",
        "pool_mean_std",
        layers_to_extract[0],
    )
    assert os.path.exists(file_path)

    dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="train",
        name="id_train_dataset",
        num_samples=NUM_SAMPLES_LIMIT,
        split_input_and_output_ids=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = AutoModelForCausalLM.from_pretrained(
        JORDAN_MODEL_NAME,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    pooling_function = pool_mean_std
    buffers = extract_representations(
        model,
        dataloader,
        pooling_function,
        layers_to_extract,
    )
    assert buffers is not None
    assert set(buffers.keys()) == set(
        layers_to_extract
    ), "Buffers keys do not match layers to extract"
    assert np.allclose(buffers[2], test_buffers[2]), "Buffers do not match"
    assert np.allclose(buffers[3], test_buffers[3]), "Buffers do not match"


if __name__ == "__main__":
    test_extract_representations()
