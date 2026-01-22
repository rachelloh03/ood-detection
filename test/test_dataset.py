"""Test script for JordanDataset."""

import torch
from data.sliding_window import SlidingWindowDataset
from src.data.jordan_dataset import JordanDataset
from src.constants.data_constants import JORDAN_DATASET_FILEPATH


def test_jordan_dataset():
    """Test the JordanDataset class."""
    try:
        train_dataset = JordanDataset(
            data_dir=JORDAN_DATASET_FILEPATH,
            split="train",
            name="testcase_jordan_dataset",
        )
        print("Successfully loaded train dataset")
        print(f"Length: {len(train_dataset)}")

        print("Testing sample access...")
        sample = train_dataset[50]
        print(f"Keys: {list(sample.keys())}")
        print(f"input_ids type: {type(sample['input_ids'])}")
        print(f"input_ids length: {len(sample['input_ids'])}")
        print(f"First 50 tokens: {sample['input_ids'][:50]}")
        print(f"output_ids type: {type(sample['output_ids'])}")
        print(f"output_ids length: {len(sample['output_ids'])}")
        print(f"First 50 output_ids: {sample['output_ids'][:50]}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {JORDAN_DATASET_FILEPATH}")
    except Exception as e:
        print(f"Error: {e}")


def test_sliding_window_dataset():
    """Test the SlidingWindowDataset class."""
    dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="train",
        name="testcase_jordan_dataset",
        num_samples=2,
    )
    k = 12
    stride = 3
    sliding_window_dataset = SlidingWindowDataset(
        base_dataset=dataset,
        name="testcase_sliding_window_dataset",
        k=k,
        stride=stride,
        drop_last=True,
        lm_labels=False,
        first_token_special=True,
    )

    assert torch.equal(
        sliding_window_dataset[0]["input_ids"], dataset[0]["input_ids"][:13]
    )
    assert torch.equal(
        sliding_window_dataset[1]["input_ids"],
        torch.cat([dataset[0]["input_ids"][:1], dataset[0]["input_ids"][4:16]]),
    )
    num_samples_per_sample = (len(dataset[0]["input_ids"]) - k - 1) // stride + 1
    assert torch.equal(
        sliding_window_dataset[num_samples_per_sample - 1]["input_ids"],
        torch.cat([dataset[0]["input_ids"][:1], dataset[0]["input_ids"][-k:]]),
    )
    assert torch.equal(
        sliding_window_dataset[num_samples_per_sample]["input_ids"],
        dataset[1]["input_ids"][: k + 1],
    )


if __name__ == "__main__":
    test_jordan_dataset()
    test_sliding_window_dataset()
