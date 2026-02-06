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
            split_input_and_output_ids=True,
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
        split_input_and_output_ids=True,
        num_samples=2,
    )
    # k and stride are in *events*; TOKENS_PER_EVENT=3 so k=4 -> 12 tokens, stride=1 -> 3 tokens
    k_events = 4
    stride_events = 1
    k_tokens = k_events * 3
    stride_tokens = stride_events * 3
    sliding_window_dataset = SlidingWindowDataset(
        base_dataset=dataset,
        name="testcase_sliding_window_dataset",
        k=k_events,
        stride=stride_events,
        drop_last=True,
        lm_labels=False,
        first_token_special=True,
        max_silence_length=-1,  # disable gap filtering so window indices are deterministic
    )

    assert torch.equal(
        sliding_window_dataset[0]["input_ids"], dataset[0]["input_ids"][: k_tokens + 1]
    )
    assert torch.equal(
        sliding_window_dataset[1]["input_ids"],
        torch.cat(
            [
                dataset[0]["input_ids"][:1],
                dataset[0]["input_ids"][
                    1 + stride_tokens : 1 + stride_tokens + k_tokens
                ],
            ]
        ),
    )
    n = len(dataset[0]["input_ids"])
    num_samples_per_sample = (n - 1 - k_tokens) // stride_tokens + 1
    assert torch.equal(
        sliding_window_dataset[num_samples_per_sample - 1]["input_ids"],
        torch.cat([dataset[0]["input_ids"][:1], dataset[0]["input_ids"][-k_tokens:]]),
    )
    assert torch.equal(
        sliding_window_dataset[num_samples_per_sample]["input_ids"],
        dataset[1]["input_ids"][: k_tokens + 1],
    )


if __name__ == "__main__":
    test_jordan_dataset()
    test_sliding_window_dataset()
