"""Test script for JordanDataset."""

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
        sample = train_dataset[0]
        print(f"Keys: {list(sample.keys())}")
        print(f"input_ids type: {type(sample['input_ids'])}")
        print(f"input_ids length: {len(sample['input_ids'])}")
        print(f"First 100 tokens: {sample['input_ids'][:100]}")
        print(f"labels type: {type(sample['labels'])}")
        print(f"labels length: {len(sample['labels'])}")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {JORDAN_DATASET_FILEPATH}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_jordan_dataset()
