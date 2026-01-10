"""
Setup script for the OOD detection system.

The aim is for all menial setup tasks to be done here.

For now, only downloads the Jordan dataset.
"""

from src.constants.data_constants import JORDAN_DATASET_FILEPATH, JORDAN_DATASET_NAME
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
import os


def download_jordan_dataset():
    """Download the Jordan dataset."""
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    print(f"Loading dataset {JORDAN_DATASET_NAME}...")
    dataset = load_dataset(JORDAN_DATASET_NAME, token=token)
    print(f"Saving dataset to {JORDAN_DATASET_FILEPATH}...")
    dataset.save_to_disk(JORDAN_DATASET_FILEPATH)
    print("Dataset downloaded and saved successfully!")


def test_dataset_retrieval():
    """Test retrieving the train/test datasets."""
    print(f"Loading dataset from {JORDAN_DATASET_FILEPATH}...")
    try:
        dataset = load_from_disk(JORDAN_DATASET_FILEPATH)
        if "train" in dataset:
            print("\nTesting retrieving first row of training dataset:")
            train_sample = dataset["train"][0]
            print(f"Keys: {list(train_sample.keys())}")
            print(f"First sample: {train_sample}")

            print(f"Size of training dataset: {len(dataset['train'])}")
        else:
            print("No 'train' split found in dataset")
            print(f"Available splits: {list(dataset.keys())}")

        if "validation" in dataset:
            print("\nTesting retrieving first row of validation dataset:")
            val_sample = dataset["validation"][0]
            print(f"Keys: {list(val_sample.keys())}")
            print(f"First sample: {val_sample}")
            print(f"Size of validation dataset: {len(dataset['validation'])}")
        elif "test" in dataset:
            print("\nTesting retrieving first row of test dataset:")
            test_sample = dataset["test"][0]
            print(f"Keys: {list(test_sample.keys())}")
            print(f"First sample: {test_sample}")
        else:
            print("No 'validation' or 'test' split found in dataset")
            print(f"Available splits: {list(dataset.keys())}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        msg = (
            f"Make sure the dataset has been downloaded to "
            f"{JORDAN_DATASET_FILEPATH}"
        )
        print(msg)


if __name__ == "__main__":
    download_jordan_dataset()
    test_dataset_retrieval()
