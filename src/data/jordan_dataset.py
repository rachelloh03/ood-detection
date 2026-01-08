from torch.utils.data import Dataset
from datasets import load_from_disk
import torch
from constants import INCLUDE_VELOCITY
from utils.sanity_checks import check_valid_input_ids


class JordanDataset(Dataset):
    def __init__(
        self, data_dir, split, num_samples=None, include_velocity=INCLUDE_VELOCITY
    ):
        """
        Load Jordan dataset from Arrow files and pre-process input_ids.

        Args:
            data_dir: Directory where dataset was saved with save_to_disk()
            split: Dataset split to load ('train', 'validation')
            num_samples: Optional limit on number of samples to load
        """
        print(f"Loading {split} split from {data_dir}...")
        dataset = load_from_disk(data_dir)

        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(
                f"Split '{split}' not found in dataset. "
                f"Available splits: {available_splits}"
            )

        self.dataset = dataset[split]

        if num_samples is not None:
            self.dataset = self.dataset.select(
                range(min(num_samples, len(self.dataset)))
            )

        processed_data = []
        bad_samples = 0
        for sample in self.dataset:
            if "text" in sample:
                input_ids = [int(x) for x in sample["text"].split()]
                if input_ids[4] == 127:
                    print("Something might be wrong, skipping")
                    bad_samples += 1
                    continue
            elif "input_ids" in sample:
                input_ids = sample["input_ids"]
            else:
                raise KeyError(
                    f"Sample must have either 'text' or 'input_ids' field. "
                    f"Found keys: {list(sample.keys())}"
                )

            assert (
                len(input_ids) % 4 == 1
            ), "Input IDs must be a multiple of 4 apart from the first token"
            if not include_velocity:
                input_ids = [
                    x for i, x in enumerate(input_ids) if (i % 4 != 0 or i == 0)
                ]
                assert (
                    len(input_ids) % 3 == 1
                ), "Input IDs must be a multiple of 3 apart from the first token"

            try:
                check_valid_input_ids(input_ids)
            except AssertionError as e:
                print(f"Error: {e}")
                print(f"Sample: {sample}")
                print(f"Input IDs: {input_ids}")
                raise e

            processed_sample = {"input_ids": torch.tensor(input_ids, dtype=torch.long)}
            if "labels" in sample:
                processed_sample["labels"] = torch.tensor(
                    sample["labels"], dtype=torch.long
                )
            else:
                processed_sample["labels"] = torch.tensor(input_ids, dtype=torch.long)

            processed_data.append(processed_sample)

        self.dataset = processed_data

        print(f"Loaded {len(self.dataset)} samples from {split} split")
        if len(self.dataset) > 0:
            print(f"Sample keys: {list(self.dataset[0].keys())}")

        print(f"Skipped {bad_samples} bad samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
