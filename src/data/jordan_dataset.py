from torch.utils.data import Dataset
from datasets import load_from_disk
import torch
from constants.token_constants import (
    INCLUDE_VELOCITY,
    VELOCITY_OFFSET,
    MAX_VELOCITY,
    NOTE_OFFSET,
    AR,
)

# from utils.process_tokens import set_anticipated, set_instrument
from utils.process_tokens import filter_instrument
from utils.sanity_checks import check_valid_input_ids


class JordanDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        name,
        num_samples=None,
        include_velocity=INCLUDE_VELOCITY,
        debug=False,
    ):
        """
        Load Jordan dataset from Arrow files and pre-process input_ids.

        Args:
            data_dir: Directory where dataset was saved with save_to_disk()
            split: Dataset split to load ('train', 'validation')
            name: Name of the dataset (e.g. 'jordan_dataset', 'maestro_dataset')
            num_samples: Optional limit on number of samples to load
            num_events_per_sample: Optional limit on number of events per sample
            include_velocity: Whether to include velocity information
            debug: Whether to print debug information

        """
        if debug:
            print(f"Loading {split} split from {data_dir}...")
            print(f"Name: {name}")
            print(f"Num samples: {num_samples}")
            print(f"Include velocity: {include_velocity}")

        dataset = load_from_disk(data_dir)
        self.name = name

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
        for i, sample in enumerate(self.dataset):
            original_sample_tokens = []
            if "text" in sample:
                original_sample_tokens = [int(x) for x in sample["text"].split()]
                sample_tokens = original_sample_tokens.copy()
                for j in range(0, len(sample_tokens), 4):
                    if sample_tokens[j] == MAX_VELOCITY - 1:
                        sample_tokens[j] = MAX_VELOCITY + VELOCITY_OFFSET - 1
            elif "input_ids" in sample:
                original_sample_tokens = sample["input_ids"]
                sample_tokens = original_sample_tokens.copy()
            else:
                raise KeyError(
                    f"Sample must have either 'text' or 'input_ids' field. "
                    f"Found keys: {list(sample.keys())}"
                )

            if not include_velocity:
                sample_tokens = [
                    x for j, x in enumerate(sample_tokens) if (j % 4 != 0 or j == 0)
                ]

            sample_tokens[0] = AR

            sample_tokens[3::3] = [
                x + NOTE_OFFSET if x < 256 else x for x in sample_tokens[3::3]
            ]

            input_ids = filter_instrument(
                sample_tokens, 0, include_velocity=include_velocity
            )
            output_ids = filter_instrument(
                sample_tokens, 1, include_velocity=include_velocity
            )

            try:
                check_valid_input_ids(input_ids)
                check_valid_input_ids(output_ids)
            except AssertionError as e:
                print(f"AssertionError: {e} at sample index {i}")
                bad_samples += 1
                continue
            except ValueError as e:
                print(f"ValueError: {e} at sample index {i}")
                bad_samples += 1
                continue

            processed_sample = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "output_ids": torch.tensor(output_ids, dtype=torch.long),
            }

            processed_data.append(processed_sample)

        self.dataset = processed_data

        if debug:
            print(f"Loaded {len(self.dataset)} samples from {split} split")
        if len(self.dataset) > 0 and debug:
            print(f"Sample keys: {list(self.dataset[0].keys())}")

        print(f"Detected {bad_samples} bad samples")
        if len(self.dataset) == 0:
            print(
                f"WARNING: No valid samples loaded from {split} split. All samples were filtered out."
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
