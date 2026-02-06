from torch.utils.data import Dataset
from datasets import load_from_disk
import torch
from constants.token_constants import (
    INCLUDE_VELOCITY,
    VELOCITY_OFFSET,
    MAX_VELOCITY,
    AR,
)

from utils.process_tokens import filter_instrument, set_instrument
from utils.sanity_checks import check_valid_input_ids


class JordanDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        name,
        split_input_and_output_ids,
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
            split_input_and_output_ids: Whether to split the input_ids and output_ids into separate fields
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
        self.split_input_and_output_ids = split_input_and_output_ids

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
            elif "input_ids" in sample:
                original_sample_tokens = sample["input_ids"]
                sample_tokens = original_sample_tokens.copy()
            else:
                raise KeyError(
                    f"Sample must have either 'text' or 'input_ids' field. "
                    f"Found keys: {list(sample.keys())}"
                )

            has_velocity_tokens = any(
                VELOCITY_OFFSET <= t < VELOCITY_OFFSET + MAX_VELOCITY
                for t in sample_tokens
            )
            if not include_velocity and has_velocity_tokens:
                sample_tokens = [
                    x for j, x in enumerate(sample_tokens) if (j % 4 != 0 or j == 0)
                ]

            sample_tokens[0] = AR

            if self.split_input_and_output_ids:
                input_ids = filter_instrument(
                    sample_tokens, 0, include_velocity=include_velocity
                )
                output_ids = filter_instrument(
                    sample_tokens, 1, include_velocity=include_velocity
                )
            else:
                input_ids = set_instrument(sample_tokens, 0)
                output_ids = None

            try:
                check_valid_input_ids(input_ids)
                if output_ids is not None:
                    check_valid_input_ids(output_ids)
            except AssertionError as e:
                print(f"AssertionError: {e} at sample index {i}")
                bad_samples += 1
                print(f"Input IDs (first 50): {input_ids[:50]}")
                if output_ids is not None:
                    print(f"Output IDs (first 50): {output_ids[:50]}")
                raise e
            except ValueError as e:
                print(f"ValueError: {e} at sample index {i}")
                bad_samples += 1
                print(f"Input IDs (first 50): {input_ids[:50]}")
                if output_ids is not None:
                    print(f"Output IDs (first 50): {output_ids[:50]}")
                print(f"Original tokens (first 50): {original_sample_tokens[:50]}")
                print(max(original_sample_tokens))
                print(f"Processed sample_tokens (first 50): {sample_tokens[:50]}")
                raise e

            processed_sample = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "output_ids": (
                    torch.tensor(output_ids, dtype=torch.long)
                    if output_ids is not None
                    else None
                ),
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
