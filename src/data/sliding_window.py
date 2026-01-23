from data.jordan_dataset import check_valid_input_ids
from torch.utils.data import Dataset
import torch
import random
from constants.real_time_constants import TOKENS_PER_EVENT


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        name: str,
        k: int,
        stride: int = 1,
        num_samples: int | None = None,
        drop_last: bool = True,
        lm_labels: bool = False,
        first_token_special: bool = True,
    ):
        """
        Wraps a dataset and returns all length-k subsequences as new samples.

        If first_token_special=True, the first token is always prepended,
        and the sliding window is applied to input_ids[1:].

        Args:
            base_dataset: Dataset returning dicts with 'input_ids' (and optionally 'labels')
            k: Length of sliding window in events (excluding special first token) i.e. k=40 means 120 tokens
                if each event is 3 tokens (default)
            name: Name of the dataset
            stride: Step size between windows in events i.e. stride=10 means 30 tokens
                if each event is 3 tokens (default)
            drop_last: Drop windows shorter than k
            num_samples: Optional limit on number of samples to load, picks randomly
            lm_labels: If True, labels = input_ids shifted by 1
            first_token_special: Whether to treat input_ids[0] as fixed prefix
        """
        self.base = base_dataset
        self.k = k * TOKENS_PER_EVENT
        self.stride = stride * TOKENS_PER_EVENT
        self.drop_last = drop_last
        self.lm_labels = lm_labels
        self.first_token_special = first_token_special
        self.name = name

        self.index = []
        for i in range(len(self.base)):
            ids = self.base[i]["input_ids"]
            n = len(ids)

            if self.first_token_special:
                if n <= 1:
                    continue
                tail_len = n - 1
            else:
                tail_len = n

            max_start = tail_len - k if drop_last else tail_len - 1

            for start in range(0, max_start + 1, stride):
                self.index.append((i, start))

        if num_samples is not None and num_samples < len(self.index):
            self.index = random.sample(self.index, num_samples)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sample_idx, start = self.index[idx]
        base_sample = self.base[sample_idx]

        ids = base_sample["input_ids"]

        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)

        try:
            check_valid_input_ids(ids.tolist())
        except Exception as e:
            print(f"Error: {e} at sample index {idx}")
            print(f"Input IDs: {ids}")
            raise e

        if self.first_token_special:
            prefix = ids[:1]
            tail = ids[1:]
            window = tail[start : start + self.k]
            input_ids = torch.cat([prefix, window])
        else:
            input_ids = ids[start : start + self.k]

        if self.lm_labels:
            labels = input_ids[1:].clone()
            input_ids = input_ids[:-1].clone()
        else:
            if "labels" in base_sample:
                base_labels = base_sample["labels"]
                if not isinstance(base_labels, torch.Tensor):
                    base_labels = torch.tensor(base_labels, dtype=torch.long)
                labels = base_labels[start : start + len(input_ids)].clone()
            else:
                labels = input_ids.clone()

        return {
            "input_ids": input_ids.clone(),
            "labels": labels,
        }
