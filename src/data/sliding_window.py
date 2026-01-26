from constants.token_constants import TIME_RESOLUTION, AR
from data.jordan_dataset import check_valid_input_ids
from torch.utils.data import Dataset
import torch
import random
from constants.real_time_constants import TOKENS_PER_EVENT
from utils.process_tokens import get_readable_events


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
        max_silence_length: int = 5,
    ):
        self.base = base_dataset
        self.k = k * TOKENS_PER_EVENT
        self.stride = stride * TOKENS_PER_EVENT
        self.drop_last = drop_last
        self.lm_labels = lm_labels
        self.first_token_special = first_token_special
        self.name = name
        self.max_silence_length = max_silence_length * TIME_RESOLUTION

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

            max_start = tail_len - self.k if drop_last else tail_len - 1

            for start in range(0, max_start + 1, self.stride):
                if self._has_acceptable_gaps(ids, start):
                    self.index.append((i, start))

        if num_samples is not None and num_samples < len(self.index):
            self.index = random.sample(self.index, num_samples)

    def _has_acceptable_gaps(self, ids, start):
        if self.max_silence_length is None or self.max_silence_length < 0:
            return True

        if self.first_token_special:
            tail = ids[1:] if isinstance(ids, torch.Tensor) else ids[1:]
            window_tokens = tail[start : start + self.k]
        else:
            window_tokens = ids[start : start + self.k]

        if isinstance(window_tokens, torch.Tensor):
            window_tokens = window_tokens.tolist()

        if self.first_token_special:
            tokens_with_prefix = [AR] + window_tokens
        else:
            tokens_with_prefix = window_tokens

        readable_events = get_readable_events(
            tokens_with_prefix, include_velocity=False
        )

        events = []
        for event in readable_events:
            if "special_token" in event:
                continue
            if "onset" in event and "duration" in event:
                onset = int(event["onset"] * TIME_RESOLUTION)
                duration = int(event["duration"] * TIME_RESOLUTION)
                events.append((onset, duration))

        if len(events) < 2:
            return True

        events.sort(key=lambda x: x[0])

        max_end_time = events[0][0] + events[0][1]

        for j in range(1, len(events)):
            next_onset, next_duration = events[j]
            next_end_time = next_onset + next_duration

            if next_onset > max_end_time:
                gap = next_onset - max_end_time
                if gap > self.max_silence_length:
                    return False

            max_end_time = max(max_end_time, next_end_time)

        return True

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
