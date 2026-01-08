from torch.utils.data import Dataset
from datasets import load_from_disk


class JordanDataset(Dataset):
    def __init__(self, data_dir, split, num_samples=None):
        """
        Load Jordan dataset from Arrow files.

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
            # Select first num_samples
            self.dataset = self.dataset.select(
                range(min(num_samples, len(self.dataset)))
            )

        print(f"Loaded {len(self.dataset)} samples from {split} split")
        if len(self.dataset) > 0:
            print(f"Sample keys: {list(self.dataset[0].keys())}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if "text" in sample:
            input_ids = [int(x) for x in sample["text"].split()]
        elif "input_ids" in sample:
            input_ids = sample["input_ids"]
        else:
            raise KeyError(
                f"Sample must have either 'text' or 'input_ids' field. "
                f"Found keys: {list(sample.keys())}"
            )

        result = {"input_ids": input_ids}
        if "labels" in sample:
            result["labels"] = sample["labels"]
        else:
            result["labels"] = input_ids

        return result
