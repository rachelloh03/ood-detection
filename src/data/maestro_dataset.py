import os

from torch.utils.data import Dataset

from constants.token_constants import INCLUDE_VELOCITY


class MaestroDataset(Dataset):
    def __init__(self, data_dir, split, name, num_samples=None):
        self.name = name
        data_path = os.path.join(data_dir, f"{split}.txt")
        self.data = []

        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = [int(x) for x in line.split()]
                    if not INCLUDE_VELOCITY:
                        tokens = [
                            x for i, x in enumerate(tokens) if (i % 4 != 0 or i == 0)
                        ]
                    if tokens:
                        self.data.append(tokens)
        if num_samples is not None:
            self.data = self.data[:num_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.data[idx]}
