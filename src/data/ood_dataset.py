import os

from torch.utils.data import Dataset

# TODO: upload MaestroDataset
# This will be for evaluation purposes


class MaestroDataset(Dataset):
    def __init__(self, data_dir, split, num_samples=None):
        data_path = os.path.join(data_dir, f"{split}.txt")
        self.data = []
        print("path", data_path)
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = [int(x) for x in line.split()]
                    if tokens:
                        self.data.append(tokens)
        if num_samples is not None:
            self.data = self.data[:num_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.data[idx]}
