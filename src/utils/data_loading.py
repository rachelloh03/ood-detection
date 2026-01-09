"""
Utility functions for loading data.
"""

import torch


def collate_fn(examples):
    """Collate function for DataLoader"""
    input_ids = torch.stack(
        [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
    )
    return {
        "input_ids": input_ids,
    }
