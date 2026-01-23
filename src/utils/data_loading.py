"""
Utility functions for loading data.
"""

# import torch


# def collate_fn(examples):
#     """Collate function for DataLoader"""
#     input_ids = torch.stack(
#         [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]
#     )
#     return {
#         "input_ids": input_ids,
#     }
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(examples):
    """
    Collate function that pads sequences to the same length.
    """
    # Extract input_ids and convert to tensors
    input_ids_list = []
    for ex in examples:
        if isinstance(ex["input_ids"], torch.Tensor):
            input_ids_list.append(ex["input_ids"].detach().clone())
        else:
            input_ids_list.append(torch.tensor(ex["input_ids"], dtype=torch.long))
    
    # Pad sequences to same length (batch_first=True → shape: [batch, seq_len])
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    
    return {"input_ids": input_ids}
