import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import login

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def collate_fn(examples):
    """Collate function for DataLoader"""
    input_ids = torch.stack([torch.tensor(ex['input_ids']) for ex in examples])
    attention_mask = torch.stack([torch.tensor(ex['attention_mask']) for ex in examples])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

@torch.no_grad()
def extract_representations(
    model,
    dataloader,
    save_dir,
    layers,
):
    """Extract pooled representations from specified layers"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    buffers = {l: [] for l in layers}
    
    for batch in tqdm(dataloader, desc="Extracting hidden states"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
        )
        
        hidden_states = outputs.hidden_states  # tuple(len = n_layers+1)
        
        for l in layers:
            h = hidden_states[l]  # (B, T, D)
            
            # Pool using mean and std across sequence dimension
            mean = h.mean(dim=1)
            std = h.std(dim=1)
            pooled = torch.cat([mean, std], dim=-1)
            buffers[l].append(pooled.cpu())
    
    # Save all layers
    for l in layers:
        X = torch.cat(buffers[l], dim=0).numpy()
        np.save(os.path.join(save_dir, f"layer_{l}.npy"), X)
        print(f"Saved layer {l}: {X.shape}")
    
    return buffers

def main():
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "mitmedialab/JordanAI-disklavier-v0.1-pytorch",
        torch_dtype=torch.float32,
    ).to(DEVICE)
    
    # Get number of layers
    n_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.model.layers)
    layers_to_extract = list(range(n_layers + 1))  # +1 for embeddings
    
    print(f"Model has {n_layers} layers, extracting from layers: {layers_to_extract}")
    
    # Load dataset
    print("Loading dataset...")
    # Check if we need to authenticate (for private datasets)
    # Token can be provided via HF_TOKEN environment variable or huggingface-cli login
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    
    dataset = load_dataset(
        "mitmedialab/jordan_rudess__disklavier__trading_inst0_inst1__free_time__pedal__velocity__v1",
        name="tokens/train/jordan_rudess__disklavier__trading_inst0_inst1__free_time__pedal__velocity__v1__train_noped",
        split="train",
        token=token,  # Pass token explicitly if available
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Extract representations
    save_dir = "representations"
    extract_representations(
        model=model,
        dataloader=dataloader,
        save_dir=save_dir,
        layers=layers_to_extract,
    )
    
    print(f"All representations saved to {save_dir}/")

if __name__ == "__main__":
    main()