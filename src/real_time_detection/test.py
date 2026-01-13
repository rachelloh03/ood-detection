# test_model.py
import torch
from transformers import AutoModelForCausalLM

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "mitmedialab/JordanAI-disklavier-v0.1-pytorch",
    torch_dtype=torch.float32,
)
print("Model loaded!")

print("Moving to CPU...")
model = model.to("cpu")
print("Model on CPU!")

print("Setting eval mode...")
model.eval()
print("Success!")