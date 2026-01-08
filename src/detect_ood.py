"""Detection pipeline for OOD detection."""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ood_detector import OODDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OODPipeline:
    """Complete pipeline for OOD detection on new prompts"""

    def __init__(self, model_name, detector_path, layer_idx):
        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(DEVICE)
        self.model.eval()

        print(f"Loading OOD detector from {detector_path}...")
        self.detector = OODDetector.load(detector_path)

        self.layer_idx = layer_idx

        print("Pipeline ready!")

    @torch.no_grad()
    def extract_representation(self, input_ids, attention_mask):
        """Extract representation from the specified layer"""

        # Prepare input
        batch = {
            "input_ids": input_ids.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE),
        }

        # Forward pass
        outputs = self.model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract hidden state from target layer
        h = outputs.hidden_states[self.layer_idx]  # (B, T, D)

        # Pool: concatenate mean and std
        mean = h.mean(dim=1)
        std = h.std(dim=1)
        pooled = torch.cat([mean, std], dim=-1)

        return pooled.cpu().numpy()

    def check_ood(self, input_ids, attention_mask):
        """
        Check if input is out-of-distribution

        Returns:
            dict with keys: is_ood, score, threshold
        """
        # Extract representation
        X = self.extract_representation(input_ids, attention_mask)

        # Compute OOD score
        score = self.detector.score(X)[0]
        is_ood = score > self.detector.threshold

        # Compute percentile relative to training distribution
        percentile = (self.detector.train_scores < score).mean() * 100

        return {
            "is_ood": bool(is_ood),
            "score": float(score),
            "threshold": float(self.detector.threshold),
            "percentile": float(percentile),
            "z_score": float(
                (score - self.detector.train_scores.mean())
                / (self.detector.train_scores.std() + 1e-8)
            ),
        }


def main():
    # Load configuration
    with open("analysis/best_layer.txt", "r") as f:
        best_layer = f.read().strip()

    layer_idx = int(best_layer.split("_")[1])

    # Initialize pipeline
    pipeline = OODPipeline(
        model_name="mitmedialab/JordanAI-disklavier-v0.1-pytorch",
        detector_path=f"ood_detector_{best_layer}.pkl",
        layer_idx=layer_idx,
    )

    # Example usage - you'll need to prepare your input properly
    print("\n" + "=" * 60)
    print("Example: Testing a sample prompt")
    print("=" * 60)

    # Create a dummy example (replace with your actual tokenized input)
    # For actual use, tokenize your prompt properly
    example_input_ids = torch.randint(0, 1000, (1, 512))  # (batch_size, seq_len)
    example_attention_mask = torch.ones(1, 512)

    result = pipeline.check_ood(example_input_ids, example_attention_mask)

    print(f"\nOOD Detection Result:")
    print(f"  Is OOD: {result['is_ood']}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  Percentile: {result['percentile']:.1f}%")
    print(f"  Z-score: {result['z_score']:.2f}")

    if result["is_ood"]:
        print("\n⚠️  WARNING: This input appears to be out-of-distribution!")
    else:
        print("\n✓ This input appears to be in-distribution.")

    return pipeline


if __name__ == "__main__":
    pipeline = main()

    print("\n" + "=" * 60)
    print("To use in your code:")
    print("=" * 60)
    print(
        """
from detect_ood import OODPipeline

# Load pipeline
pipeline = OODPipeline(
    model_name="mitmedialab/JordanAI-disklavier-v0.1-pytorch",
    detector_path="ood_detector_layer_X.pkl",
    layer_idx=X
)

# Check if prompt is OOD
result = pipeline.check_ood(input_ids, attention_mask)

if result['is_ood']:
    print("Out of distribution!")
    """
    )
