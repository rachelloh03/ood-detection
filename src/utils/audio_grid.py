"""
Utility function to create a grid visualization of audio samples.

This is used in example.ipynb to visualize the audio samples.

Thanks cursor! - Joel
"""

import torch
from IPython.display import HTML, display
from utils.convert import sequence_to_wav
import os
import tempfile
import shutil
import base64


def create_audio_grid(
    samples: torch.Tensor,
    indices: list[int] | None = None,
    step: int = 10,
    max_samples: int = 20,
    cols: int = 4,
    include_velocity: bool = False,
):
    """
    Create a grid visualization of audio samples.

    Args:
        samples: Tensor of shape (N, L) where N is number of samples and L is sequence length
        indices: Optional list of specific indices to display. If None, uses step-based selection.
        step: Step size for selecting indices (e.g., 10 means indices 0, 10, 20, ...)
        max_samples: Maximum number of samples to display
        cols: Number of columns in the grid
        include_velocity: Whether to include velocity when converting to audio

    Returns:
        HTML display object with audio grid
    """
    if indices is None:
        indices = list(range(0, len(samples), step))[:max_samples]
    else:
        indices = indices[:max_samples]

    temp_dir = tempfile.mkdtemp()
    audio_data_list = []

    try:
        for idx in indices:
            if idx >= len(samples):
                break

            sequence = samples[idx].tolist()
            audio_path = os.path.join(temp_dir, f"sample_{idx}.wav")
            try:
                sequence_to_wav(sequence, audio_path, include_velocity=include_velocity)
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                    audio_data_list.append((idx, audio_base64))
            except Exception as e:
                print(f"Warning: Could not create audio for index {idx}: {e}")
                continue

        html_content = f'<div style="display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 20px; " + \
        "padding: 20px; max-width: 1200px;">'

        for idx, audio_base64 in audio_data_list:
            html_content += f"""
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center; " + \
            "background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-weight: bold; margin-bottom: 10px; font-size: 14px; color: #333;">Index {idx}</div>
                <audio controls style="width: 100%; max-width: 250px;">
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """

        html_content += "</div>"

        display(HTML(html_content))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
