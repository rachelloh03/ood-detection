"""Real-time OOD detection using MIDI input."""

from extract_layers.pooling_functions import pool_mean_std
from real_time_detection.helpers import setup_ood_detector, extract_layer
import mido
import time
from transformers import AutoModelForCausalLM
from constants import JORDAN_MODEL_NAME, DEVICE

import torch


def main():
    layer_idxs = [12]
    pedal_down = False
    buffer = []
    pooling_function = pool_mean_std
    ood_detector = setup_ood_detector(layer_idxs)
    model = AutoModelForCausalLM.from_pretrained(
        JORDAN_MODEL_NAME,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    model.eval()

    print("Available MIDI ports:")
    for port in mido.get_input_names():
        print(port)

    INPUT_NAME = "Insert input name for the piano"

    with mido.open_input(INPUT_NAME) as inport:
        for msg in inport:
            if msg.type == "control_change" and msg.control == 64:
                if msg.value >= 64 and not pedal_down:
                    pedal_down = True
                    buffer = []
                    print("Pedal down, listening...")

                elif msg.value < 64 and pedal_down:
                    pedal_down = False
                    print("Pedal up, scoring...")
                    extracted_layer = extract_layer(
                        buffer, pooling_function, model, layer_idxs
                    )  # (D,)
                    ood_score = ood_detector.score(extracted_layer.unsqueeze(0))  # (1,)
                    print(f"OOD score: {ood_score.item()}")
                    buffer = []

            # Notes (buffer only if pedal is down)
            elif pedal_down and msg.type in ("note_on", "note_off"):
                # Treat note_on velocity=0 as note_off
                if msg.type == "note_on" and msg.velocity == 0:
                    msg = msg.copy(type="note_off")

                buffer.append((time.perf_counter(), msg))


if __name__ == "__main__":
    main()
