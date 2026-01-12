"""Real-time OOD detection using MIDI input."""

from extract_layers.pooling_functions import pool_mean_std
from real_time_detection.helpers import extract_layer
from main.transformations import Transformations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from main.scoring_functions import mahalanobis_distance
from main.ood_detector import OODDetector
import numpy as np
from constants.data_constants import SCRATCH_FILEPATH
import mido
import time
from transformers import AutoModelForCausalLM
from constants.model_constants import JORDAN_MODEL_NAME, DEVICE

import torch


def setup_ood_detector(layer_idxs):
    """
    Sets up the OOD detector for the given layer indices.
    """
    id_train_data = []
    for layer_idx in layer_idxs:
        id_train_data.append(
            np.load(f"{SCRATCH_FILEPATH}/id_train_dataset/layer_{layer_idx}.npy")
        )  # (N, D)
    id_train_data = np.concatenate(id_train_data, axis=0)  # (L, N, D)
    # print(id_train_data.shape)
    # if len(layer_idxs) == 1:
    #     id_train_data = id_train_data.squeeze(0)  # (N, D)

    transformations = Transformations(
        [
            PCA(n_components=10),
            StandardScaler(),
        ]
    )
    scoring_function = mahalanobis_distance
    ood_detector = OODDetector(
        embedding_function=transformations,
        scoring_function=scoring_function,
        id_train_data=id_train_data,
    )
    return ood_detector


def main():
    layer_idxs = [12]
    pedal_down = False
    buffer = []
    pooling_function = pool_mean_std
    ood_detector = setup_ood_detector(layer_idxs)
    model = AutoModelForCausalLM.from_pretrained(
        JORDAN_MODEL_NAME,
        dtype=torch.float32
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
