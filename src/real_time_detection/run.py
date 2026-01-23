"""Real-time OOD detection using MIDI input."""

# import sys
# from pathlib import Path

# # Add src to path
# # comment out these next two lines out if running on hai-res
# # added for local machine purposes
# src_dir = Path(__file__).parent.parent
# sys.path.insert(0, str(src_dir))

# if you set PYTHONPATH in ~/.bashrc you don't need to do the sys.path.etc

from data.jordan_dataset import JordanDataset
from extract_layers.extract_layers_main import BATCH_SIZE
from main.transformation_functions import extract_layer_transformation
from extract_layers.pooling_functions import pool_mean_std
from real_time_detection.helpers import buffer_to_midifile
from utils.convert import midi_to_events
from main.transformations import Transformations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from main.scoring_functions import mahalanobis_distance
from main.ood_detector import OODDetector
from constants.data_constants import JORDAN_DATASET_FILEPATH
from torch.utils.data import DataLoader
import mido
import time
from transformers import AutoModelForCausalLM
from constants.model_constants import JORDAN_MODEL_NAME, DEVICE
from constants.real_time_constants import SLIDING_WINDOW_LEN
import torch
from collections import deque
from utils.data_loading import collate_fn
from main.save_ood_detector_params import save_ood_detector_params

model = AutoModelForCausalLM.from_pretrained(JORDAN_MODEL_NAME, dtype=torch.float32).to(
    DEVICE
)


def setup_ood_detector(layer_idxs):
    """
    Sets up the OOD detector for the given layer indices.
    """
    dataset = JordanDataset(
        data_dir=JORDAN_DATASET_FILEPATH,
        split="train",
        name="id_train_dataset",
    )
    id_train_dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    transformations = Transformations(
        [
            extract_layer_transformation(model, pool_mean_std, layer_idxs),
            PCA(n_components=10),
            StandardScaler(),
        ]
    )
    scoring_function = mahalanobis_distance
    ood_detector = OODDetector(
        embedding_function=transformations,
        scoring_function=scoring_function,
        id_train_data=id_train_dataloader,
    )
    save_ood_detector_params(ood_detector, "ood_detector_params")

    return ood_detector


def main():
    layer_idxs = [12]
    pedal_down = False
    buffer = deque(maxlen=SLIDING_WINDOW_LEN)
    ood_detector = setup_ood_detector(layer_idxs)

    print("Available MIDI ports:")
    for port in mido.get_input_names():
        print(port)

    INPUT_NAME = "LUMI Keys BLOCK"

    with mido.open_input(INPUT_NAME) as inport:
        for msg in inport:
            if msg.type in ("note_on", "note_off"):
                if msg.note == 48:  # let left-most note be the "pedal" for now
                    if not pedal_down:
                        pedal_down = True
                        print("Pedal down, listening...")
                    else:
                        pedal_down = False
                        print("Pedal up, resetting buffer...")
                        buffer.clear()  # Clear buffer on pedal up

                # Notes (check OOD if pedal is down)
                elif pedal_down:
                    # Treat note_on velocity=0 as note_off
                    if msg.type == "note_on" and msg.velocity == 0:
                        msg = msg.copy(type="note_off")

                    buffer.append((time.perf_counter(), msg))

                    midifile = buffer_to_midifile(list(buffer))
                    tokens = midi_to_events(midifile)  # (L,)
                    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(
                        0
                    )  # (1, L)
                    ood_score = ood_detector.score(tokens_tensor)  # (1,)
                    print(
                        f"OOD score (buffer size {len(buffer)}): {ood_score.item():.4f}"
                    )


if __name__ == "__main__":
    main()
