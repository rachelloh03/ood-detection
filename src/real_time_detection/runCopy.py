"""Real-time OOD detection using MIDI input. ALSO includes recording mode to record MIDI input and play it back."""

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
from extract_layers.pooling_functions import pool_mean_std
from main.transformation_functions import (
    extract_layer_transformation,
)
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
import json
from real_time_detection.helpers import buffer_to_midifile
from utils.convert import midi_to_events

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


def save_midi_recording(messages, filepath):
    """
    Save recorded MIDI messages to a JSON file.

    Args:
        messages: List of tuples (timestamp, msg_dict)
        filepath: Path to save the recording
    """
    recording_data = {
        "messages": [
            {
                "timestamp": timestamp,
                "type": msg["type"],
                "note": msg.get("note"),
                "velocity": msg.get("velocity"),
                "channel": msg.get("channel"),
            }
            for timestamp, msg in messages
        ]
    }

    with open(filepath, "w") as f:
        json.dump(recording_data, f, indent=2)

    print(f"Recording saved to {filepath}")


def load_midi_recording(filepath):
    """
    Load recorded MIDI messages from a JSON file.

    Returns:
        List of tuples (timestamp, msg_dict)
    """
    with open(filepath, "r") as f:
        recording_data = json.load(f)

    messages = [(msg["timestamp"], msg) for msg in recording_data["messages"]]

    print(f"Recording loaded from {filepath} ({len(messages)} messages)")
    return messages


def playback_midi_recording(messages, ood_detector, layer_idxs, pooling_function):
    """
    Playback recorded MIDI messages and process them.

    Args:
        messages: List of tuples (timestamp, msg_dict) from load_midi_recording
        ood_detector: Fitted OOD detector
        model: The model to use for extraction
        layer_idxs: Layer indices to extract
        pooling_function: Pooling function to use
    """
    buffer = deque(maxlen=SLIDING_WINDOW_LEN)
    pedal_down = False

    print("Playing back recording...")

    for timestamp, msg_dict in messages:
        # Reconstruct mido.Message from dict
        msg = mido.Message(
            msg_dict["type"],
            note=msg_dict.get("note", 0),
            velocity=msg_dict.get("velocity", 0),
            channel=msg_dict.get("channel", 0),
        )

        if msg.type in ("note_on", "note_off"):
            if msg.note == 48:  # Pedal
                if not pedal_down:
                    pedal_down = True
                    print("Pedal down, listening...")
                else:
                    pedal_down = False
                    print("Pedal up, resetting buffer...")
                    buffer.clear()

            elif pedal_down:
                # Treat note_on velocity=0 as note_off
                if msg.type == "note_on" and msg.velocity == 0:
                    msg = msg.copy(type="note_off")

                buffer.append((timestamp, msg))

                # Check OOD with current buffer
                midifile = buffer_to_midifile(list(buffer))
                tokens = midi_to_events(midifile)  # (L,)
                tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(
                    0
                )  # (1, L)
                ood_score = ood_detector.score(tokens_tensor)  # (1,)
                print(f"OOD score (buffer size {len(buffer)}): {ood_score.item():.4f}")


def main():
    layer_idxs = [12]
    buffer = deque(maxlen=SLIDING_WINDOW_LEN)
    pedal_down = False
    pooling_function = pool_mean_std
    ood_detector = setup_ood_detector(layer_idxs)

    # Configuration
    RECORD_MODE = True  # Set to True to record, False to playback
    RECORDING_FILE = "midi_recording.json"

    if RECORD_MODE:
        # Recording mode
        print("Available MIDI ports:")
        for port in mido.get_input_names():
            print(port)

        INPUT_NAME = "LUMI Keys BLOCK"

        # List to store all messages for recording
        recorded_messages = []

        with mido.open_input(INPUT_NAME) as inport:
            print(f"Recording MIDI from {INPUT_NAME}...")
            print("Press Ctrl+C to stop recording and save")

            try:
                for msg in inport:
                    if msg.type in ("note_on", "note_off"):
                        timestamp = time.perf_counter()

                        # Record the message
                        recorded_messages.append(
                            (
                                timestamp,
                                {
                                    "type": msg.type,
                                    "note": msg.note,
                                    "velocity": msg.velocity,
                                    "channel": msg.channel,
                                },
                            )
                        )

                        if msg.note == 48:  # Pedal
                            if not pedal_down:
                                pedal_down = True
                                print("Pedal down, listening...")
                            else:
                                pedal_down = False
                                print("Pedal up, resetting buffer...")
                                buffer.clear()

                        elif pedal_down:
                            # Treat note_on velocity=0 as note_off
                            if msg.type == "note_on" and msg.velocity == 0:
                                msg = msg.copy(type="note_off")

                            buffer.append((timestamp, msg))

                            # Check OOD with current buffer
                            midifile = buffer_to_midifile(list(buffer))
                            tokens = midi_to_events(midifile)  # (L,)
                            tokens_tensor = torch.tensor(
                                tokens, dtype=torch.long
                            ).unsqueeze(
                                0
                            )  # (1, L)
                            ood_score = ood_detector.score(tokens_tensor)  # (1,)
                            print(
                                f"OOD score (buffer size {len(buffer)}): {ood_score.item():.4f}"
                            )

            except KeyboardInterrupt:
                print("\nRecording stopped by user")
                save_midi_recording(recorded_messages, RECORDING_FILE)

    else:
        # Playback mode
        print("Playback mode enabled")
        messages = load_midi_recording(RECORDING_FILE)
        playback_midi_recording(messages, ood_detector, layer_idxs, pooling_function)


if __name__ == "__main__":
    main()
