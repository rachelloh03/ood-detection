from constants import SCRATCH_FILEPATH
import mido
from main.transformations import Transformations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from main.scoring_functions import mahalanobis_distance
import numpy as np
import torch
from main.ood_detector import OODDetector
from utils.convert import midi_to_events


def buffer_to_midifile(buffer):
    """
    Convert a buffer [(timestamp, mido.Message), ...] to a MidiFile.
    """
    if not buffer:
        return None

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Sort buffer by timestamp just in case
    buffer.sort(key=lambda x: x[0])

    # Compute relative delta times in seconds
    last_time = buffer[0][0]
    for t, msg in buffer:
        msg_copy = msg.copy(time=t - last_time)  # delta time in seconds
        track.append(msg_copy)
        last_time = t

    return mid


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
    if len(layer_idxs) == 1:
        id_train_data = id_train_data.squeeze(0)  # (N, D)

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


def extract_layer(buffer, pooling_function, model, layer_idxs):
    """
    Takes in the buffer from MIDI and returns the layer activations for the given layer indices.

    Args:
        buffer: list of (time, msg) tuples
        model: the model to extract layer activations from
        layer_idxs: list of layer indices to extract
    Returns:
        list of layer activations
    """
    midifile = buffer_to_midifile(buffer)  # MidiFile
    tokens = midi_to_events(midifile)  # (L,)

    output = model(**tokens, output_hidden_states=True)  # (1, L, D)
    hidden_states = output.hidden_states  # (n_layers + 1)-tuple of (1, L, D)

    layer_activations = []
    for layer_idx in layer_idxs:
        hidden_state = hidden_states[layer_idx]  # (1, L, D)
        pooled = pooling_function(hidden_state).squeeze(0)  # (D,)
        layer_activations.append(pooled)
    all_layer_activations = torch.stack(layer_activations)  # (len(layer_idxs), D)
    if len(layer_idxs) == 1:
        all_layer_activations = all_layer_activations.squeeze(0)  # (D,)
    return all_layer_activations
