"""Helper functions for real-time OOD detection."""

import mido
import torch
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

    buffer.sort(key=lambda x: x[0])

    last_time = buffer[0][0]
    for t, msg in buffer:
        msg_copy = msg.copy(time=t - last_time)
        track.append(msg_copy)
        last_time = t

    return mid


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

    output = model(
        input_ids=torch.tensor(tokens, dtype=torch.long).unsqueeze(0),
        output_hidden_states=True,
    )  # (1, L, D)
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
