"""Helper functions for real-time OOD detection."""

import mido


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
