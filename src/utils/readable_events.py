"""Convert tokens to readable text."""

from constants import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    NOTES,
    SEP,
    REST,
    AR,
    AAR,
    TIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    INCLUDE_VELOCITY,
    VELOCITY_OFFSET,
    AVELOCITY_OFFSET,
)


def get_readable_events(tokens, include_velocity=INCLUDE_VELOCITY):
    """Convert tokens to readable events."""
    tokens_per_event = 4 if include_velocity else 3

    output = []
    # first token
    if tokens[0] == AR:
        output.append({"special_token": "AR"})
    elif tokens[0] == AAR:
        output.append({"special_token": "AAR"})

    i = 1
    while i < len(tokens):
        token = tokens[i]

        anticipated = ATIME_OFFSET <= token < ADUR_OFFSET

        event_tokens = tokens[i : i + tokens_per_event]

        # if any token in the event is SEP, output SEP
        if any(t == SEP for t in event_tokens):
            output.append({"special_token": "SEP"})
            i += tokens_per_event
            continue

        event_repr = {}
        for idx, t in enumerate(event_tokens):
            event_repr = token_to_event(
                event_repr,
                idx % tokens_per_event,
                t,
                anticipated,
                include_velocity=include_velocity,
            )
        event_repr["anticipated"] = anticipated
        output.append(event_repr)
        i += tokens_per_event

    return output


def token_to_event(
    event_repr, idx, token, anticipated, include_velocity=INCLUDE_VELOCITY
):
    """Convert tokens into a readable event"""
    tokens_per_event = 4 if include_velocity else 3
    time_offset = TIME_OFFSET if not anticipated else ATIME_OFFSET
    dur_offset = DUR_OFFSET if not anticipated else ADUR_OFFSET
    note_offset = NOTE_OFFSET if not anticipated else ANOTE_OFFSET
    velocity_offset = VELOCITY_OFFSET if not anticipated else AVELOCITY_OFFSET

    if idx % tokens_per_event == 0:
        event_repr["onset"] = (token - time_offset) * 0.01
    elif idx % tokens_per_event == 1:
        event_repr["duration"] = (token - dur_offset) * 0.01
    elif idx % tokens_per_event == 2:
        if token == REST:
            event_repr["special_token"] = "REST"
            return event_repr
        val = token - note_offset
        instrument = val // 128
        pitch = val % 128
        event_repr["instrument"] = instrument
        event_repr["pitch"] = pitch_to_note(pitch)
    elif idx % tokens_per_event == 3:
        event_repr["velocity"] = token - velocity_offset
    return event_repr


def pitch_to_note(pitch):
    """Convert pitch to note"""
    return f"{NOTES[pitch % 12]}{pitch // 12 - 1}"
