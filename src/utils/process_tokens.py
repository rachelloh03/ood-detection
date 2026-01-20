"""Token-processing utilities."""

from constants.token_constants import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    NOTES,
    SEP,
    REST,
    AR,
    AAR,
    TIME_OFFSET,
    MAX_INSTR,
    MAX_PITCH,
    DUR_OFFSET,
    NOTE_OFFSET,
    INCLUDE_VELOCITY,
    VELOCITY_OFFSET,
    AVELOCITY_OFFSET,
    VOCAB_SIZE,
    MAX_VELOCITY,
    TIME_RESOLUTION,
)


def set_instrument(
    tokens: list[int],
    instrument: int,
) -> list[int]:
    """
    Set the instrument for all tokens in the list to be the given instrument.
    """

    def set_instrument_for_token(token: int) -> int:
        if NOTE_OFFSET <= token < NOTE_OFFSET + MAX_INSTR * MAX_PITCH:
            return (
                instrument * MAX_PITCH + (token - NOTE_OFFSET) % MAX_PITCH + NOTE_OFFSET
            )
        if ANOTE_OFFSET <= token < ANOTE_OFFSET + MAX_INSTR * MAX_PITCH:
            return (
                instrument * MAX_PITCH
                + (token - ANOTE_OFFSET) % MAX_PITCH
                + ANOTE_OFFSET
            )
        return token

    return [set_instrument_for_token(token) for token in tokens]


def set_anticipated(
    tokens: list[int],
    anticipated: bool,
) -> list[int]:
    """
    Set the anticipated flag for all tokens in the list to be the given anticipated.
    """

    def set_anticipated_for_token(token: int, anticipated: bool) -> int:
        if anticipated and (0 <= token < ATIME_OFFSET - 1):
            return token + ATIME_OFFSET
        if anticipated and (VELOCITY_OFFSET <= token < AVELOCITY_OFFSET):
            return token + MAX_VELOCITY
        if not anticipated and (ATIME_OFFSET <= token < 2 * ATIME_OFFSET - 1):
            return token - ATIME_OFFSET
        if not anticipated and (AVELOCITY_OFFSET <= token < VOCAB_SIZE):
            return token - MAX_VELOCITY
        return token

    return [set_anticipated_for_token(token, anticipated) for token in tokens]


def get_readable_events(tokens, include_velocity: bool = INCLUDE_VELOCITY):
    """Convert tokens to readable events."""
    output = []
    tokens_per_event = 4 if include_velocity else 3
    # first token
    if tokens[0] == AR:
        output.append({"special_token": "AR"})
    elif tokens[0] == AAR:
        output.append({"special_token": "AAR"})

    i = 1
    while i < len(tokens):
        token = tokens[i]

        anticipated = ATIME_OFFSET <= token < ADUR_OFFSET or (
            AVELOCITY_OFFSET <= token < AVELOCITY_OFFSET + 2 * MAX_VELOCITY
        )

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
        event_repr["onset"] = (token - time_offset) / TIME_RESOLUTION
    elif idx % tokens_per_event == 1:
        event_repr["duration"] = (token - dur_offset) / TIME_RESOLUTION
    elif idx % tokens_per_event == 2:
        if token == REST:
            event_repr["special_token"] = "REST"
            return event_repr
        val = token - note_offset
        instrument = val // MAX_PITCH
        pitch = val % MAX_PITCH
        event_repr["instrument"] = instrument
        event_repr["pitch"] = pitch_to_note(pitch)
    elif idx % tokens_per_event == 3:
        event_repr["velocity"] = token - velocity_offset
    return event_repr


def pitch_to_note(pitch):
    """Convert pitch to note"""
    return f"{NOTES[pitch % 12]}{pitch // 12 - 1}"
