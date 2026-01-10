"""
Utilities for converting to and from Midi data and encoded/tokenized data.
"""

from collections import defaultdict

import mido


from constants.token_constants import (
    REST,
    ATIME_OFFSET,
    MAX_DUR,
    MAX_INSTR,
    MAX_PITCH,
    TIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_RESOLUTION,
    VELOCITY_OFFSET,
    SEP,
    INCLUDE_VELOCITY,
)

DRUMS_CHANNEL = 9


def unpad(tokens, include_velocity=INCLUDE_VELOCITY):
    new_tokens = []

    if include_velocity:

        for time, dur, note, vel in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            if note == REST:
                continue

            new_tokens.extend([time, dur, note, vel])

        return new_tokens
    else:

        for time, dur, note in zip(
            tokens[0::3], tokens[1::3], tokens[2::3], strict=True
        ):
            if note == REST:
                continue

            new_tokens.extend([time, dur, note])

        return new_tokens


def midi_to_compound(midifile, debug=False):
    """
    Convert a MIDI file to a compound sequence.

    In the compound sequence, for each note we have
    (time, -1, note, instrument, velocity)
    """
    if isinstance(midifile, str):
        midi = mido.MidiFile(midifile)
    else:
        midi = midifile

    tokens = []
    note_idx = 0
    open_notes = defaultdict(list)

    time = 0
    instruments = defaultdict(int)
    for message in midi:
        time += message.time

        if message.time < 0:
            raise ValueError

        if message.type == "program_change":
            instruments[message.channel] = message.program
        elif message.type in ["note_on", "note_off"]:
            instr = (
                128
                if message.channel == DRUMS_CHANNEL
                else instruments[message.channel]
            )

            if message.type == "note_on" and message.velocity > 0:  # onset
                time_in_ticks = round(TIME_RESOLUTION * time)

                compound_word = [
                    time_in_ticks,
                    -1,
                    message.note,
                    instr,
                    message.velocity,
                ]
                tokens.extend(compound_word)

                open_notes[(instr, message.note, message.channel)].append(
                    (note_idx, time)
                )
                note_idx += 1
            else:  # offset
                try:
                    open_idx, onset_time = open_notes[
                        (instr, message.note, message.channel)
                    ].pop(0)
                except IndexError:
                    if debug:
                        print("WARNING: ignoring bad offset")
                else:
                    duration_ticks = round(TIME_RESOLUTION * (time - onset_time))
                    tokens[5 * open_idx + 1] = duration_ticks
                    # del open_notes[(instr,message.note,message.channel)]
        elif message.type == "set_tempo":
            pass  # we use real time
        elif message.type == "time_signature":
            pass  # we use real time
        elif message.type in [
            "aftertouch",
            "polytouch",
            "pitchwheel",
            "sequencer_specific",
        ]:
            pass  # we don't attempt to model these
        elif message.type == "control_change":
            pass  # this includes pedal and per-track volume: ignore for now
        elif message.type in [
            "track_name",
            "text",
            "end_of_track",
            "lyrics",
            "key_signature",
            "copyright",
            "marker",
            "instrument_name",
            "cue_marker",
            "device_name",
            "sequence_number",
        ]:
            pass  # possibly useful metadata but ignore for now
        elif message.type == "channel_prefix":
            pass  # relatively common, but can we ignore this?
        elif message.type in ["midi_port", "smpte_offset", "sysex"]:
            pass  # I have no idea what this is
        else:
            if debug:
                print("UNHANDLED MESSAGE", message.type, message)

    unclosed_count = 0
    for _, v in open_notes.items():
        unclosed_count += len(v)

    if debug and unclosed_count > 0:
        print(f"WARNING: {unclosed_count} unclosed notes")
        print("  ", midifile)

    return tokens


def compound_to_midi(tokens, debug=False):
    """
    Convert a compound sequence to a MIDI file.

    In the compound sequence, for each note we have
    (time, -1, note, instrument, velocity)
    """
    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION // 2  # 2 beats/second at quarter=120

    it = iter(tokens)
    time_index = defaultdict(list)
    for _, (time_in_ticks, duration, note, instrument, velocity) in enumerate(
        zip(it, it, it, it, it, strict=True)
    ):
        time_index[(time_in_ticks, 0)].append((note, instrument, velocity))  # 0 = onset
        time_index[(time_in_ticks + duration, 1)].append(
            (note, instrument, velocity)
        )  # 1 = offset

    track_idx = {}  # maps instrument to (track number, current time)
    num_tracks = 0
    for time_in_ticks, event_type in sorted(time_index.keys()):
        for note, instrument, velocity in time_index[(time_in_ticks, event_type)]:
            if event_type == 0:  # onset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    idx = num_tracks
                    previous_time = 0
                    track = mido.MidiTrack()
                    mid.tracks.append(track)
                    if instrument == 128:  # drums always go on channel 9
                        idx = 9
                        message = mido.Message("program_change", channel=idx, program=0)
                    else:
                        message = mido.Message(
                            "program_change", channel=idx, program=instrument
                        )
                    track.append(message)
                    num_tracks += 1
                    if num_tracks == 9:
                        num_tracks += 1  # skip the drums track

                track.append(
                    mido.Message(
                        "note_on",
                        note=note,
                        channel=idx,
                        velocity=velocity,
                        time=time_in_ticks - previous_time,
                    )
                )
                track_idx[instrument] = (track, time_in_ticks, idx)
            else:  # offset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    # shouldn't happen because we should have a corresponding onset
                    if debug:
                        print("IGNORING bad offset")

                    continue

                track.append(
                    mido.Message(
                        "note_off",
                        note=note,
                        channel=idx,
                        time=time_in_ticks - previous_time,
                    )
                )
                track_idx[instrument] = (track, time_in_ticks, idx)

    return mid


def compound_to_events(tokens, stats=False, include_velocity=False):
    """
    Convert a compound sequence to a sequence of events.

    In the sequence of events, for each note, we have (time, duration, note, velocity)
    """
    assert len(tokens) % 5 == 0
    tokens = tokens.copy()

    # remove velocities
    velocities = tokens[4::5]

    del tokens[4::5]

    # combine (note, instrument)
    assert all(-1 <= tok < 2**7 for tok in tokens[2::4])
    assert all(-1 <= tok < 129 for tok in tokens[3::4])
    tokens[2::4] = [
        SEP if note == -1 else MAX_PITCH * instr + note
        for note, instr in zip(tokens[2::4], tokens[3::4], strict=True)
    ]
    tokens[2::4] = [NOTE_OFFSET + tok for tok in tokens[2::4]]

    if include_velocity:
        tokens[3::4] = [VELOCITY_OFFSET + tok for tok in velocities[0::1]]

        # max duration cutoff and set unknown durations to 250ms
        truncations = sum([1 for tok in tokens[1::4] if tok >= MAX_DUR])
        tokens[1::4] = [
            TIME_RESOLUTION // 4 if tok == -1 else min(tok, MAX_DUR - 1)
            for tok in tokens[1::4]
        ]
        tokens[1::4] = [DUR_OFFSET + tok for tok in tokens[1::4]]

        assert min(tokens[0::4]) >= 0
        tokens[0::4] = [TIME_OFFSET + tok for tok in tokens[0::4]]

        assert len(tokens) % 4 == 0
    else:
        del tokens[3::4]

        # max duration cutoff and set unknown durations to 250ms
        truncations = sum([1 for tok in tokens[1::3] if tok >= MAX_DUR])
        tokens[1::3] = [
            TIME_RESOLUTION // 4 if tok == -1 else min(tok, MAX_DUR - 1)
            for tok in tokens[1::3]
        ]
        tokens[1::3] = [DUR_OFFSET + tok for tok in tokens[1::3]]

        assert min(tokens[0::3]) >= 0
        tokens[0::3] = [TIME_OFFSET + tok for tok in tokens[0::3]]

        assert len(tokens) % 3 == 0

    if stats:
        return tokens, truncations

    return tokens


def events_to_compound(tokens, debug=False, include_velocity=False):
    tokens = unpad(tokens, include_velocity=include_velocity)

    print("converting event to compound", len(tokens))

    tokens_per_event = 3
    if include_velocity:
        tokens_per_event = 4

    # move all velocity tokens to zero-offset
    tokens = [
        tok - VELOCITY_OFFSET if tok >= VELOCITY_OFFSET else tok for tok in tokens
    ]

    # move all tokens to zero-offset for synthesis
    tokens = [
        tok - ATIME_OFFSET if tok >= ATIME_OFFSET and tok != SEP else tok
        for tok in tokens
    ]

    # remove type offsets
    tokens[0::tokens_per_event] = [
        tok - TIME_OFFSET if tok != SEP else tok for tok in tokens[0::tokens_per_event]
    ]
    tokens[1::tokens_per_event] = [
        tok - DUR_OFFSET if tok != SEP else tok for tok in tokens[1::tokens_per_event]
    ]
    tokens[2::tokens_per_event] = [
        tok - NOTE_OFFSET if tok != SEP else tok for tok in tokens[2::tokens_per_event]
    ]

    offset = 0  # add max time from previous track for synthesis
    track_max = 0  # keep track of max time in track

    if include_velocity:
        for j, (time, dur, note) in enumerate(
            zip(tokens[0::4], tokens[1::4], tokens[2::4], strict=True)
        ):
            if note == SEP:
                offset += track_max
                track_max = 0
                if debug:
                    print("Sequence Boundary")
            else:
                track_max = max(track_max, time + dur)
                tokens[4 * j] += offset
    else:
        for j, (time, dur, note) in enumerate(
            zip(tokens[0::3], tokens[1::3], tokens[2::3], strict=True)
        ):
            if note == SEP:
                offset += track_max
                track_max = 0
                if debug:
                    print("Sequence Boundary")
            else:
                track_max = max(track_max, time + dur)
                tokens[3 * j] += offset

    # strip sequence SEPs
    assert len([tok for tok in tokens if tok == SEP]) % tokens_per_event == 0
    tokens = [tok for tok in tokens if tok != SEP]

    assert len(tokens) % tokens_per_event == 0
    out = 5 * (len(tokens) // tokens_per_event) * [0]
    out[0::5] = tokens[0::tokens_per_event]
    out[1::5] = tokens[1::tokens_per_event]
    out[2::5] = [tok - (2**7) * (tok // 2**7) for tok in tokens[2::tokens_per_event]]
    out[3::5] = [tok // 2**7 for tok in tokens[2::tokens_per_event]]

    print("out", out)

    if include_velocity:
        # Constrain velocity values to valid MIDI range (0-127)
        out[4::5] = [min(max(tok, 0), 127) for tok in tokens[3::4]]
    else:
        out[4::5] = (len(tokens) // 3) * [72]  # default velocity

    assert max(out[1::5]) < MAX_DUR
    assert max(out[2::5]) < MAX_PITCH
    assert max(out[3::5]) < MAX_INSTR
    assert all(tok >= 0 for tok in out)

    return out


def events_to_midi(tokens, debug=False, include_velocity=False):
    """
    Convert a sequence of events to a MIDI file.

    In the sequence of events, for each note, we have (time, duration, note, velocity)
    """
    return compound_to_midi(
        events_to_compound(tokens, debug=debug, include_velocity=include_velocity),
        debug=debug,
    )


def midi_to_events(midifile, debug=False, include_velocity=False):
    """
    Convert a MIDI file to a sequence of events.

    In the sequence of events, for each note, we have (time, duration, note, velocity)
    """
    return compound_to_events(
        midi_to_compound(midifile, debug=debug), include_velocity=include_velocity
    )
