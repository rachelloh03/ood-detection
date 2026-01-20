"""
Utilities for operating on encoded Midi sequences.

Taken from AMT implementation.
"""

from collections import defaultdict

from constants.token_constants import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    AVELOCITY_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    SEP,
    AR,
    AAR,
    TIME_OFFSET,
    VELOCITY_OFFSET,
    TIME_RESOLUTION,
    INCLUDE_VELOCITY,
)

from constants.model_constants import DELTA


def print_tokens(tokens, include_velocity=False):
    print("---------------------")
    if include_velocity:
        for j, (tm, dur, note, vel) in enumerate(
            zip(tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True)
        ):
            if note == SEP:
                assert tm == SEP and dur == SEP
                print(j, "SEP")
                continue

            if note == REST:
                assert tm < ATIME_OFFSET
                assert dur == DUR_OFFSET + 0
                print(j, tm, "REST")
                continue

            if note < ATIME_OFFSET:
                tm = tm - TIME_OFFSET
                dur = dur - DUR_OFFSET
                note = note - NOTE_OFFSET
                instr = note // 2**7
                pitch = note - (2**7) * instr
                vel = vel - VELOCITY_OFFSET
                print(j, tm, dur, instr, pitch, vel)
            else:
                tm = tm - ATIME_OFFSET
                dur = dur - ADUR_OFFSET
                note = note - ANOTE_OFFSET
                instr = note // 2**7
                pitch = note - (2**7) * instr
                vel = vel - AVELOCITY_OFFSET
                print(j, tm, dur, instr, pitch, vel, "(A)")
    else:
        for j, (tm, dur, note) in enumerate(
            zip(tokens[0::3], tokens[1::3], tokens[2::3], strict=True)
        ):
            if note == SEP:
                assert tm == SEP and dur == SEP
                print(j, "SEP")
                continue

            if note == REST:
                assert tm < ATIME_OFFSET
                assert dur == DUR_OFFSET + 0
                print(j, tm, "REST")
                continue

            if note < ATIME_OFFSET:
                tm = tm - TIME_OFFSET
                dur = dur - DUR_OFFSET
                note = note - NOTE_OFFSET
                instr = note // 2**7
                pitch = note - (2**7) * instr
                print(j, tm, dur, instr, pitch)
            else:
                tm = tm - ATIME_OFFSET
                dur = dur - ADUR_OFFSET
                note = note - ANOTE_OFFSET
                instr = note // 2**7
                pitch = note - (2**7) * instr
                print(j, tm, dur, instr, pitch, "(A)")


# changed
def clip(tokens, start, end, clip_duration=True, seconds=True, include_velocity=False):
    if seconds:
        start = int(TIME_RESOLUTION * start)
        end = int(TIME_RESOLUTION * end)

    new_tokens = []
    if include_velocity:
        for time, dur, note, vel in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            if note < ATIME_OFFSET:
                this_time = time - TIME_OFFSET
                this_dur = dur - DUR_OFFSET
            else:
                this_time = time - ATIME_OFFSET
                this_dur = dur - ADUR_OFFSET

            if this_time < start or end < this_time:
                continue

            # truncate extended notes
            if clip_duration and end < this_time + this_dur:
                dur -= this_time + this_dur - end

            new_tokens.extend([time, dur, note, vel])

        return new_tokens
    else:
        for time, dur, note in zip(
            tokens[0::3], tokens[1::3], tokens[2::3], strict=True
        ):
            if note < ATIME_OFFSET:
                this_time = time - TIME_OFFSET
                this_dur = dur - DUR_OFFSET
            else:
                this_time = time - ATIME_OFFSET
                this_dur = dur - ADUR_OFFSET

            if this_time < start or end < this_time:
                continue

            # truncate extended notes
            if clip_duration and end < this_time + this_dur:
                dur -= this_time + this_dur - end

            new_tokens.extend([time, dur, note])

        return new_tokens


# changed
def mask(tokens, start, end, include_velocity=False):
    new_tokens = []
    if include_velocity:
        for time, dur, note, vel in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            if note < ATIME_OFFSET:
                this_time = (time - TIME_OFFSET) / float(TIME_RESOLUTION)
            else:
                this_time = (time - ATIME_OFFSET) / float(TIME_RESOLUTION)

            if start < this_time < end:
                continue

            new_tokens.extend([time, dur, note, vel])

        return new_tokens
    else:
        for time, dur, note in zip(
            tokens[0::3], tokens[1::3], tokens[2::3], strict=True
        ):
            if note < ATIME_OFFSET:
                this_time = (time - TIME_OFFSET) / float(TIME_RESOLUTION)
            else:
                this_time = (time - ATIME_OFFSET) / float(TIME_RESOLUTION)

            if start < this_time < end:
                continue

            new_tokens.extend([time, dur, note])

        return new_tokens


# changed
def delete(tokens, criterion, include_velocity=False):
    new_tokens = []
    if include_velocity:
        for token in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            if criterion(token):
                continue

            new_tokens.extend(token)

        return new_tokens

    else:
        for token in zip(tokens[0::3], tokens[1::3], tokens[2::3], strict=True):
            if criterion(token):
                continue

            new_tokens.extend(token)

        return new_tokens


# changed
def sort(tokens, include_velocity=False):
    """sort sequence of events or controls (but not both)"""

    tokens_per_event = 3
    if include_velocity:
        tokens_per_event = 4

    times = tokens[0::tokens_per_event]
    indices = sorted(range(len(times)), key=times.__getitem__)

    sorted_tokens = []
    for idx in indices:
        sorted_tokens.extend(
            tokens[tokens_per_event * idx : tokens_per_event * (idx + 1)]
        )

    return sorted_tokens


# changed
def split(tokens, include_velocity=False):
    """split a sequence into events and controls"""

    events = []
    controls = []

    if include_velocity:
        for time, dur, note, vel in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            if note < ATIME_OFFSET:
                events.extend([time, dur, note, vel])
            else:
                controls.extend([time, dur, note, vel])
    else:
        for time, dur, note in zip(
            tokens[0::3], tokens[1::3], tokens[2::3], strict=True
        ):
            if note < ATIME_OFFSET:
                events.extend([time, dur, note])
            else:
                controls.extend([time, dur, note])

    return events, controls


# changed
def pad(tokens, end_time=None, density=TIME_RESOLUTION, include_velocity=False):
    end_time = TIME_OFFSET + (end_time if end_time else max_time(tokens, seconds=False))
    new_tokens = []
    previous_time = TIME_OFFSET + 0
    if include_velocity:
        for time, dur, note, vel in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            # must pad before separation, anticipation
            assert note < ATIME_OFFSET

            # insert pad tokens to ensure the desired density
            while time > previous_time + density:
                new_tokens.extend(
                    [previous_time + density, DUR_OFFSET + 0, REST, VELOCITY_OFFSET]
                )
                previous_time += density

            new_tokens.extend([time, dur, note, vel])
            previous_time = time

        while end_time > previous_time + density:
            new_tokens.extend(
                [previous_time + density, DUR_OFFSET + 0, REST, VELOCITY_OFFSET]
            )
            previous_time += density

        return new_tokens
    else:
        for time, dur, note in zip(
            tokens[0::3], tokens[1::3], tokens[2::3], strict=True
        ):
            # must pad before separation, anticipation
            assert note < ATIME_OFFSET

            # insert pad tokens to ensure the desired density
            while time > previous_time + density:
                new_tokens.extend([previous_time + density, DUR_OFFSET + 0, REST])
                previous_time += density

            new_tokens.extend([time, dur, note])
            previous_time = time

        while end_time > previous_time + density:
            new_tokens.extend([previous_time + density, DUR_OFFSET + 0, REST])
            previous_time += density

        return new_tokens


def unpad(tokens, include_velocity=INCLUDE_VELOCITY):
    new_tokens = []
    if tokens[0] in [SEP, REST, AR, AAR]:
        tokens = tokens[1:]

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


# changed
def anticipate(events, controls, delta=DELTA * TIME_RESOLUTION, include_velocity=False):
    """
    Interleave a sequence of events with anticipated controls.

    Inputs:
      events   : a sequence of events
      controls : a sequence of time-localized controls
      delta    : the anticipation interval

    Returns:
      tokens   : interleaved events and anticipated controls
      controls : unconsumed controls (control time > max_time(events) + delta)
    """

    if len(controls) == 0:
        return events, controls

    tokens = []
    event_time = 0
    control_time = controls[0] - ATIME_OFFSET

    if include_velocity:
        for time, dur, note, vel in zip(
            events[0::4], events[1::4], events[2::4], events[3::4], strict=True
        ):
            while event_time >= control_time - delta:
                tokens.extend(controls[0:4])
                controls = controls[4:]  # consume this control
                control_time = (
                    controls[0] - ATIME_OFFSET if len(controls) > 0 else float("inf")
                )

            assert note < ATIME_OFFSET
            event_time = time - TIME_OFFSET
            tokens.extend([time, dur, note, vel])
    else:
        for time, dur, note in zip(
            events[0::3], events[1::3], events[2::3], strict=True
        ):
            while event_time >= control_time - delta:
                tokens.extend(controls[0:3])
                controls = controls[3:]  # consume this control
                control_time = (
                    controls[0] - ATIME_OFFSET if len(controls) > 0 else float("inf")
                )

            assert note < ATIME_OFFSET
            event_time = time - TIME_OFFSET
            tokens.extend([time, dur, note])

    return tokens, controls


# changed
def sparsity(tokens, include_velocity=False):
    max_dt = 0
    previous_time = TIME_OFFSET + 0

    if include_velocity:
        for time, _, note, _ in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            if note == SEP:
                continue
            assert note < ATIME_OFFSET  # don't operate on interleaved sequences

            max_dt = max(max_dt, time - previous_time)
            previous_time = time

        return max_dt

    else:

        for time, _, note in zip(tokens[0::3], tokens[1::3], tokens[2::3], strict=True):
            if note == SEP:
                continue
            assert note < ATIME_OFFSET  # don't operate on interleaved sequences

            max_dt = max(max_dt, time - previous_time)
            previous_time = time

        return max_dt


# changed
def min_time(tokens, seconds=True, instr=None, include_velocity=False):
    mt = None

    if include_velocity:
        for time, _, note, _ in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            # stop calculating at sequence SEP
            if note == SEP:
                break

            if note < ATIME_OFFSET:
                time -= TIME_OFFSET
                note -= NOTE_OFFSET
            else:
                time -= ATIME_OFFSET
                note -= ANOTE_OFFSET

            # min time of a particular instrument
            if instr is not None and instr != note // 2**7:
                continue

            mt = time if mt is None else min(mt, time)

        if mt is None:
            mt = 0
        return mt / float(TIME_RESOLUTION) if seconds else mt
    else:
        for time, _, note in zip(tokens[0::3], tokens[1::3], tokens[2::3], strict=True):
            # stop calculating at sequence SEP
            if note == SEP:
                break

            if note < ATIME_OFFSET:
                time -= TIME_OFFSET
                note -= NOTE_OFFSET
            else:
                time -= ATIME_OFFSET
                note -= ANOTE_OFFSET

            # min time of a particular instrument
            if instr is not None and instr != note // 2**7:
                continue

            mt = time if mt is None else min(mt, time)

        if mt is None:
            mt = 0
        return mt / float(TIME_RESOLUTION) if seconds else mt


# changed
def max_time(tokens, seconds=True, instr=None, include_velocity=False):
    mt = 0

    tokens_per_event = 3  # time, dur, note
    if include_velocity:
        tokens_per_event = 4  # time, dur, note, velocity

    assert (
        len(tokens) % tokens_per_event == 0
    ), "Number of tokens is not a multiple of 3 (if no velocity) or 4 (if velocity included)"

    for time, _, note in zip(
        tokens[0::tokens_per_event],
        tokens[1::tokens_per_event],
        tokens[2::tokens_per_event],
        strict=True,
    ):
        # keep checking for max_time, even if it appears after a SEP
        # (this is important because we use this check for vocab overflow in tokenization)
        if note == SEP:
            continue

        if note < ATIME_OFFSET:
            time -= TIME_OFFSET
            note -= NOTE_OFFSET
        else:
            time -= ATIME_OFFSET
            note -= ANOTE_OFFSET

        # max time of a particular instrument
        if instr is not None and instr != note // 2**7:
            continue

        mt = max(mt, time)

    return mt / float(TIME_RESOLUTION) if seconds else mt


# changed
def get_instruments(tokens, include_velocity=False):
    instruments = defaultdict(int)

    if include_velocity:
        for _, _, note, _ in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            if note >= SEP:
                continue

            if note < ATIME_OFFSET:
                note -= NOTE_OFFSET
            else:
                note -= ANOTE_OFFSET

            instr = note // 2**7
            instruments[instr] += 1

        return instruments
    else:
        for _, _, note in zip(tokens[0::3], tokens[1::3], tokens[2::3], strict=True):
            if note >= SEP:
                continue

            if note < ATIME_OFFSET:
                note -= NOTE_OFFSET
            else:
                note -= ANOTE_OFFSET

            instr = note // 2**7
            instruments[instr] += 1

        return instruments


# changed
def translate(tokens, dt, seconds=False, include_velocity=False):
    if seconds:
        dt = int(TIME_RESOLUTION * dt)

    new_tokens = []
    print("translating", tokens[:100])
    if include_velocity:
        for time, dur, note, vel in zip(
            tokens[0::4], tokens[1::4], tokens[2::4], tokens[3::4], strict=True
        ):
            # stop translating after EOT
            if note == SEP:
                new_tokens.extend([time, dur, note, vel])
                dt = 0
                continue

            if note < ATIME_OFFSET:
                this_time = time - TIME_OFFSET
            else:
                this_time = time - ATIME_OFFSET

            assert 0 <= this_time + dt
            new_tokens.extend([time + dt, dur, note, vel])

        return new_tokens

    else:
        for time, dur, note in zip(
            tokens[0::3], tokens[1::3], tokens[2::3], strict=True
        ):
            # stop translating after EOT
            if note == SEP:
                new_tokens.extend([time, dur, note])
                dt = 0
                continue

            if note < ATIME_OFFSET:
                this_time = time - TIME_OFFSET
            else:
                this_time = time - ATIME_OFFSET

            assert 0 <= this_time + dt
            new_tokens.extend([time + dt, dur, note])

        return new_tokens


# changed
def combine(events, controls, include_velocity=False):
    def remove_control(tok):
        if tok >= AVELOCITY_OFFSET:
            return tok - AVELOCITY_OFFSET + VELOCITY_OFFSET
        return tok - ATIME_OFFSET

    processed_controls = [remove_control(tok) for tok in controls]
    return sort(events + processed_controls, include_velocity=include_velocity)
