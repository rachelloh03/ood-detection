"""
API functions for sampling from anticipatory infilling models.

All taken from AMT implementation.
"""

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from constants.model_constants import DELTA, DEVICE
from constants.token_constants import (
    MAX_DUR,
    MAX_INSTR,
    MAX_NOTE,
    MAX_TIME,
    MAX_PITCH,
    TIME_RESOLUTION,
    ADUR_OFFSET,
    ANOTE_OFFSET,
    AAR,
    ATIME_OFFSET,
    AR,
    AVELOCITY_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
    VELOCITY_OFFSET,
    MAX_VELOCITY,
    INCLUDE_VELOCITY,
    SEP,
)

from utils.ops import min_time, max_time, pad, clip, print_tokens, sort, split, unpad


def safe_logits(logits, idx, include_velocity=False):
    logits[ATIME_OFFSET:SEP] = -float("inf")
    logits[REST] = -float("inf")
    if include_velocity:
        logits[SEP:VELOCITY_OFFSET] = -float(
            "inf"
        )  # don't generate special tokens but still allow velocities
        logits[VELOCITY_OFFSET + MAX_VELOCITY :] = -float(
            "inf"
        )  # don't generate velocities above 127
        # don't generate stuff in the wrong time slot
        if idx % 4 == 0:
            logits[DUR_OFFSET : DUR_OFFSET + MAX_DUR] = -float("inf")
            logits[NOTE_OFFSET : NOTE_OFFSET + MAX_NOTE] = -float("inf")
            logits[VELOCITY_OFFSET:] = -float("inf")
        elif idx % 4 == 1:
            logits[TIME_OFFSET : TIME_OFFSET + MAX_TIME] = -float("inf")
            logits[NOTE_OFFSET : NOTE_OFFSET + MAX_NOTE + 1] = -float("inf")
            logits[VELOCITY_OFFSET:] = -float("inf")
        elif idx % 4 == 2:
            logits[TIME_OFFSET : TIME_OFFSET + MAX_TIME] = -float("inf")
            logits[DUR_OFFSET : DUR_OFFSET + MAX_DUR] = -float("inf")
            logits[VELOCITY_OFFSET:] = -float("inf")
        elif idx % 4 == 3:
            logits[:VELOCITY_OFFSET] = -float("inf")
    else:
        logits[SEP:] = -float("inf")  # don't generate special tokens and velocities
        # don't generate stuff in the wrong time slot
        if idx % 3 == 0:
            logits[DUR_OFFSET : DUR_OFFSET + MAX_DUR] = -float("inf")
            logits[NOTE_OFFSET : NOTE_OFFSET + MAX_NOTE] = -float("inf")
        elif idx % 3 == 1:
            logits[TIME_OFFSET : TIME_OFFSET + MAX_TIME] = -float("inf")
            logits[NOTE_OFFSET : NOTE_OFFSET + MAX_NOTE] = -float("inf")
        elif idx % 3 == 2:
            logits[TIME_OFFSET : TIME_OFFSET + MAX_TIME] = -float("inf")
            logits[DUR_OFFSET : DUR_OFFSET + MAX_DUR] = -float("inf")

    return logits


def nucleus(logits, top_p):
    # from HF implementation
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("inf")

    return logits


def future_logits(logits, curtime):
    """don't sample events in the past"""
    if curtime > 0:
        logits[TIME_OFFSET : TIME_OFFSET + curtime] = -float("inf")

    return logits


def instr_logits(logits, full_history, include_velocity=False):
    """don't sample more than 16 instruments"""
    # instrs = get_instruments(full_history, include_velocity=include_velocity)
    # if len(instrs) < 15:  # 16 - 1 to account for the reserved drum track
    #     return logits
    instrs = [0]

    for instr in range(MAX_INSTR):
        if instr not in instrs:
            logits[
                NOTE_OFFSET + instr * MAX_PITCH : NOTE_OFFSET + (instr + 1) * MAX_PITCH
            ] = -float("inf")

    return logits


# changed
def add_token(
    model, z, tokens, top_p, current_time, debug=False, include_velocity=False
):
    if include_velocity:
        # will add 4 new tokens
        return add_token_with_velocity(model, z, tokens, top_p, current_time, debug)
    else:
        # will add 3 new tokens
        return add_token_without_velocity(model, z, tokens, top_p, current_time, debug)


def add_token_with_velocity(model, z, tokens, top_p, current_time, debug=False):
    assert len(tokens) % 4 == 0, "Tokens length must be divisible by 4"

    history = tokens.copy()
    lookback = max(len(tokens) - 1016, 0)
    history = history[lookback:]  # Markov window
    offset = min_time(history, seconds=False, include_velocity=True)
    history[::4] = [
        tok - offset for tok in history[::4]
    ]  # relativize time in the history buffer

    new_token = []
    with torch.no_grad():
        for i in range(4):
            input_tokens = (
                torch.tensor(z + history + new_token).unsqueeze(0).to(model.device)
            )
            logits = model(input_tokens).logits[0, -1]

            idx = input_tokens.shape[1] - 1
            logits = safe_logits(logits, idx, include_velocity=True)
            if i == 0:
                logits = future_logits(logits, current_time - offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)
            logits = nucleus(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            new_token.append(int(token))

    new_token[0] += offset  # revert to full sequence timing
    if debug:
        print(f"  OFFSET = {offset}, LEN = {len(history)}, TIME = {tokens[::4][-5:]}")

    return new_token


def add_token_without_velocity(model, z, tokens, top_p, current_time, debug=False):
    assert len(tokens) % 3 == 0, "Tokens length must be divisible by 3"

    history = tokens.copy()
    lookback = max(len(tokens) - 1017, 0)
    history = history[lookback:]  # Markov window
    offset = min_time(history, seconds=False, include_velocity=False)
    history[::3] = [
        tok - offset for tok in history[::3]
    ]  # relativize time in the history buffer

    new_token = []
    with torch.no_grad():
        for i in range(3):
            input_tokens = (
                torch.tensor(z + history + new_token).unsqueeze(0).to(model.device)
            )
            logits = model(input_tokens).logits[0, -1]
            idx = input_tokens.shape[1] - 1
            logits = safe_logits(logits, idx)

            if i == 0:
                logits = future_logits(logits, current_time - offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)
            logits = nucleus(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            # print highest probability tokens
            # print("highest probability tokens", torch.topk(probs, 10))
            token = torch.multinomial(probs, 1)
            # greedy
            # token = torch.argmax(probs)
            new_token.append(int(token))

    new_token[0] += offset  # revert to full sequence timing
    if debug:
        print(f"  OFFSET = {offset}, LEN = {len(history)}, TIME = {tokens[::3][-5:]}")

    return new_token


# changed
def generate(
    model,
    start_time,
    end_time,
    inputs=None,
    controls=None,
    top_p=1.0,
    debug=False,
    delta=DELTA * TIME_RESOLUTION,
    include_velocity=False,
):
    if inputs is None:
        inputs = []

    if controls is None:
        controls = []

    start_time = int(TIME_RESOLUTION * start_time)
    end_time = int(TIME_RESOLUTION * end_time)

    print("model", model)
    # prompt is events up to start_time
    prompt = pad(
        clip(
            inputs,
            0,
            start_time,
            clip_duration=False,
            seconds=False,
            include_velocity=include_velocity,
        ),
        start_time,
        include_velocity=include_velocity,
    )

    # treat events beyond start_time as controls
    future = clip(
        inputs,
        start_time + 1,
        max_time(inputs, seconds=False, include_velocity=include_velocity),
        clip_duration=False,
        seconds=False,
        include_velocity=include_velocity,
    )
    if debug:
        print("Future")
        print_tokens(future, include_velocity=include_velocity)

    # clip controls that preceed the sequence
    controls = clip(
        controls,
        DELTA,
        max_time(controls, seconds=False, include_velocity=include_velocity),
        clip_duration=False,
        seconds=False,
        include_velocity=include_velocity,
    )

    if debug:
        print("Controls")
        print_tokens(controls, include_velocity=include_velocity)

    z = [AAR] if len(controls) > 0 or len(future) > 0 else [AR]
    if debug:
        print("AR Mode" if z[0] == AR else "AAR Mode")

    # interleave the controls with the events
    tokens, controls = AAR(
        prompt,
        sort(
            controls + [ATIME_OFFSET + token for token in future],
            include_velocity=include_velocity,
        ),
        include_velocity=include_velocity,
    )

    if debug:
        print("Prompt")
        print_tokens(tokens, include_velocity=include_velocity)

    current_time = max_time(prompt, seconds=False, include_velocity=include_velocity)
    if debug:
        print("Current time:", current_time)

    with tqdm(range(end_time - start_time)) as progress:
        if controls:
            atime, adur, anote = controls[0:3]
            if include_velocity:
                avel = controls[3]
                AARd_tokens = controls[4:]
            else:
                AARd_tokens = controls[3:]
            AARd_time = atime - ATIME_OFFSET
        else:
            # nothing to AAR
            AARd_time = math.inf

        while True:
            while current_time >= AARd_time - delta:
                if include_velocity:
                    tokens.extend([atime, adur, anote, avel])
                else:
                    tokens.extend([atime, adur, anote])
                if debug:
                    note = anote - ANOTE_OFFSET
                    instr = note // 2**7
                    if include_velocity:
                        print(
                            "A",
                            atime - ATIME_OFFSET,
                            adur - ADUR_OFFSET,
                            instr,
                            note - (2**7) * instr,
                            avel - AVELOCITY_OFFSET,
                        )
                    else:
                        print(
                            "A",
                            atime - ATIME_OFFSET,
                            adur - ADUR_OFFSET,
                            instr,
                            note - (2**7) * instr,
                        )

                if len(AARd_tokens) > 0:
                    atime, adur, anote = AARd_tokens[0:3]
                    if include_velocity:
                        avel = AARd_tokens[3]
                        AARd_tokens = AARd_tokens[4:]
                    else:
                        AARd_tokens = AARd_tokens[3:]
                    AARd_time = atime - ATIME_OFFSET
                else:
                    # nothing more to AAR
                    AARd_time = math.inf

            new_token = add_token(
                model,
                z,
                tokens,
                top_p,
                max(start_time, current_time),
                include_velocity=include_velocity,
            )

            if debug:
                print("current time", current_time)
                print("new_token", new_token)

            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note // 2**7
                new_pitch = new_note - (2**7) * new_instr
                if include_velocity:
                    new_vel = new_token[3] - VELOCITY_OFFSET
                    print(
                        "C",
                        new_time,
                        new_token[1] - DUR_OFFSET,
                        new_instr,
                        new_pitch,
                        new_vel,
                    )
                else:
                    print(
                        "C", new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch
                    )

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)

            # Periodic memory cleanup during long generation
            if len(tokens) % 100 == 0:
                torch.cuda.empty_cache()

    events, _ = split(tokens, include_velocity=include_velocity)
    return sort(
        unpad(events, include_velocity=include_velocity) + future,
        include_velocity=include_velocity,
    )


def generate_old(
    model,
    start_time,
    end_time,
    inputs=None,
    controls=None,
    top_p=1.0,
    debug=False,
    delta=DELTA * TIME_RESOLUTION,
):
    if inputs is None:
        inputs = []

    if controls is None:
        controls = []

    start_time = int(TIME_RESOLUTION * start_time)
    end_time = int(TIME_RESOLUTION * end_time)

    # prompt is events up to start_time
    prompt = pad(
        clip(inputs, 0, start_time, clip_duration=False, seconds=False), start_time
    )

    # treat events beyond start_time as controls
    future = clip(
        inputs,
        start_time + 1,
        max_time(inputs, seconds=False),
        clip_duration=False,
        seconds=False,
    )
    if debug:
        print("Future")
        print_tokens(future)

    # clip controls that preceed the sequence
    controls = clip(
        controls,
        DELTA,
        max_time(controls, seconds=False),
        clip_duration=False,
        seconds=False,
    )

    if debug:
        print("Controls")
        print_tokens(controls)

    z = [AAR] if len(controls) > 0 or len(future) > 0 else [AR]
    if debug:
        print("AR Mode" if z[0] == AR else "AAR Mode")

    # interleave the controls with the events
    tokens, controls = AAR(
        prompt, sort(controls + [ATIME_OFFSET + token for token in future])
    )

    if debug:
        print("Prompt")
        print_tokens(tokens)

    current_time = max_time(prompt, seconds=False)
    if debug:
        print("Current time:", current_time)

    with tqdm(range(end_time - start_time)) as progress:
        if controls:
            atime, adur, anote = controls[0:3]
            AARd_tokens = controls[3:]
            AARd_time = atime - ATIME_OFFSET
        else:
            # nothing to AAR
            AARd_time = math.inf

        while True:
            while current_time >= AARd_time - delta:
                tokens.extend([atime, adur, anote])
                if debug:
                    note = anote - ANOTE_OFFSET
                    instr = note // 2**7
                    print(
                        "A",
                        atime - ATIME_OFFSET,
                        adur - ADUR_OFFSET,
                        instr,
                        note - (2**7) * instr,
                    )

                if len(AARd_tokens) > 0:
                    atime, adur, anote = AARd_tokens[0:3]
                    AARd_tokens = AARd_tokens[3:]
                    AARd_time = atime - ATIME_OFFSET
                else:
                    # nothing more to AAR
                    AARd_time = math.inf

            if debug:
                print("current time", current_time)
                print("Adding token...")
            new_token = add_token(
                model,
                z,
                tokens,
                top_p,
                max(start_time, current_time),
                include_velocity=False,
            )
            if debug:
                print("new_token", new_token)
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note // 2**7
                new_pitch = new_note - (2**7) * new_instr
                print("C", new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch)

            tokens.extend(new_token)
            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time
            progress.update(dt)

    events, _ = split(tokens)
    return sort(unpad(events) + future)


# changed
def generate_ar(
    model,
    start_time,
    end_time,
    inputs=None,
    controls=None,
    top_p=1.0,
    debug=False,
    delta=DELTA * TIME_RESOLUTION,
    include_velocity=False,
):
    """
    Generate a sequence of events ARively.
    """
    if inputs is None:
        inputs = []

    if controls is None:
        controls = []
    else:
        # treat controls as ordinary tokens
        def remove_ATIME_OFFSET(token):
            if token >= VELOCITY_OFFSET:
                return token - AVELOCITY_OFFSET + VELOCITY_OFFSET
            if token >= ATIME_OFFSET:
                return token - ATIME_OFFSET
            return token

        controls = [remove_ATIME_OFFSET(token) for token in controls]

    start_time = int(TIME_RESOLUTION * start_time)
    end_time = int(TIME_RESOLUTION * end_time)

    inputs = sort(inputs + controls, include_velocity=include_velocity)

    # prompt is events up to start_time
    prompt = pad(
        clip(
            inputs,
            0,
            start_time,
            clip_duration=False,
            seconds=False,
            include_velocity=include_velocity,
        ),
        start_time,
        include_velocity=include_velocity,
    )
    if debug:
        print("Prompt")
        print_tokens(prompt, include_velocity=include_velocity)

    # treat events beyond start_time as controls
    controls = clip(
        inputs,
        start_time + 1,
        max_time(inputs, seconds=False, include_velocity=include_velocity),
        clip_duration=False,
        seconds=False,
        include_velocity=include_velocity,
    )
    if debug:
        print("Future")
        print_tokens(controls, include_velocity=include_velocity)

    z = [AR]
    if debug:
        print("AR Mode")

    current_time = max_time(prompt, seconds=False, include_velocity=include_velocity)
    if debug:
        print("Current time:", current_time)

    tokens = prompt
    with tqdm(range(end_time - start_time)) as progress:
        if controls:
            atime, adur, anote = controls[0:3]
            if include_velocity:
                avel = controls[3]
                AARd_tokens = controls[4:]
            else:
                AARd_tokens = controls[3:]
            AARd_time = atime - TIME_OFFSET
        else:
            # nothing to AAR
            AARd_time = math.inf

        while True:
            new_token = add_token(
                model,
                z,
                tokens,
                top_p,
                max(start_time, current_time),
                include_velocity=include_velocity,
            )
            new_time = new_token[0] - TIME_OFFSET
            if new_time >= end_time:
                break

            dt = new_time - current_time
            assert dt >= 0
            current_time = new_time

            # backfill anything that should have come before the new token
            while current_time >= AARd_time:
                if include_velocity:
                    tokens.extend([atime, adur, anote, avel])
                else:
                    tokens.extend([atime, adur, anote])
                if debug:
                    note = anote - NOTE_OFFSET
                    instr = note // 2**7
                    if include_velocity:
                        print(
                            "A",
                            atime - TIME_OFFSET,
                            adur - DUR_OFFSET,
                            instr,
                            note - (2**7) * instr,
                            avel - VELOCITY_OFFSET,
                        )
                    else:
                        print(
                            "A",
                            atime - TIME_OFFSET,
                            adur - DUR_OFFSET,
                            instr,
                            note - (2**7) * instr,
                        )

                if len(AARd_tokens) > 0:
                    atime, adur, anote = AARd_tokens[0:3]
                    if include_velocity:
                        avel = AARd_tokens[3]
                        AARd_tokens = AARd_tokens[4:]
                    else:
                        AARd_tokens = AARd_tokens[3:]
                    AARd_time = atime - TIME_OFFSET
                else:
                    # nothing more to AAR
                    AARd_time = math.inf

            if debug:
                new_note = new_token[2] - NOTE_OFFSET
                new_instr = new_note // 2**7
                new_pitch = new_note - (2**7) * new_instr
                if include_velocity:
                    new_vel = new_token[3] - VELOCITY_OFFSET
                    if new_vel > MAX_VELOCITY - 1:
                        new_vel -= MAX_VELOCITY
                    print(
                        "C",
                        new_time,
                        new_token[1] - DUR_OFFSET,
                        new_instr,
                        new_pitch,
                        new_vel,
                    )
                else:
                    print(
                        "C", new_time, new_token[1] - DUR_OFFSET, new_instr, new_pitch
                    )

            tokens.extend(new_token)
            progress.update(dt)

    if AARd_time != math.inf:
        if include_velocity:
            tokens.extend([atime, adur, anote, avel])
        else:
            tokens.extend([atime, adur, anote])

    return sort(
        unpad(tokens, include_velocity=include_velocity) + controls,
        include_velocity=include_velocity,
    )


def sample_from_model(model, input_sample):
    input_sample = {
        k: v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v
        for k, v in input_sample.items()
    }
    with torch.no_grad():
        output = model(
            **input_sample,
            output_hidden_states=False,
            use_cache=False,
        )
    return output


def generate_tokens(
    model,
    z,
    input_sequence,
    num_tokens,
    top_p=1.0,
    debug=False,
    include_velocity=INCLUDE_VELOCITY,
):
    """
    Generate a specified number of tokens from an input sequence.

    Args:
        model: The language model to use for generation
        z: Special token prefix (e.g., [AR] or [AAR])
        input_sequence: List of input tokens (must be divisible by 3 or 4)
        num_tokens: Number of tokens to generate (must be divisible by 3 or 4)
        top_p: Nucleus sampling parameter (default: 1.0)
        debug: Whether to print debug information
        include_velocity: Whether to include velocity tokens. If None, auto-detect from input_sequence

    Returns:
        List of generated tokens (input_sequence + newly generated tokens)
    """
    tokens = input_sequence.copy()

    if include_velocity is None:
        if len(tokens) % 4 == 0:
            include_velocity = True
        elif len(tokens) % 3 == 0:
            include_velocity = False
        else:
            raise ValueError(
                f"Input sequence length ({len(tokens)}) must be divisible by 3 or 4"
            )

    if include_velocity:
        assert (
            len(tokens) % 4 == 0
        ), "Input sequence length must be divisible by 4 when include_velocity=True"
        tokens_per_event = 4
    else:
        assert (
            len(tokens) % 3 == 0
        ), "Input sequence length must be divisible by 3 when include_velocity=False"
        tokens_per_event = 3

    assert num_tokens % tokens_per_event == 0, (
        f"num_tokens ({num_tokens}) must be divisible by {tokens_per_event} "
        f"when include_velocity={include_velocity}"
    )

    num_events_to_generate = num_tokens // tokens_per_event

    current_time = max_time(tokens, seconds=False, include_velocity=include_velocity)

    if debug:
        print(f"Generating {num_events_to_generate} events ({num_tokens} tokens)")
        print(f"Starting with {len(tokens)} tokens, current_time={current_time}")

    for event_idx in range(num_events_to_generate):
        new_token = add_token(
            model,
            z,
            tokens,
            top_p,
            current_time,
            debug=debug,
            include_velocity=include_velocity,
        )

        new_time = new_token[0] - TIME_OFFSET
        current_time = new_time

        tokens.extend(new_token)

        if debug:
            print(f"Generated event {event_idx + 1}/{num_events_to_generate}")

        if len(tokens) % 100 == 0:
            torch.cuda.empty_cache()

    return tokens
