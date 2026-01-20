"""Sanity checks for the dataset."""

from constants.token_constants import VOCAB_SIZE, INCLUDE_VELOCITY, MAX_VELOCITY
from utils.process_tokens import get_readable_events


def check_valid_input_ids(input_ids, include_velocity=INCLUDE_VELOCITY):
    """Check if the input_ids are valid."""
    if not isinstance(input_ids, list):
        raise ValueError("input_ids must be a list")
    if not all(isinstance(x, int) for x in input_ids):
        raise ValueError("input_ids must be a list of integers")
    if not all(x >= 0 for x in input_ids):
        raise ValueError("input_ids must be a list of non-negative integers")
    if not all(x < VOCAB_SIZE for x in input_ids):
        # write out the input_ids that are not valid
        invalid_ids = [x for x in input_ids if x >= VOCAB_SIZE]
        print(f"Invalid input_ids: {invalid_ids}")
        raise ValueError(
            f"input_ids must be a list of integers < VOCAB_SIZE ({VOCAB_SIZE}), found {invalid_ids}"
        )

    readable_events = get_readable_events(input_ids, include_velocity)
    if not all(isinstance(x, dict) for x in readable_events):
        raise ValueError("readable_events must be a list of dictionaries")
    for event in readable_events:
        if "special_token" in event:
            continue
        if "onset" in event:
            assert 0 <= event["onset"] <= 100, "onset must be between 0 and 100 seconds"
        if "duration" in event:
            assert (
                0 <= event["duration"] <= 10
            ), "duration must be between 0 and 10 seconds"
        if "note" in event:
            assert (
                0 <= event["note"] < MAX_VELOCITY
            ), f"note must be between 0 and MAX_VELOCITY - 1 ({MAX_VELOCITY - 1})"
        if "velocity" in event:
            assert (
                0 <= event["velocity"] < MAX_VELOCITY
            ), f"velocity must be between 0 and MAX_VELOCITY - 1 ({MAX_VELOCITY - 1})"
