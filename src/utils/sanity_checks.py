"""Sanity checks for the dataset."""

from constants import VOCAB_SIZE
from utils.readable_events import get_readable_events


def check_valid_input_ids(input_ids):
    """Check if the input_ids are valid."""
    if not isinstance(input_ids, list):
        raise ValueError("input_ids must be a list")
    if not all(isinstance(x, int) for x in input_ids):
        raise ValueError("input_ids must be a list of integers")
    if not all(x >= 0 for x in input_ids):
        raise ValueError("input_ids must be a list of non-negative integers")
    if not all(x < VOCAB_SIZE for x in input_ids):
        raise ValueError("input_ids must be a list of integers less than VOCAB_SIZE")

    readable_events = get_readable_events(input_ids)
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
            assert 0 <= event["note"] <= 127, "note must be between 0 and 127"
        if "velocity" in event:
            assert 0 <= event["velocity"] <= 127, "velocity must be between 0 and 127"
