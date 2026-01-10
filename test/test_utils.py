"""Tests for utils."""

import pytest
from src.constants.token_constants import ATIME_OFFSET
from src.utils.process_tokens import (
    pitch_to_note,
    get_readable_events,
    set_anticipated,
    set_instrument,
)


def test_readable_tokens_no_vel():
    """Test the tokens_to_text function."""
    assert pitch_to_note(60) == "C4"
    tokens = [
        55026,
        55025,
        55025,
        55025,
        0,
        10048,
        11060,
        50,
        10048,
        11060,
        100,
        10048,
        11067,
        150,
        10048,
        11067,
        200,
        10048,
        11069,
        250,
        10048,
        11069,
        300,
        10095,
        11067,
        400,
        10048,
        11065,
        450,
        10048,
        11065,
        500,
        10048,
        11064,
        550,
        10048,
        11064,
        600,
        10048,
        11062,
        650 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11062 + ATIME_OFFSET,
        700,
        10095,
        11060,
        800,
        10005,
        27512,
    ]
    readable_repr = get_readable_events(tokens, include_velocity=False)
    assert readable_repr[0]["special_token"] == "AR"
    assert readable_repr[1]["special_token"] == "SEP"
    assert readable_repr[2]["anticipated"] is False
    assert readable_repr[2]["onset"] == pytest.approx(0.0)
    assert readable_repr[2]["duration"] == pytest.approx(0.48)
    assert readable_repr[2]["instrument"] == 0
    assert readable_repr[2]["pitch"] == "C4"
    assert readable_repr[-3]["onset"] == pytest.approx(6.5)
    assert readable_repr[-3]["duration"] == pytest.approx(0.48)
    assert readable_repr[-3]["instrument"] == 0
    assert readable_repr[-3]["pitch"] == "D4"
    assert readable_repr[-3]["anticipated"]

    tokens = [
        55026,
        55025,
        55025,
        55025,
        55025,
        0,
        10048,
        11060,
        55029,
        50 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11062 + ATIME_OFFSET,
        55030 + 128,
    ]
    readable_repr = get_readable_events(tokens, include_velocity=True)
    assert readable_repr[0]["special_token"] == "AR"
    assert readable_repr[1]["special_token"] == "SEP"
    assert readable_repr[2]["anticipated"] is False
    assert readable_repr[2]["onset"] == pytest.approx(0.0)
    assert readable_repr[2]["duration"] == pytest.approx(0.48)
    assert readable_repr[2]["instrument"] == 0
    assert readable_repr[2]["pitch"] == "C4"
    assert readable_repr[2]["velocity"] == 1
    assert readable_repr[3]["anticipated"]
    assert readable_repr[3]["onset"] == pytest.approx(0.5)
    assert readable_repr[3]["duration"] == pytest.approx(0.48)
    assert readable_repr[3]["instrument"] == 0
    assert readable_repr[3]["pitch"] == "D4"
    assert readable_repr[3]["velocity"] == 2


def test_set_instrument():
    """Test the set_instrument function."""
    tokens = [
        55026,
        55025,
        55025,
        55025,
        0,
        10048,
        11060 + 3 * 128,
        50,
        10048,
        11060 + 4 * 128,
        100 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11062 + ATIME_OFFSET,
    ]
    expected_tokens = [
        55026,
        55025,
        55025,
        55025,
        0,
        10048,
        11060 + 5 * 128,
        50,
        10048,
        11060 + 5 * 128,
        100 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11062 + ATIME_OFFSET + 5 * 128,
    ]
    assert (
        set_instrument(tokens, 5) == expected_tokens
    ), f"Expected {expected_tokens} but got {set_instrument(tokens, 5)}"


def test_set_anticipated():
    """Test the set_anticipated function."""
    tokens = [
        55026,
        55025,
        55025,
        55025,
        0,
        10048,
        11060,
        50,
        10048,
        11060,
        100 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11062 + ATIME_OFFSET,
    ]
    expected_non_anticipated_tokens = [
        55026,
        55025,
        55025,
        55025,
        0,
        10048,
        11060,
        50,
        10048,
        11060,
        100,
        10048,
        11062,
    ]
    expected_anticipated_tokens = [
        55026,
        55025,
        55025,
        55025,
        0 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11060 + ATIME_OFFSET,
        50 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11060 + ATIME_OFFSET,
        100 + ATIME_OFFSET,
        10048 + ATIME_OFFSET,
        11062 + ATIME_OFFSET,
    ]
    assert (
        set_anticipated(tokens, True) == expected_anticipated_tokens
    ), f"Expected {expected_anticipated_tokens} but got {set_anticipated(tokens, True)}"
    assert (
        set_anticipated(tokens, False) == expected_non_anticipated_tokens
    ), f"Expected {expected_non_anticipated_tokens} but got {set_anticipated(tokens, False)}"


if __name__ == "__main__":
    test_readable_tokens_no_vel()
    test_set_instrument()
    test_set_anticipated()
