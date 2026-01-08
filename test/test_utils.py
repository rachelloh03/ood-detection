"""Tests for utils."""

import pytest
from src.constants import ATIME_OFFSET
from src.utils.readable_tokens import pitch_to_note, readable_tokens


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
    readable_repr = readable_tokens(tokens, include_velocity=False)
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
        55030,
    ]
    readable_repr = readable_tokens(tokens, include_velocity=True)
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


if __name__ == "__main__":
    test_readable_tokens_no_vel()
