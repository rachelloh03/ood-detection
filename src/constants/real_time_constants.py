"""
Constants related to the real-time OOD detection.
"""

from constants.token_constants import INCLUDE_VELOCITY

SLIDING_WINDOW_LEN = 40
STRIDE = 10
TOKENS_PER_EVENT = 4 if INCLUDE_VELOCITY else 3
