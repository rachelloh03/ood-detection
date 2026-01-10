"""
Token-related constants for the OOD detection project.
"""

# Special tokens
REST = 27512
SEP = 55025
AR = 55026
AAR = 55027

# Token offsets (not anticipated)
TIME_OFFSET = 0
DUR_OFFSET = 10000
NOTE_OFFSET = 11000
VELOCITY_OFFSET = 55028

# Token offsets (anticipated)
ATIME_OFFSET = 27513
ADUR_OFFSET = 37513
ANOTE_OFFSET = 38513
AVELOCITY_OFFSET = 55156

# Vocabulary settings
INCLUDE_VELOCITY = False
VOCAB_SIZE = VELOCITY_OFFSET + 256 if INCLUDE_VELOCITY else VELOCITY_OFFSET

# Note names
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
