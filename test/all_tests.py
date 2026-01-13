"""
Run all tests using pytest.

This script provides a convenient entry point to run all tests.
You can also run tests directly using: pytest
"""

import pytest
import sys

if __name__ == "__main__":
    exit_code = pytest.main(sys.argv[1:])
    sys.exit(exit_code)
