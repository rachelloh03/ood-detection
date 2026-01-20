"""Tests to verify that all Jupyter notebooks can execute without errors."""

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

project_root = Path(__file__).parent.parent


def find_notebooks():
    """Find all Jupyter notebooks in the project."""
    notebooks = []
    for notebook_path in project_root.rglob("*.ipynb"):

        if any(part.startswith(".") for part in notebook_path.parts):
            continue
        notebooks.append(notebook_path)
    return sorted(notebooks)


@pytest.mark.parametrize("notebook_path", find_notebooks())
def test_notebook_execution(notebook_path):
    """Test that a notebook can be executed without errors.

    Args:
        notebook_path: Path to the notebook file
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(
        timeout=600,
        kernel_name="python3",
        allow_errors=False,
    )

    try:
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
    except Exception as e:
        pytest.fail(
            f"Notebook {notebook_path.relative_to(project_root)} failed to execute: {str(e)}"
        )
