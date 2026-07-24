"""Tests for .env loading in the med_reason_evals package.

`load_dotenv()` is not called at package-import time (`med_reason_evals/__init__.py`
is a plain docstring with no side effects). Instead, each verifier script (e.g.
`medbullets.py`, `medqa.py`, `metamedqa.py`, `pubmedqa.py`) calls `load_dotenv()`
as the first statement of its own `main()`. This test exercises that real code
path via `medbullets.main()`, stopping execution right after the dotenv call
(by making the next statement - dataset construction - raise) so the test stays
fast and network-free while still verifying genuine behavior.
"""

from __future__ import annotations

import os

import pytest

from med_reason_evals.verifiers import medbullets


class _StopAfterDotenvError(Exception):
    """Sentinel used to halt `main()` right after `load_dotenv()` runs."""


def test_loads_dotenv_groq_api_key(tmp_path, monkeypatch):
    """Ensure `main()` populates GROQ_API_KEY from a .env file via load_dotenv()."""
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=dotenv-key\n")

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    def _raise(*args, **kwargs):
        raise _StopAfterDotenvError

    # MedBulletsDataset(...) is the statement immediately after load_dotenv()
    # in main(); short-circuiting it avoids a real HF dataset download while
    # still exercising the genuine load_dotenv() call.
    monkeypatch.setattr(medbullets, "MedBulletsDataset", _raise)

    with pytest.raises(_StopAfterDotenvError):
        medbullets.main()

    assert os.getenv("GROQ_API_KEY") == "dotenv-key"
