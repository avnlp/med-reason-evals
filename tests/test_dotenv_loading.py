"""Tests for .env loading in the med_reason_evals package."""

from __future__ import annotations

import importlib
import os

import pytest

import med_reason_evals


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY must be set from a .env file",
)
def test_loads_dotenv_groq_api_key(tmp_path, monkeypatch):
    """Ensure .env files populate GROQ_API_KEY on import."""
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=dotenv-key\n")

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    importlib.reload(med_reason_evals)

    assert os.getenv("GROQ_API_KEY") == "dotenv-key"
