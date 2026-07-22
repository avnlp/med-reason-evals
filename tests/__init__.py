"""Test suite for med_reason_evals.

This package contains comprehensive tests for the medical reasoning evaluation
framework, including:
    - Data loading and transformation tests
    - Utility function tests (text normalization, parsing, extraction)
    - Verifiers integration tests (environments, evaluators, reward functions)
    - Verl integration tests (rollouts, reward functions for RL training)

Test Organization:
    - conftest.py: Shared pytest fixtures and mock clients
    - data/: Tests for HuggingFace dataset loading classes
    - utils/: Tests for text processing and parsing utilities
    - verifiers/: Tests for environment-based evaluation framework
    - verl/: Tests for RL training reward functions

All tests use mocked API clients to avoid network dependencies during testing.
"""
