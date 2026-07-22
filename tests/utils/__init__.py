"""Tests for utility modules.

Utility functions tested here are used across the evaluation framework for:
    - Text normalization (Unicode handling, whitespace, semantic normalization)
    - Answer extraction from various formats (XML tags, boxed notation)
    - Response parsing (JSON extraction, yes/no parsing, think tag stripping)

These utilities are critical for reliable answer comparison across different
model output formats.
"""
