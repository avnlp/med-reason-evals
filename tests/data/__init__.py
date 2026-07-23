"""Tests for dataset loading modules.

Tests cover the BaseDataset abstract class and its implementations for various
medical datasets including MedQA, PubMedQA, HealthBench, and others.

Each dataset test verifies:
    - Dataset structure and required columns
    - Proper mapping to verifiers/verl formats
    - Streaming vs non-streaming behavior
    - Split handling (train/validation/test)
"""
