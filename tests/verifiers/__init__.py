"""Tests for verifiers evaluation framework integration.

The verifiers framework provides environment-based evaluation for medical LLMs.
Tests cover:
    - Environment construction (SingleTurnEnv, MultiTurnEnv, ToolEnv)
    - Evaluator classes for each medical dataset
    - Reward functions for different evaluation modes
    - End-to-end evaluation smoke tests

Key test patterns:
    - Mocking HuggingFace datasets to avoid network calls
    - Verifying parser/rubric configuration matches expected format
    - Testing both XML and boxed answer formats
"""
