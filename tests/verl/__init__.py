"""Tests for Verl RL training integration.

Verl (Value-based reinforcement learning) provides reward functions and rollout
infrastructure for training medical LLMs with RL. Tests cover:
    - Base evaluator classes (BaseMCQEvaluator, BaseJudgeEvaluator)
    - Dataset-specific evaluator implementations
    - GroqRollouts client for model inference
    - Reward functions compatible with verl's scoring interface

Unlike verifiers tests which focus on evaluation environments, these tests
focus on the training-time rollout and reward computation pipeline.
"""
