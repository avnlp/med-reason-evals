"""Tests for Verl reward functions.

Reward functions tested here conform to verl's scoring interface and are used
during RL training to provide feedback signals. Key functions:
    - compute_score (multiple_choice_accuracy): Binary scoring for MCQ tasks
    - compute_score (semantic_equivalence): Flexible matching for open answers

These functions are designed to be called by the verl framework during
training rollouts and must return normalized scores (typically 0.0 to 1.0).
"""
