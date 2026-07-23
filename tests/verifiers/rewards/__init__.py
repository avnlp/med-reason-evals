"""Tests for verifiers reward functions.

Reward functions tested here are used within the verifiers framework to score
model outputs during evaluation. Key functions:
    - multiple_choice_accuracy: Binary reward for MCQ tasks
    - llm_as_judge: LLM-based judgment for open-ended medical reasoning

These reward functions are designed for training-time evaluation and support
features like judge feedback tracking and template-based prompting.
"""
