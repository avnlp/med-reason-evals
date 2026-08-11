"""Medical reasoning evaluation framework integrating Verifiers and Verl ecosystems.

This package provides dataset loaders, evaluation harnesses, and reward functions
for benchmarking medical language models. The public API is intentionally flat to
provide stable entry points that won't break when internal modules are reorganized.

Typical usage:
    >>> from med_reason_evals import MedQADataset
    >>> dataset = MedQADataset(split="test", streaming=False)
    >>> verifiers_dataset = dataset.get_verifiers_dataset()
    >>> verl_dataset = dataset.get_verl_dataset()

The datasets are designed to work with both the Verifiers environment framework
(for RL training) and Verl reward functions (for post-hoc evaluation).
"""

from dotenv import load_dotenv


load_dotenv()


from med_reason_evals.data import (
    BaseDataset,
    HealthBenchDataset,
    MedBulletsDataset,
    MedCaseReasoningDataset,
    MedMCQADataset,
    MedQADataset,
    MedXpertQADataset,
    MetaMedQADataset,
    MMLUProHealthDataset,
    PubHealthBenchDataset,
    PubMedQADataset,
)


__all__ = [
    "BaseDataset",
    "HealthBenchDataset",
    "MedBulletsDataset",
    "MedCaseReasoningDataset",
    "MedMCQADataset",
    "MedQADataset",
    "MedXpertQADataset",
    "MetaMedQADataset",
    "MMLUProHealthDataset",
    "PubHealthBenchDataset",
    "PubMedQADataset",
]
