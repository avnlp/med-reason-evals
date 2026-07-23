"""Dataset adapters used by medical reasoning evaluators.

Each dataset wrapper exposes a consistent interface for Verifiers evaluation
and Verl reward-model training, while encapsulating the source-specific mapping
logic and prompt shaping.
"""

from med_reason_evals.data.base import BaseDataset


__all__ = [
    "BaseDataset",
]
