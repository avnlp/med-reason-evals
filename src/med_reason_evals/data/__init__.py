"""Dataset adapters used by medical reasoning evaluators.

Each dataset wrapper exposes a consistent interface for Verifiers evaluation
and Verl reward-model training, while encapsulating the source-specific mapping
logic and prompt shaping.
"""

from med_reason_evals.data.base import BaseDataset
from med_reason_evals.data.med_mcqa import MedMCQADataset
from med_reason_evals.data.medqa import MedQADataset
from med_reason_evals.data.metamedqa import MetaMedQADataset
from med_reason_evals.data.pubmedqa import PubMedQADataset


__all__ = [
    "BaseDataset",
    "MedMCQADataset",
    "MedQADataset",
    "MetaMedQADataset",
    "PubMedQADataset",
]
