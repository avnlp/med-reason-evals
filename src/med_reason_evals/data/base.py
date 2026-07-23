"""Shared dataset contracts for medical reasoning evaluation adapters.

The base class defines the standard projection format for both the Verifiers
and Verl evaluation ecosystems. Concrete datasets implement the conversion from
raw Hugging Face examples into those canonical schemas.
"""

from abc import ABC, abstractmethod

from datasets import Dataset
from datasets.iterable_dataset import IterableDataset


class BaseDataset(ABC):
    """Abstract base class for medical reasoning datasets.

    Implementations are responsible for materializing both the Verifiers
    evaluation schema and the Verl training schema, keeping dataset-specific
    logic encapsulated in one place.
    """

    def __init__(self, split: str = "test", streaming: bool = True, **kwargs) -> None:
        """Initialize the dataset adapter.

        Args:
            split: The dataset split to use (e.g., "train", "test", "validation").
            streaming: Whether to stream the dataset (recommended for large datasets).
            **kwargs: Additional keyword arguments for dataset configuration.
        """
        self.split = split
        self.streaming = streaming
        self._kwargs = kwargs

    @property
    @abstractmethod
    def num_options(self) -> int:
        """Return the number of MCQ options.

        Returns:
            An integer representing the number of options. Returns 1 for
            non-MCQ (e.g., open-ended) datasets.
        """
        ...

    @abstractmethod
    def get_verifiers_dataset(self) -> Dataset | IterableDataset:
        """Return a dataset formatted for the Verifiers evaluation harness.

        Returns:
            A Hugging Face Dataset or IterableDataset with ``question``,
            ``answer``, and ``info`` fields populated for the evaluator.
        """

    @abstractmethod
    def get_verl_dataset(self) -> Dataset | IterableDataset:
        """Return a dataset formatted for Verl reward-model training.

        Returns:
            A Hugging Face Dataset or IterableDataset with ``prompt``,
            ``ground_truth``, ``data_source``, and ``metadata`` fields.
        """
