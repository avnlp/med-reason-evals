"""Tests for base dataset class."""

from typing import ClassVar

import pytest
from datasets import Dataset

from med_reason_evals.data.base import BaseDataset


class _ValidTestDataset(BaseDataset):
    """Minimal valid subclass implementing all abstract members."""

    _num_options: ClassVar[int] = 4

    @property
    def num_options(self) -> int:
        return self._num_options

    def get_verifiers_dataset(self) -> Dataset:
        data = {
            "question": ["What is X?"],
            "answer": ["A"],
            "info": [{"source": "test"}],
        }
        return Dataset.from_dict(data)

    def get_verl_dataset(self) -> Dataset:
        data = {
            "prompt": ["What is X?"],
            "ground_truth": ["A"],
            "data_source": ["test"],
            "metadata": [{"source": "test"}],
        }
        return Dataset.from_dict(data)


class _EmptyVerifiers(_ValidTestDataset):
    """A subclass returning an empty verifiers dataset."""

    def get_verifiers_dataset(self) -> Dataset:
        return Dataset.from_dict({"question": [], "answer": [], "info": []})


class _EmptyVerl(_ValidTestDataset):
    """A subclass returning an empty verl dataset."""

    def get_verl_dataset(self) -> Dataset:
        return Dataset.from_dict(
            {
                "prompt": [],
                "ground_truth": [],
                "data_source": [],
                "metadata": [],
            }
        )


class _EmptyInfo(_ValidTestDataset):
    """A subclass with an empty dict for the info field."""

    def get_verifiers_dataset(self) -> Dataset:
        return Dataset.from_dict({"question": ["Q1"], "answer": ["B"], "info": [{}]})


class _EmptyStrings(_ValidTestDataset):
    """A subclass with empty strings for string fields."""

    def get_verifiers_dataset(self) -> Dataset:
        return Dataset.from_dict({"question": [""], "answer": [""], "info": [{}]})


class TestBaseDataset:
    """Tests for BaseDataset ABC."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        """BaseDataset should not be directly instantiated."""
        with pytest.raises(TypeError):
            BaseDataset()

    def test_abstract_methods_are_declared(self) -> None:
        """BaseDataset should define required abstract methods."""
        assert "get_verifiers_dataset" in BaseDataset.__abstractmethods__
        assert "get_verl_dataset" in BaseDataset.__abstractmethods__
        assert "num_options" in BaseDataset.__abstractmethods__

    def test_num_options_is_an_abstract_property(self) -> None:
        """BaseDataset should define num_options as an abstract property."""
        assert isinstance(BaseDataset.__dict__["num_options"], property)

    def test_valid_subclass_is_instantiable(self) -> None:
        """A subclass implementing all abstract members should be instantiable."""
        ds = _ValidTestDataset()
        assert isinstance(ds, BaseDataset)

    def test_get_verifiers_dataset_returns_expected_schema(self) -> None:
        """get_verifiers_dataset should return a Dataset with the expected columns."""
        ds = _ValidTestDataset()
        result = ds.get_verifiers_dataset()
        assert isinstance(result, Dataset)
        assert set(result.column_names) == {"question", "answer", "info"}

    def test_get_verl_dataset_returns_expected_schema(self) -> None:
        """get_verl_dataset should return a Dataset with the expected columns."""
        ds = _ValidTestDataset()
        result = ds.get_verl_dataset()
        assert isinstance(result, Dataset)
        assert set(result.column_names) == {
            "prompt",
            "ground_truth",
            "data_source",
            "metadata",
        }

    def test_num_options_returns_configured_value(self) -> None:
        """num_options should return the value set by the subclass."""
        ds = _ValidTestDataset()
        assert ds.num_options == 4

    def test_single_item_dataset_preserves_values(self) -> None:
        """A single-item valid subclass should preserve its values."""
        ds = _ValidTestDataset()
        verifiers = ds.get_verifiers_dataset()
        assert len(verifiers) == 1
        assert verifiers[0]["answer"] == "A"

        verl = ds.get_verl_dataset()
        assert len(verl) == 1
        assert verl[0]["ground_truth"] == "A"

    def test_empty_verifiers_dataset_is_valid(self) -> None:
        """An empty Dataset should be valid for the verifiers schema."""
        result = _EmptyVerifiers().get_verifiers_dataset()
        assert len(result) == 0

    def test_empty_verl_dataset_is_valid(self) -> None:
        """An empty Dataset should be valid for the verl schema."""
        result = _EmptyVerl().get_verl_dataset()
        assert len(result) == 0

    def test_info_field_accepts_empty_dict(self) -> None:
        """The info field should accept an empty dict."""
        result = _EmptyInfo().get_verifiers_dataset()
        assert result[0]["info"] == {}

    def test_string_fields_accept_empty_values(self) -> None:
        """String fields like question and answer should accept empty strings."""
        result = _EmptyStrings().get_verifiers_dataset()
        assert result[0]["question"] == ""
        assert result[0]["answer"] == ""
