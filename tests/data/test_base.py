"""Tests for base dataset class."""

import pytest

from med_reason_evals.data.base import BaseDataset


class TestBaseDataset:
    """Tests for BaseDataset ABC."""

    def test_base_cannot_be_instantiated(self):
        """BaseDataset should not be directly instantiated."""
        with pytest.raises(TypeError):
            BaseDataset()

    def test_base_has_required_attributes(self):
        """BaseDataset should define required abstract methods."""
        assert hasattr(BaseDataset, "get_verifiers_dataset")
        assert hasattr(BaseDataset, "get_verl_dataset")
