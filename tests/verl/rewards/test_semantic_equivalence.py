"""Tests for Verl semantic equivalence reward function."""

import pytest

from med_reason_evals.verl.rewards.semantic_equivalence import (
    compute_score,
    exact_match,
    substring_match,
)


class TestExactMatch:
    """Tests for exact_match function."""

    def test_exact_match_single(self):
        """Exact match with single answer should work."""
        assert exact_match("diabetes", "Diabetes")
        assert not exact_match("diabetes", "Hypertension")

    def test_exact_match_multiple(self):
        """Exact match with multiple answers should work."""
        assert exact_match("diabetes", ["Hypertension", "Diabetes"])
        assert not exact_match("cancer", ["Hypertension", "Diabetes"])

    def test_normalization_applied(self):
        """Normalization should be applied before comparison."""
        assert exact_match("DIABETES", "diabetes")
        assert exact_match("myocardial infarction", "MYOCARDIAL INFARCTION")


class TestSubstringMatch:
    """Tests for substring_match function."""

    def test_substring_match(self):
        """Substring should be found."""
        assert substring_match("acute myocardial infarction", "myocardial infarction")

    def test_no_match(self):
        """Non-matching strings should return False."""
        assert not substring_match("diabetes", "hypertension")


class TestComputeScore:
    """Tests for compute_score function."""

    def test_exact_match_score(self):
        """Exact match should return 1.0."""
        result = compute_score(
            "<answer>Diabetes</answer>",
            {"target": "Diabetes"},
            method="exact",
        )
        assert result == 1.0

    def test_substring_match_score(self):
        """Substring match should return 1.0."""
        result = compute_score(
            "<answer>acute myocardial infarction</answer>",
            {"target": "myocardial infarction"},
            method="substring",
        )
        assert result == 1.0

    def test_no_match_score(self):
        """No match should return 0.0."""
        result = compute_score(
            "<answer>Diabetes</answer>",
            {"target": "Hypertension"},
        )
        assert result == 0.0

    def test_compute_score_no_answer_extracted(self):
        """No extractable answer should return 0.0."""
        result = compute_score(
            "... --- !!! ???",
            {"target": "Diabetes"},
        )
        assert result == 0.0

    def test_compute_score_missing_golden(self):
        """Missing target/answer key in ground_truth returns 0.0."""
        result = compute_score(
            "<answer>Diabetes</answer>",
            {"other_key": "value"},
        )
        assert result == 0.0

    def test_compute_score_empty_golden(self):
        """Empty ground truth value returns 0.0."""
        result = compute_score(
            "<answer>Diabetes</answer>",
            {"target": ""},
        )
        assert result == 0.0

    def test_compute_score_substring_method(self):
        """Substring method should match partial answers."""
        result = compute_score(
            "<answer>acute myocardial infarction</answer>",
            {"target": "myocardial infarction"},
            method="substring",
        )
        assert result == 1.0

    def test_compute_score_invalid_method(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_score(
                "<answer>Diabetes</answer>",
                {"target": "Diabetes"},
                method="fuzzy",
            )

    def test_compute_score_uses_answer_key(self):
        """Should work with 'answer' key when 'target' is absent."""
        result = compute_score(
            "<answer>Diabetes</answer>",
            {"answer": "Diabetes"},
            method="exact",
        )
        assert result == 1.0

    def test_compute_score_custom_scores(self):
        """Custom score and format_score should be used."""
        result = compute_score(
            "<answer>wrong</answer>",
            {"target": "correct"},
            method="exact",
            score=0.9,
            format_score=0.1,
        )
        assert result == 0.1


class TestExactMatchExtended:
    """Extended tests for exact_match edge cases."""

    def test_exact_match_empty_prediction(self):
        """Empty prediction should return False."""
        assert not exact_match("", "Diabetes")

    def test_exact_match_list_golden(self):
        """List of golden answers should work."""
        assert exact_match("diabetes", ["cancer", "diabetes", "flu"])
        assert not exact_match("headache", ["cancer", "diabetes", "flu"])


class TestSubstringMatchExtended:
    """Extended tests for substring_match edge cases."""

    def test_substring_match_empty_prediction(self):
        """Empty prediction should return False."""
        assert not substring_match("", "something")

    def test_substring_match_list_golden(self):
        """List of golden answers should work with substring match."""
        assert substring_match("acute myocardial infarction", ["AMI", "myocardial"])
        assert not substring_match("headache", ["cancer", "diabetes"])
