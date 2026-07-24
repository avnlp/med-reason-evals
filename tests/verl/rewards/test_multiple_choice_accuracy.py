"""Tests for Verl multiple choice accuracy reward function."""

from med_reason_evals.verl.rewards.multiple_choice_accuracy import compute_score


class TestComputeScore:
    """Tests for compute_score function."""

    def test_correct_answer(self):
        """Correct answer should return 1.0."""
        result = compute_score(
            "<answer>C</answer>",
            {"answer": "C"},
        )
        assert result == 1.0

    def test_incorrect_answer(self):
        """Incorrect answer should return 0.0 or format_score."""
        result = compute_score(
            "<answer>A</answer>",
            {"answer": "C"},
        )
        assert result == 0.0

    def test_no_answer_extracted(self):
        """No extractable answer should return 0.0."""
        result = compute_score(
            "Random text without an answer",
            {"answer": "C"},
        )
        assert result == 0.0

    def test_answer_text_match(self):
        """Answer text should match."""
        result = compute_score(
            "<answer>Diabetes</answer>",
            {"answer": "A", "answer_text": "Diabetes"},
        )
        assert result == 1.0

    def test_custom_score(self):
        """Custom score values should be used."""
        result = compute_score(
            "<answer>C</answer>",
            {"answer": "C"},
            score=0.5,
        )
        assert result == 0.5
