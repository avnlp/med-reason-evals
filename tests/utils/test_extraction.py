"""Tests for extraction utilities."""

from med_reason_evals.utils.extraction import (
    extract_answer,
    extract_boxed_answer,
    extract_xml_answer,
)


class TestExtractXmlAnswer:
    """Tests for extract_xml_answer function."""

    def test_simple_xml(self):
        """Simple XML content should be extracted."""
        text = "<answer>C</answer>"
        assert extract_xml_answer(text) == "C"

    def test_with_whitespace(self):
        """Whitespace should be stripped."""
        text = "<answer>  C  </answer>"
        assert extract_xml_answer(text) == "C"

    def test_custom_field(self):
        """Custom field names should work."""
        text = "<response>Hello</response>"
        assert extract_xml_answer(text, field="response") == "Hello"

    def test_no_match(self):
        """No match should return None."""
        assert extract_xml_answer("No tags here") is None

    def test_case_insensitive(self):
        """Tags should be case-insensitive."""
        text = "<ANSWER>C</ANSWER>"
        assert extract_xml_answer(text) == "C"

    def test_last_match_returned(self):
        """Last match should be returned."""
        text = "<answer>First</answer> then <answer>Second</answer>"
        assert extract_xml_answer(text) == "Second"

    def test_empty_tag(self):
        """Empty tag should return empty string."""
        text = "<answer></answer>"
        assert extract_xml_answer(text) == ""

    def test_multiline_content(self):
        """Multiline content should be extracted."""
        text = "<answer>Line 1\nLine 2</answer>"
        assert extract_xml_answer(text) == "Line 1\nLine 2"

    def test_none_input(self):
        """None input should return None."""
        assert extract_xml_answer(None) is None  # type: ignore

    def test_empty_string(self):
        """Empty string should return None."""
        assert extract_xml_answer("") is None


class TestExtractBoxedAnswer:
    """Tests for extract_boxed_answer function."""

    def test_simple_boxed(self):
        """Simple boxed content should be extracted."""
        text = "The answer is \\boxed{C}"
        assert extract_boxed_answer(text) == "C"

    def test_nested_braces(self):
        """Nested braces should be handled."""
        text = "\\boxed{f(x) = {x}}"
        assert extract_boxed_answer(text) == "f(x) = {x}"

    def test_no_boxed(self):
        """Text without boxed should return original."""
        text = "Just some text"
        assert extract_boxed_answer(text) == "Just some text"

    def test_multiple_boxed(self):
        """Last boxed should be returned."""
        text = "\\boxed{A} then \\boxed{B}"
        assert extract_boxed_answer(text) == "B"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert extract_boxed_answer("") == ""

    def test_empty_boxed(self):
        """Empty boxed should return empty string."""
        text = "\\boxed{}"
        assert extract_boxed_answer(text) == ""

    def test_boxed_with_whitespace(self):
        """Whitespace around boxed content should be stripped."""
        text = "\\boxed{  answer  }"
        assert extract_boxed_answer(text) == "answer"

    def test_complex_nested(self):
        """Complex nested braces should work."""
        text = "\\boxed{\\frac{1}{2}}"
        # Should extract content up to first balanced closing brace
        assert extract_boxed_answer(text) == "\\frac{1}{2}"

    def test_none_input(self):
        """None input should return None."""
        assert extract_boxed_answer(None) is None  # type: ignore


class TestExtractAnswer:
    """Tests for extract_answer function with different strategies."""

    def test_xml_strategy(self):
        """XML answer tags should be extracted."""
        assert extract_answer("<answer>C</answer>") == "C"
        assert extract_answer("Some text <answer>B</answer>") == "B"

    def test_boxed_strategy(self):
        """Boxed format should be extracted."""
        assert extract_answer("\\boxed{A}") == "A"
        assert extract_answer("The answer is \\boxed{D}") == "D"

    def test_anchored_patterns(self):
        """Anchored patterns should be extracted."""
        assert extract_answer("Final answer: C") == "C"
        assert extract_answer("The answer is B") == "B"
        assert extract_answer("Choice: D") == "D"
        assert extract_answer("Option is A") == "A"

    def test_last_token_strategy(self):
        """Last letter/number token should be extracted."""
        assert extract_answer("The final answer is C") == "C"
        assert extract_answer("Therefore, the answer is 5") == "5"

    def test_empty_returns_none(self):
        """Empty input should return None."""
        assert extract_answer("") is None
        assert extract_answer(None) is None  # type: ignore

    def test_no_answer_returns_none(self):
        """Text without recognizable answer should return None."""
        assert extract_answer("This is just some text") is None

    def test_xml_priority_over_boxed(self):
        """XML should take priority over boxed."""
        text = "\\boxed{A} and <answer>B</answer>"
        assert extract_answer(text) == "B"

    def test_looks_at_last_500_chars(self):
        """Should only look at last 500 chars for efficiency."""
        prefix = "X" * 1000
        text = prefix + "<answer>Z</answer>"
        assert extract_answer(text) == "Z"

    def test_multiple_xml_returns_last(self):
        """Multiple XML tags should return last one."""
        text = "<answer>A</answer> then <answer>B</answer>"
        assert extract_answer(text) == "B"

    def test_answer_with_parens(self):
        """Answer in parentheses should be handled."""
        assert extract_answer("The answer is (C)") == "C"

    def test_numeric_answers(self):
        """Numeric answers should be extracted."""
        assert extract_answer("The answer is 42") == "42"
        assert extract_answer("Answer: 10") == "10"
