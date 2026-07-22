"""Tests for parsing utilities."""

from med_reason_evals.utils.parsing import (
    parse_json_response,
    parse_yes_no,
    strip_think_tags,
)


class TestStripThinkTags:
    """Tests for strip_think_tags function."""

    def test_no_think_tags(self):
        """Text without think tags should be returned as-is."""
        text = "The answer is C"
        assert strip_think_tags(text) == "The answer is C"

    def test_single_think_pair(self):
        """Single think pair should be stripped."""
        text = "<think>Some reasoning</think>The answer is C"
        assert strip_think_tags(text) == "The answer is C"

    def test_case_insensitive(self):
        """Think tags should be case-insensitive."""
        text = "<THINK>Reasoning</THINK>Answer"
        assert strip_think_tags(text) == "Answer"

    def test_unpaired_closing_tag(self):
        """Unpaired closing tag should still work."""
        text = "Some reasoning</think>The answer"
        assert strip_think_tags(text) == "The answer"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert strip_think_tags("") == ""
        assert strip_think_tags(None) == ""  # type: ignore

    def test_multiple_think_pairs(self):
        """Multiple think pairs should return full text."""
        text = "<think>First</think>Middle<think>Second</think>End"
        assert strip_think_tags(text) == text.strip()

    def test_think_with_whitespace(self):
        """Think tags with whitespace should be handled."""
        text = "<think>\n\nReasoning\n\n</think>\n\nAnswer"
        assert strip_think_tags(text) == "Answer"

    def test_only_closing_tag(self):
        """Only closing tag without opening should work."""
        text = "Reasoning content</think>Final Answer"
        assert strip_think_tags(text) == "Final Answer"

    def test_think_at_end(self):
        """Think tag at end should return empty string."""
        text = "Content<think>reasoning</think>"
        assert strip_think_tags(text) == ""

    def test_nested_looking_content(self):
        """Content that looks like nested tags."""
        text = "<think>Some <b>bold</b> reasoning</think>Answer"
        assert strip_think_tags(text) == "Answer"


class TestParseJsonResponse:
    """Tests for parse_json_response function."""

    def test_json_in_markdown_code_block(self):
        """JSON wrapped in markdown code block should be parsed."""
        text = '```json\n{"criteria_met": true, "explanation": "Good"}\n```'
        result = parse_json_response(text)
        assert result == {"criteria_met": True, "explanation": "Good"}

    def test_json_in_markdown_without_language(self):
        """JSON in markdown code block without json tag should be parsed."""
        text = '```\n{"criteria_met": false, "explanation": "Poor"}\n```'
        result = parse_json_response(text)
        assert result == {"criteria_met": False, "explanation": "Poor"}

    def test_plain_json_object(self):
        """Plain JSON object should be parsed."""
        text = '{"criteria_met": true, "explanation": "Correct"}'
        result = parse_json_response(text)
        assert result == {"criteria_met": True, "explanation": "Correct"}

    def test_json_extraction_from_mixed_text(self):
        """JSON embedded in surrounding text should be extracted."""
        text = 'Analysis: {"criteria_met": true, "explanation": "Valid"} End.'
        result = parse_json_response(text)
        assert result.get("criteria_met") is True

    def test_malformed_json_returns_empty_dict(self):
        """Malformed JSON should return empty dict."""
        text = "{not valid json"
        result = parse_json_response(text)
        assert result == {}

    def test_no_json_returns_empty_dict(self):
        """Text without JSON should return empty dict."""
        text = "This is just plain text."
        result = parse_json_response(text)
        assert result == {}

    def test_multiple_json_objects_returns_first(self):
        """When multiple JSON objects exist, should return first valid one."""
        text = '{"first": true} and then {"second": false}'
        result = parse_json_response(text)
        assert result.get("first") is True

    def test_empty_json_object(self):
        """Empty JSON object should be parsed."""
        text = "{}"
        result = parse_json_response(text)
        assert result == {}

    def test_whitespace_handling(self):
        """JSON with extra whitespace should be parsed."""
        text = '```json\n  { "criteria_met" : true }  \n```'
        result = parse_json_response(text)
        assert result.get("criteria_met") is True

    def test_nested_json(self):
        """Nested JSON structures should be parsed."""
        text = '{"outer": {"inner": "value"}}'
        result = parse_json_response(text)
        assert result["outer"]["inner"] == "value"

    def test_json_array(self):
        """JSON array should be parsed."""
        text = '[{"item": 1}, {"item": 2}]'
        result = parse_json_response(text)
        assert len(result) == 2


class TestParseYesNo:
    """Tests for parse_yes_no function."""

    def test_plain_yes(self):
        """Plain 'yes' should return True."""
        assert parse_yes_no("yes") is True

    def test_plain_no(self):
        """Plain 'no' should return False."""
        assert parse_yes_no("no") is False

    def test_case_insensitivity(self):
        """Parsing should be case insensitive."""
        assert parse_yes_no("YES") is True
        assert parse_yes_no("Yes") is True
        assert parse_yes_no("NO") is False
        assert parse_yes_no("No") is False

    def test_yes_with_explanation(self):
        """'yes' followed by explanation should return True."""
        assert parse_yes_no("Yes, the diagnosis is correct.") is True
        assert parse_yes_no("yes - the predictions match") is True

    def test_no_with_explanation(self):
        """'no' followed by explanation should return False."""
        assert parse_yes_no("No, the answer is incorrect.") is False
        assert parse_yes_no("no - the values differ") is False

    def test_yesterday_false_positive(self):
        """'yesterday' should NOT be parsed as yes (strictness test)."""
        result = parse_yes_no("yesterday")
        assert result is None or result is False

    def test_json_criteria_met_true(self):
        """JSON with criteria_met: true should return True."""
        assert parse_yes_no('{"criteria_met": true}') is True

    def test_json_criteria_met_false(self):
        """JSON with criteria_met: false should return False."""
        assert parse_yes_no('{"criteria_met": false}') is False

    def test_json_correct_true(self):
        """JSON with correct: true should return True."""
        assert parse_yes_no('{"correct": true}') is True

    def test_json_correct_false(self):
        """JSON with correct: false should return False."""
        assert parse_yes_no('{"correct": false}') is False

    def test_json_string_values(self):
        """JSON with string boolean values should work."""
        assert parse_yes_no('{"criteria_met": "yes"}') is True
        assert parse_yes_no('{"criteria_met": "true"}') is True
        assert parse_yes_no('{"correct": "no"}') is False
        assert parse_yes_no('{"correct": "false"}') is False

    def test_json_embedded_in_text(self):
        """JSON embedded in surrounding text should be parsed."""
        assert parse_yes_no('Based on analysis: {"criteria_met": true}') is True
        assert parse_yes_no('Result: {"correct": false} due to mismatch') is False

    def test_empty_string(self):
        """Empty string should return None."""
        assert parse_yes_no("") is None

    def test_whitespace_only(self):
        """Whitespace only should return None."""
        assert parse_yes_no("   ") is None

    def test_unrelated_text(self):
        """Unrelated text should return None."""
        assert parse_yes_no("The answer is correct.") is None
        assert parse_yes_no("I believe this is right.") is None

    def test_leading_whitespace(self):
        """Leading whitespace before yes/no should be handled."""
        assert parse_yes_no("  yes") is True
        assert parse_yes_no("  no") is False

    def test_json_criteria_met_priority_over_correct(self):
        """criteria_met should be checked before correct."""
        assert parse_yes_no('{"criteria_met": true, "correct": false}') is True
        assert parse_yes_no('{"criteria_met": false, "correct": true}') is False

    def test_malformed_json(self):
        """Malformed JSON should not crash and return None."""
        assert parse_yes_no("{criteria_met: true}") is None
        assert parse_yes_no('{"criteria_met": }') is None

    def test_yes_no_not_at_start_returns_none(self):
        """yes/no not at start should return None (strict anchoring)."""
        result = parse_yes_no("I think yes")
        assert result is None

    def test_newlines_before_yes_no(self):
        """Newlines before yes/no should still work."""
        assert parse_yes_no("\nyes") is True
        assert parse_yes_no("\n\nno") is False
