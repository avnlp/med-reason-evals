"""Tests for text normalization utilities."""

import pytest

from med_reason_evals.utils.text import (
    nfkc_casefold,
    normalize_answer,
    normalize_spaces,
)


class TestNfkcCasefold:
    """Tests for nfkc_casefold function."""

    def test_basic_casefold(self):
        """Basic case folding should work."""
        assert nfkc_casefold("Hello World") == "hello world"
        assert nfkc_casefold("HELLO") == "hello"

    def test_unicode_normalization(self):
        """Unicode NFKC normalization should handle special characters."""
        # Mathematical script H (U+210C) should normalize to 'h'
        assert nfkc_casefold("ℌello") == "hello"
        # Fullwidth characters should normalize
        assert nfkc_casefold("Ｈｅｌｌｏ") == "hello"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert nfkc_casefold("") == ""

    def test_none_handling(self):
        """None should be treated as empty string."""
        assert nfkc_casefold(None) == ""  # type: ignore

    def test_whitespace_preservation(self):
        """Whitespace should be preserved (not normalized)."""
        assert nfkc_casefold("hello   world") == "hello   world"
        assert nfkc_casefold("  leading") == "  leading"

    def test_combining_characters(self):
        """Combining characters should be normalized."""
        # é as e + combining acute vs single character
        assert nfkc_casefold("calf\u0065\u0301") == nfkc_casefold("calf\u00e9")

    def test_special_symbols(self):
        """Various Unicode symbols should normalize correctly."""
        # Roman numerals
        assert nfkc_casefold("Ⅰ") == "i"
        assert nfkc_casefold("Ⅴ") == "v"
        assert nfkc_casefold("Ⅹ") == "x"

    def test_mixed_scripts(self):
        """Mixed scripts should normalize where possible."""
        result = nfkc_casefold("Hello ℌ-world")
        assert result == "hello h-world"


class TestNormalizeSpaces:
    """Tests for normalize_spaces function."""

    def test_multiple_spaces(self):
        """Multiple spaces should collapse to single."""
        assert normalize_spaces("hello   world") == "hello world"

    def test_tabs_and_newlines(self):
        """Tabs and newlines should be normalized to spaces."""
        assert normalize_spaces("hello\tworld") == "hello world"
        assert normalize_spaces("hello\nworld") == "hello world"
        assert normalize_spaces("hello\r\nworld") == "hello world"

    def test_leading_trailing_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        assert normalize_spaces("  hello world  ") == "hello world"
        assert normalize_spaces("\t\nhello\t\n") == "hello"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert normalize_spaces("") == ""

    def test_single_spaces_preserved(self):
        """Single spaces between words should be preserved."""
        assert normalize_spaces("hello world test") == "hello world test"

    def test_mixed_whitespace(self):
        """Mixed whitespace types should all normalize."""
        text = "hello \t \n \r world"
        assert normalize_spaces(text) == "hello world"

    def test_all_whitespace(self):
        """String with only whitespace should return empty."""
        assert normalize_spaces("   ") == ""
        assert normalize_spaces("\t\n\r ") == ""


class TestNormalizeAnswer:
    """Tests for normalize_answer function with all modes."""

    class TestBasicMode:
        """Tests for 'basic' normalization mode."""

        def test_basic_normalization(self):
            """Basic text should be normalized."""
            assert normalize_answer("  Hello World  ", mode="basic") == "hello world"

        def test_unicode_normalization(self):
            """Unicode should be normalized."""
            assert normalize_answer("ℌello", mode="basic") == "hello"

        def test_whitespace_collapse(self):
            """Multiple whitespace should collapse."""
            assert normalize_answer("hello   world", mode="basic") == "hello world"

        def test_empty_string(self):
            """Empty string should return empty."""
            assert normalize_answer("", mode="basic") == ""

        def test_none_returns_empty(self):
            """None should return empty string."""
            assert normalize_answer(None, mode="basic") == ""  # type: ignore

    class TestSemanticMode:
        """Tests for 'semantic' normalization mode."""

        def test_removes_articles(self):
            """Articles should be removed."""
            assert normalize_answer("the answer", mode="semantic") == "answer"
            assert normalize_answer("a test", mode="semantic") == "test"
            assert normalize_answer("an example", mode="semantic") == "example"

        def test_removes_punctuation(self):
            """Punctuation should be removed."""
            assert normalize_answer("hello, world!", mode="semantic") == "hello world"
            assert normalize_answer("test.", mode="semantic") == "test"

        def test_case_insensitive(self):
            """Should be case insensitive."""
            assert normalize_answer("The Answer", mode="semantic") == "answer"

        def test_whitespace_normalized(self):
            """Whitespace should be normalized."""
            assert normalize_answer("the   answer", mode="semantic") == "answer"

        def test_empty_string(self):
            """Empty string should return empty."""
            assert normalize_answer("", mode="semantic") == ""

        def test_complex_sentence(self):
            """Complex sentence with articles and punctuation."""
            text = "The quick, brown fox jumps over a lazy dog!"
            result = normalize_answer(text, mode="semantic")
            assert result == "quick brown fox jumps over lazy dog"

    class TestMcqMode:
        """Tests for 'mcq' normalization mode."""

        def test_single_letter_uppercased(self):
            """Single letters should be uppercased."""
            assert normalize_answer("a", mode="mcq") == "A"
            assert normalize_answer("b", mode="mcq") == "B"
            assert normalize_answer("  c  ", mode="mcq") == "C"

        def test_single_digit_preserved(self):
            """Single digits should be preserved."""
            assert normalize_answer("1", mode="mcq") == "1"
            assert normalize_answer("  5  ", mode="mcq") == "5"

        def test_multiple_digits_preserved(self):
            """Multiple digits should be preserved."""
            assert normalize_answer("10", mode="mcq") == "10"
            assert normalize_answer("42", mode="mcq") == "42"

        def test_text_normalized(self):
            """Multi-character text should be normalized."""
            assert normalize_answer("  hello  ", mode="mcq") == "hello"
            assert normalize_answer("Diabetes", mode="mcq") == "diabetes"

        def test_whitespace_stripped_before_check(self):
            """Whitespace should be stripped before length check."""
            assert normalize_answer("  a  ", mode="mcq") == "A"

    class TestDefaultMode:
        """Tests for default mode (should be basic)."""

        def test_default_is_basic(self):
            """Default mode should behave like basic."""
            assert normalize_answer("  Hello  ") == "hello"
            assert normalize_answer("ℌello") == "hello"

    class TestInvalidMode:
        """Tests for invalid mode handling."""

        def test_unknown_mode_raises_error(self):
            """Unknown mode should raise ValueError."""
            with pytest.raises(ValueError, match="Unknown normalization mode"):
                normalize_answer("test", mode="invalid")  # type: ignore
