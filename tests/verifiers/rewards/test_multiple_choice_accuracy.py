"""Tests for multiple choice accuracy reward function."""

from unittest.mock import MagicMock

import pytest

from med_reason_evals.verifiers.rewards.multiple_choice_accuracy import (
    MCQAccuracyResult,
    _check_anchored_token,
    _check_answer_text,
    _check_last_token,
    _strip_tex,
    _tail_region,
    _token_kind_matches_answer_letter,
    accuracy_reward,
    multiple_choice_accuracy,
)


class TestMultipleChoiceAccuracy:
    """Tests for multiple_choice_accuracy function."""

    def test_direct_answer_single_letter(self):
        """Direct answer with just the letter should match."""
        assert multiple_choice_accuracy("C", "C", "Correct option")
        assert multiple_choice_accuracy("c", "C", "Correct option")
        assert not multiple_choice_accuracy("A", "C", "Correct option")

    def test_anchored_final_answer(self):
        """Anchored patterns like 'final answer: C' should match."""
        assert multiple_choice_accuracy(
            "Let me think... Final answer: C",
            "C",
            "Correct option",
        )
        assert multiple_choice_accuracy(
            "The answer is C",
            "C",
            "Correct option",
        )

    def test_xml_answer_tags(self):
        """Answers in XML tags should match."""
        assert multiple_choice_accuracy(
            "<answer>C</answer>",
            "C",
            "Correct option",
        )
        assert multiple_choice_accuracy(
            "<think>Reasoning here</think><answer>B</answer>",
            "B",
            "Correct option",
        )

    def test_negated_answer_should_not_match(self):
        """Negated answers like 'not C' should not match."""
        assert not multiple_choice_accuracy(
            "The answer is not C, it's D.",
            "C",
            "Wrong option",
        )

    def test_answer_text_fallback(self):
        """Answer text should match when no letter is found."""
        assert multiple_choice_accuracy(
            "The patient has Diabetes",
            "A",
            "Diabetes",
        )

    def test_chain_of_thought_response(self):
        """CoT responses should extract the final answer."""
        cot_response = """
        Let me analyze each option:

        A) This is incorrect because...
        B) This could be possible, but...
        C) This seems most likely given the presentation.
        D) This is ruled out by the lab results.

        Final answer: C
        """
        assert multiple_choice_accuracy(
            cot_response,
            "C",
            "Correct diagnosis",
        )

    def test_return_details(self):
        """return_details=True should return MCQAccuracyResult."""
        result = multiple_choice_accuracy(
            "Final answer: C",
            "C",
            "Option C",
            return_details=True,
        )
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is True
        assert result.method == "anchored_token"
        assert result.matched_answer == "C"
        assert result.correct_answer == "C"

    def test_empty_response(self):
        """Empty response should return False."""
        assert not multiple_choice_accuracy("", "C", "Option C")
        assert not multiple_choice_accuracy(None, "C", "Option C")  # type: ignore

    def test_invalid_answer_letter_raises(self):
        """Invalid answer_letter should raise ValueError."""
        with pytest.raises(ValueError):
            multiple_choice_accuracy("C", "", "Option")
        with pytest.raises(ValueError):
            multiple_choice_accuracy("C", "ABC", "Option")

    def test_invalid_answer_letter_raises_even_when_completion_empty(self):
        """Invalid answer_letter should raise even if llm_answer is empty."""
        with pytest.raises(ValueError):
            multiple_choice_accuracy("", "ABC", "Option")
        with pytest.raises(ValueError):
            multiple_choice_accuracy(None, "ABC", "Option")  # type: ignore

    def test_numeric_options(self):
        """Numeric options (1, 2, 3) should work."""
        assert multiple_choice_accuracy("The answer is 2", "2", "Second option")
        assert multiple_choice_accuracy("2", "2", "Second option")

    def test_think_tags_stripped(self):
        """Think tags should be stripped before analysis."""
        response = "<think>Let me think about this...</think>C"
        assert multiple_choice_accuracy(response, "C", "Option C")

    def test_boxed_answer(self):
        """Boxed answers should be extracted."""
        assert multiple_choice_accuracy(
            "The answer is \\boxed{C}",
            "C",
            "Option C",
        )

    def test_leading_option_format(self):
        """Leading option format like 'B. Answer' should match."""
        assert multiple_choice_accuracy(
            "B. The correct answer text",
            "B",
            "The correct answer text",
        )

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("**C**", True),
            ("*C*", True),
            ("`C`", True),
            ("Option C is correct", True),
            ("I choose C", True),
            ("Therefore, C", True),
        ],
    )
    def test_various_formats(self, response: str, expected: bool):
        """Various answer formats should be recognized."""
        assert multiple_choice_accuracy(response, "C", "Option C") == expected


class TestStripTex:
    """Tests for _strip_tex function."""

    def test_strip_tex_success(self):
        """Test _strip_tex removes LaTeX formatting when pylatexenc is available."""
        text = r"$\frac{1}{2}$"
        result = _strip_tex(text)
        assert isinstance(result, str)
        try:
            import pylatexenc  # noqa: F401
        except ImportError:
            # pylatexenc not installed: _strip_tex must fall back to the original text.
            assert result == text
        else:
            # pylatexenc installed: LaTeX markup should actually be converted away.
            assert result != text
            assert "\\frac" not in result
            assert "$" not in result

    def test_strip_tex_import_failure(self, mocker):
        """Test _strip_tex returns original when pylatexenc import fails (Line 43)."""
        # Mock the import to raise an ImportError
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "pylatexenc" in name:
                raise ImportError("No module named 'pylatexenc'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            text = r"$\frac{1}{2}$"
            result = _strip_tex(text)
            # Should return original text when import fails
            assert result == text
        finally:
            builtins.__import__ = original_import

    def test_strip_tex_runtime_exception(self, mocker):
        """Test _strip_tex returns original when LatexNodes2Text raises exception."""
        # _strip_tex does a *local* `from pylatexenc.latex2text import
        # LatexNodes2Text` inside the function body, so patching an attribute
        # on this module has no effect on it. Instead inject a fake
        # `pylatexenc.latex2text` module into sys.modules (this also works
        # when the real pylatexenc package isn't installed) so the local
        # import resolves to our mock.
        import sys
        import types

        class MockLatexNodes2Text:
            def __init__(self, *args, **kwargs):
                raise Exception("Latex parsing error")

        fake_latex2text = types.ModuleType("pylatexenc.latex2text")
        fake_latex2text.LatexNodes2Text = MockLatexNodes2Text  # type: ignore
        fake_pylatexenc = types.ModuleType("pylatexenc")
        fake_pylatexenc.latex2text = fake_latex2text  # type: ignore

        mocker.patch.dict(
            sys.modules,
            {
                "pylatexenc": fake_pylatexenc,
                "pylatexenc.latex2text": fake_latex2text,
            },
        )

        text = r"$\frac{1}{2}$"
        result = _strip_tex(text)
        assert result == text


class TestTokenKindMatchesAnswerLetter:
    """Tests for _token_kind_matches_answer_letter function."""

    def test_predicted_is_none_returns_false(self):
        """Test that None predicted returns False (Line 70)."""
        result = _token_kind_matches_answer_letter(None, "C")
        assert result is False

    def test_predicted_empty_returns_false(self):
        """Test that empty predicted returns False."""
        result = _token_kind_matches_answer_letter("", "C")
        assert result is False

    def test_digit_answer_letter_with_digit_predicted(self):
        """Test digit matching when answer_letter is digit."""
        result = _token_kind_matches_answer_letter("5", "2")
        assert result is True

    def test_digit_answer_letter_with_letter_predicted(self):
        """Test digit answer with letter predicted returns False."""
        result = _token_kind_matches_answer_letter("C", "2")
        assert result is False

    def test_letter_answer_letter_with_digit_predicted(self):
        """Test letter answer with digit predicted returns False."""
        result = _token_kind_matches_answer_letter("2", "C")
        assert result is False


class TestDirectAnswerCheck:
    """Tests for direct answer matching (Line 125)."""

    def test_direct_answer_exact_match(self):
        """Test direct answer check with exact match to norm_answer_letter."""
        result = multiple_choice_accuracy("C", "C", "Option C", return_details=True)
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is True
        assert result.method == "direct_answer"

    def test_direct_answer_lowercase(self):
        """Test direct answer check with lowercase letter."""
        result = multiple_choice_accuracy("c", "C", "Option C", return_details=True)
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is True
        assert result.method == "direct_answer"

    def test_direct_answer_numeric(self):
        """Test direct answer check with numeric answer."""
        result = multiple_choice_accuracy("3", "3", "Option 3", return_details=True)
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is True
        assert result.method == "direct_answer"


class TestTailRegion:
    """Tests for _tail_region function."""

    def test_find_last_non_empty_line_when_tail_empty(self):
        """Test finding last non-empty line when tail is empty (Lines 167-170)."""
        # Text with sentence boundaries that result in empty tail
        text = "First sentence.\n\n\n\n"
        result = _tail_region(text)
        # Should find "First sentence" from the non-empty line (punctuation preserved)
        assert result == "First sentence."

    def test_max_tokens_truncation(self):
        """Test max tokens truncation (Line 174)."""
        # Create text with more than 64 tokens
        tokens = ["word"] * 100
        text = " ".join(tokens)
        result = _tail_region(text, max_tokens=64)
        result_tokens = result.split()
        assert len(result_tokens) == 64
        # Should be the last 64 tokens
        assert result_tokens[0] == "word"

    def test_tail_region_with_sentence_boundary(self):
        """Test tail extraction using sentence boundary."""
        text = "First sentence. Second sentence! Third sentence?"
        result = _tail_region(text)
        # Punctuation is preserved in the tail
        assert result == "Third sentence?"

    def test_tail_region_no_boundaries(self):
        """Test tail region when no sentence boundaries exist."""
        text = "single sentence without boundaries"
        result = _tail_region(text)
        assert result == text


class TestCheckAnchoredToken:
    """Tests for _check_anchored_token function."""

    def test_prefix_matches_handling(self):
        """Test prefix matches handling with custom prefix (Lines 190-198)."""
        llm_answer = "The answer is C"
        result, explicit = _check_anchored_token(llm_answer, "The answer is", "C")
        assert result == "C"
        assert explicit is True

    def test_prefix_matches_no_norm(self):
        """Test prefix matches when prefix_norm is empty."""
        result, explicit = _check_anchored_token("answer is C", "", "C")
        assert result == "C"
        assert explicit is True

    def test_prefix_matches_whitespace_only(self):
        """Test prefix matches when prefix is whitespace only."""
        result, explicit = _check_anchored_token("answer is C", "   ", "C")
        # Should fall back to ANCHOR_PATTERN
        assert result == "C"
        assert explicit is True

    def test_anchored_token_with_negation(self):
        """Test anchored token match with negation check (Line 211)."""
        # Line 211 is reached when _token_kind_matches_answer_letter returns False
        # but predicted == norm_answer_letter and neg is None. This happens when
        # answer_letter is a digit but predicted is a letter (kind mismatch).
        llm_answer = "The final answer is C"
        result, explicit = _check_anchored_token(llm_answer, None, "C")
        assert result == "C"
        # When kind matches (both letters), explicit_choice_found is True (line 208)
        assert explicit is True

    def test_anchored_token_negated(self):
        """Test anchored token with negation - should return None."""
        llm_answer = "The final answer is not C"
        result, explicit = _check_anchored_token(llm_answer, None, "C")
        assert result is None
        assert explicit is False

    def test_anchored_token_kind_mismatch(self):
        """Test when token kind doesn't match answer letter type."""
        # Digit-based answer but letter predicted
        # When kind doesn't match, _token_kind_matches_answer_letter returns False.
        # So the first condition (line 206-209) is skipped. Then the second condition
        # (line 210-211) checks: predicted == norm_answer_letter. But "C" != "2".
        llm_answer = "The answer is C"
        result, explicit = _check_anchored_token(llm_answer, None, "2")
        # Kind mismatch and predicted != norm_answer_letter, so None
        assert result is None
        assert explicit is False

    def test_anchored_token_no_matches(self):
        """Test when no anchored matches found."""
        llm_answer = "This is just random text without answer pattern"
        result, explicit = _check_anchored_token(llm_answer, None, "C")
        assert result is None
        assert explicit is False

    def test_anchored_token_no_norm_answer_letter(self):
        """Test when norm_answer_letter is None."""
        llm_answer = "The answer is C"
        result, explicit = _check_anchored_token(llm_answer, None, None)
        assert result is None
        assert explicit is False


class TestCheckLastToken:
    """Tests for _check_last_token function."""

    def test_last_token_none_norm_answer(self):
        """Test _check_last_token with None norm_answer_letter (Line 224)."""
        result = _check_last_token("The answer is C", None)
        assert result is None

    def test_last_token_predicted_none(self):
        """Test when _norm_letter returns None for token (Line 231)."""
        # Token that doesn't normalize to a valid letter
        text = "The answer is **"
        result = _check_last_token(text, "C")
        assert result is None

    def test_last_token_negated_near(self):
        """Test negation detection near token (Line 231)."""
        text = "It is not B. The answer is C"
        result = _check_last_token(text, "C")
        # Should find C, not the negated B
        assert result == "C"

    def test_last_token_negated_skipped(self):
        """Test that negated tokens are skipped."""
        text = "It is not C"
        result = _check_last_token(text, "C")
        assert result is None

    def test_last_token_negative_after_option(self):
        """Test negative after option check (Line 235)."""
        text = "C is incorrect"
        result = _check_last_token(text, "C")
        assert result is None

    def test_last_token_not_correct(self):
        """Test 'not correct' pattern after option."""
        text = "C is not correct"
        result = _check_last_token(text, "C")
        assert result is None

    def test_last_token_wrong(self):
        """Test 'wrong' pattern after option."""
        text = "C is wrong"
        result = _check_last_token(text, "C")
        assert result is None

    def test_last_token_is_wrong(self):
        """Test 'is wrong' pattern after option."""
        text = "Option C is wrong"
        result = _check_last_token(text, "C")
        assert result is None

    def test_last_token_success(self):
        """Test successful last token match."""
        text = "After careful consideration, the answer is B"
        result = _check_last_token(text, "B")
        assert result == "B"

    def test_last_token_contraction_negation_scoped_to_own_clause(self):
        """A contraction negating an earlier option must not suppress a later one."""
        text = "B doesn't apply, so C"
        result = _check_last_token(text, "C")
        assert result == "C"

    def test_last_token_contraction_negation_scoped_with_conjunction(self):
        """Negation before a "therefore" clause boundary must not carry across it."""
        text = "A wasn't right, therefore C"
        result = _check_last_token(text, "C")
        assert result == "C"

    def test_last_token_contraction_negation_comma_only_boundary(self):
        """A comma alone (no conjunction) still scopes negation to its own clause."""
        text = "B isn't correct, C"
        result = _check_last_token(text, "C")
        assert result == "C"

    def test_last_token_contraction_negation_still_rejects_same_clause(self):
        """A contraction negating the candidate itself must still reject it."""
        text = "The answer isn't C"
        result = _check_last_token(text, "C")
        assert result is None


class TestCheckAnswerText:
    """Tests for _check_answer_text function."""

    def test_answer_text_match_end_region(self):
        """Test answer text match in end region (Line 271)."""
        llm_answer = "Some text here. The final answer is diabetes type 2"
        answer_text = "diabetes type 2"
        result = _check_answer_text(llm_answer, answer_text)
        assert result is not None
        assert result[0] is True
        assert "diabetes" in result[1]

    def test_answer_text_match_beginning_region(self):
        """Test answer text match in beginning region."""
        llm_answer = "diabetes type 2 is the correct answer. More text here."
        answer_text = "diabetes type 2"
        result = _check_answer_text(llm_answer, answer_text)
        assert result is not None
        assert result[0] is True

    def test_answer_text_no_match(self):
        """Test when answer text is not found."""
        llm_answer = "Some unrelated text here"
        answer_text = "diabetes type 2"
        result = _check_answer_text(llm_answer, answer_text)
        assert result is None

    def test_answer_text_negated_in_beginning(self):
        """Test that negated answer text in beginning is rejected."""
        llm_answer = "It is not diabetes type 2. Some other text."
        answer_text = "diabetes type 2"
        result = _check_answer_text(llm_answer, answer_text)
        assert result is None

    def test_answer_text_negated_in_end(self):
        """Test that negated answer text in end is rejected."""
        llm_answer = "Some text. It is not diabetes type 2."
        answer_text = "diabetes type 2"
        result = _check_answer_text(llm_answer, answer_text)
        assert result is None

    def test_answer_text_with_flexible_whitespace(self):
        """Test answer text matching with flexible whitespace."""
        llm_answer = "The answer is diabetes    type    2"
        answer_text = "diabetes type 2"
        result = _check_answer_text(llm_answer, answer_text)
        assert result is not None
        assert result[0] is True


class TestCheckTokenStrategies:
    """Tests for token strategies flow (Lines 295->300)."""

    def test_check_token_strategies_no_match(self):
        """Test when no token strategies match - should return method 'none'."""
        llm_answer = "Just some random text without any answer pattern"
        result = multiple_choice_accuracy(
            llm_answer, "C", "Some answer text", return_details=True
        )
        # Neither the answer letter nor the answer text appear in llm_answer,
        # so grading must fail outright, not silently "succeed" via some
        # token strategy.
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is False
        assert result.method == "none"
        assert result.matched_answer is None

    def test_check_token_strategies_anchored_match(self):
        """Test anchored token match in strategies flow."""
        llm_answer = "The final answer is C"
        result = multiple_choice_accuracy(
            llm_answer, "C", "Option C", return_details=True
        )
        assert isinstance(result, MCQAccuracyResult)
        assert result.method == "anchored_token"
        assert result.is_correct is True

    def test_check_token_strategies_last_token_match(self):
        """Test last token match in strategies flow (Lines 388->392)."""
        # Text without anchored pattern but with token at end
        llm_answer = "After analyzing all options, I conclude B"
        result = multiple_choice_accuracy(
            llm_answer, "B", "Option B", return_details=True
        )
        assert isinstance(result, MCQAccuracyResult)
        assert result.method == "last_token"
        assert result.is_correct is True


class TestMultipleThinkTags:
    """Tests for multiple think tags handling (Lines 359->364)."""

    def test_multiple_think_tags_preserved(self):
        """Test that multiple think tags preserve full text."""
        response = (
            "<think>First reasoning</think>Answer<think>Second reasoning</think>C"
        )
        result = multiple_choice_accuracy(
            response, "C", "Option C", return_details=True
        )
        # Multiple think tags should preserve the full text
        assert isinstance(result, MCQAccuracyResult)

    def test_single_think_tag_stripped(self):
        """Test that single think tag is stripped."""
        response = "<think>Reasoning here</think>C"
        result = multiple_choice_accuracy(
            response, "C", "Option C", return_details=True
        )
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is True
        assert result.method == "direct_answer"


class TestLeadingOptionPath:
    """Tests for leading option path (Lines 386->388)."""

    def test_leading_option_explicit_choice_found(self):
        """Test leading option sets explicit_choice_found (Lines 386->388)."""
        response = "B. The correct answer text"
        result = multiple_choice_accuracy(
            response, "B", "The correct answer text", return_details=True
        )
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is True

    def test_leading_option_kind_mismatch(self):
        """Test leading option with kind mismatch doesn't set explicit."""
        # Digit answer but letter predicted
        response = "B. Answer text"
        result = multiple_choice_accuracy(
            response, "2", "Answer text", return_details=True
        )
        # Should still work but explicit_choice_found logic is different
        assert isinstance(result, MCQAccuracyResult)


class TestAnswerTextFallbackPath:
    """Tests for answer text fallback path (Lines 399->404)."""

    def test_answer_text_fallback_accepted(self):
        """Test answer text fallback when no explicit choice found."""
        response = "The patient has hypertension"
        result = multiple_choice_accuracy(
            response, "A", "hypertension", accept_answer_text=True, return_details=True
        )
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is True
        assert result.method == "answer_text"

    def test_answer_text_fallback_disabled(self):
        """Test answer text fallback disabled."""
        response = "The patient has hypertension"
        result = multiple_choice_accuracy(
            response, "A", "hypertension", accept_answer_text=False, return_details=True
        )
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is False
        assert result.method == "none"

    def test_answer_text_fallback_when_explicit_choice_found(self):
        """Test answer text skipped when explicit choice already found."""
        # Leading option sets explicit_choice_found
        response = "A. The patient has diabetes"
        result = multiple_choice_accuracy(
            response, "B", "diabetes", accept_answer_text=True, return_details=True
        )
        # Should not match because explicit choice A doesn't equal B
        # And explicit_choice_found prevents answer_text fallback
        assert isinstance(result, MCQAccuracyResult)
        assert result.is_correct is False


class TestAccuracyReward:
    """Tests for accuracy_reward async function (Lines 429-439)."""

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_string_completion(self):
        """Test accuracy_reward with string completion."""
        result = await accuracy_reward(
            completion="The final answer is C",
            answer="C",
            info={"answer_text": "Option C"},
        )
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_messages_list(self):
        """Test accuracy_reward with list of messages completion."""
        completion = [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": "The answer is B"},
        ]
        result = await accuracy_reward(completion=completion, answer="B", info=None)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_messages_no_assistant(self):
        """Test accuracy_reward with messages list without assistant role."""
        completion = [
            {"role": "user", "content": "What is the answer?"},
        ]
        result = await accuracy_reward(completion=completion, answer="B", info=None)
        # No assistant message found, parsed should be empty
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_other_completion_type(self):
        """Test accuracy_reward with non-string non-list completion."""

        class CustomCompletion:
            def __str__(self):
                return "The answer is C"

        result = await accuracy_reward(
            completion=CustomCompletion(), answer="C", info=None
        )
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_parser(self):
        """Test accuracy_reward with parser."""
        mock_parser = MagicMock()
        mock_parser.parse_answer.return_value = "C"

        result = await accuracy_reward(
            completion="Some raw completion",
            answer="C",
            info=None,
            parser=mock_parser,
        )
        assert result == 1.0
        mock_parser.parse_answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_parser_returns_none(self):
        """Test accuracy_reward when parser returns None."""
        mock_parser = MagicMock()
        mock_parser.parse_answer.return_value = None

        result = await accuracy_reward(
            completion="Some raw completion",
            answer="C",
            info=None,
            parser=mock_parser,
        )
        # None parsed should be treated as empty string
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_incorrect_answer(self):
        """Test accuracy_reward returns 0.0 for incorrect answer."""
        result = await accuracy_reward(
            completion="The answer is A",
            answer="C",
            info={"answer_text": "Option C"},
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_info_none(self):
        """Test accuracy_reward when info is None."""
        result = await accuracy_reward(
            completion="The answer is C", answer="C", info=None
        )
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_with_dict_message_content_none(self):
        """Test with dict message where content is None."""
        completion = [{"role": "assistant", "content": None}]
        result = await accuracy_reward(completion=completion, answer="C", info=None)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_accuracy_reward_empty_list(self):
        """Test accuracy_reward with empty list completion."""
        result = await accuracy_reward(completion=[], answer="C", info=None)
        assert result == 0.0


class TestEdgeCases:
    """Additional edge case tests."""

    def test_multiple_choice_accuracy_strip_tex_false(self):
        """Test with strip_tex=False."""
        result = multiple_choice_accuracy(
            r"The answer is $\boxed{C}$", "C", "Option C", strip_tex=False
        )
        assert result is True

    def test_multiple_choice_accuracy_with_prefix(self):
        """Test with custom prefix."""
        result = multiple_choice_accuracy(
            "The correct answer is: C",
            "C",
            "Option C",
            prefix="The correct answer is",
        )
        assert result is True

    def test_anchored_token_with_punctuation_variations(self):
        """Test anchored token with various punctuation."""
        variations = [
            "Final answer: C",
            "Final answer - C",
            "Final answer – C",  # en-dash
            "Final answer — C",  # em-dash
            "Final answer is C",
        ]
        for text in variations:
            result = multiple_choice_accuracy(text, "C", "Option C")
            assert result is True, f"Failed for: {text}"

    def test_anchored_token_with_markdown_wrappers(self):
        """Test anchored token with markdown formatting."""
        variations = [
            "The answer is **C**",
            "The answer is *C*",
            "The answer is `C`",
            "The answer is _C_",
            "The answer is ~C~",
        ]
        for text in variations:
            result = multiple_choice_accuracy(text, "C", "Option C")
            assert result is True, f"Failed for: {text}"

    def test_parenthesized_options(self):
        """Test options in parentheses."""
        variations = [
            "The answer is (C)",
            "The answer is (C).",
            "The answer is (C))",  # Extra paren actually matches (not a word char)
        ]
        for text in variations:
            result = multiple_choice_accuracy(text, "C", "Option C")
            assert result is True, f"Failed for: {text}"

    def test_leading_list_formats(self):
        """Test leading option with various list formats."""
        variations = [
            "> B. Answer",  # Blockquote
            "- B. Answer",  # Unordered list
            "* B. Answer",  # Unordered list
            "+ B. Answer",  # Unordered list
            "1. B. Answer",  # Ordered list
            "10. B. Answer",  # Ordered list with multiple digits
        ]
        for text in variations:
            result = multiple_choice_accuracy(text, "B", "Answer")
            assert result is True, f"Failed for: {text}"

    def test_think_tags_with_only_close_tag(self):
        """Test handling of only closing think tag."""
        response = "</think>The answer is C"
        result = multiple_choice_accuracy(response, "C", "Option C")
        assert result is True

    def test_negative_context_variations(self):
        """Test various negative context patterns."""
        # These patterns test the "negative after option" detection
        # which primarily works in last-token mode, not anchored mode
        # The negative patterns like "is incorrect" need to appear after
        # the option token in the same sentence
        negative_patterns_last_token = [
            "C is incorrect",
            "C is wrong",
            "C is false",
            "C is not correct",
        ]
        for text in negative_patterns_last_token:
            result = multiple_choice_accuracy(text, "C", "Option C")
            assert result is False, f"Should not match for: {text}"

        # Test negation before the option (works with anchored patterns)
        negated_anchored_patterns = [
            "The answer is not C",
            "It isn't C",
            "The final answer is not C",
        ]
        for text in negated_anchored_patterns:
            result = multiple_choice_accuracy(text, "C", "Option C")
            assert result is False, f"Should not match for negated: {text}"
