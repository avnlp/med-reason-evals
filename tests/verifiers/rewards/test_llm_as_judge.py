"""Tests for LLM-as-judge reward functions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from med_reason_evals.verifiers.rewards.llm_as_judge import (
    MEDCASEREASONING_JUDGE_TEMPLATE,
    PUBHEALTHBENCH_JUDGE_TEMPLATE,
    binary_judge_reward_from_template,
    make_binary_judge_reward,
    parse_yes_no,
)


# Allow Unix sockets for asyncio event loop (required for async tests)
pytestmark = pytest.mark.allow_hosts(["localhost"])


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
        # The \b word boundary in the regex should prevent this
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
        assert parse_yes_no('Based on my analysis: {"criteria_met": true}') is True
        assert parse_yes_no('The result is {"correct": false} due to mismatch') is False

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
        # When both are present, criteria_met should take priority
        assert parse_yes_no('{"criteria_met": true, "correct": false}') is True
        assert parse_yes_no('{"criteria_met": false, "correct": true}') is False

    def test_malformed_json(self):
        """Malformed JSON should not crash and return None."""
        assert parse_yes_no("{criteria_met: true}") is None  # Missing quotes
        assert parse_yes_no('{"criteria_met": }') is None  # Missing value

    def test_yes_no_not_at_start_returns_none(self):
        """yes/no not at start should return None (strict anchoring)."""
        # The regex anchors to start, so these should not match
        result = parse_yes_no("I think yes")
        assert result is None

    def test_newlines_before_yes_no(self):
        """Newlines before yes/no should still work."""
        assert parse_yes_no("\nyes") is True
        assert parse_yes_no("\n\nno") is False


class TestBinaryJudgeRewardFromTemplate:
    """Tests for binary_judge_reward_from_template function."""

    @pytest.mark.asyncio
    async def test_no_prediction_returns_zero(self):
        """If parser extracts no answer, should return 0.0 and log feedback."""
        # Configure parser to return None (no prediction extracted)
        mock_parser = MagicMock(spec=["parse", "answer_field"])
        mock_parser.answer_field = "answer"
        mock_parser.parse.return_value = None

        async def mock_judge(**kwargs):
            return "yes"

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [
            {"role": "assistant", "content": "The diagnosis is diabetes"}
        ]

        result = await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=sample_info,
            parser=mock_parser,
            judge=mock_judge,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 0.0
        assert "judge_feedback" in sample_info
        assert len(sample_info["judge_feedback"]) == 1
        feedback = sample_info["judge_feedback"][0]
        assert feedback["prediction"] is None
        assert feedback["raw_judge"] == "no prediction extracted"
        assert feedback["is_correct"] is False

    @pytest.mark.asyncio
    async def test_judge_returns_yes_gives_reward_one(self):
        """If judge returns yes, should return 1.0."""
        mock_parser = MagicMock()
        mock_parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="diabetes")
        mock_parser.parse.return_value = parsed

        async def judge_yes(**kwargs):
            return "yes"

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [
            {"role": "assistant", "content": "The diagnosis is diabetes"}
        ]

        result = await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=sample_info,
            parser=mock_parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert "judge_feedback" in sample_info
        feedback = sample_info["judge_feedback"][0]
        assert feedback["prediction"] == "diabetes"
        assert feedback["parsed"] is True
        assert feedback["is_correct"] is True

    @pytest.mark.asyncio
    async def test_judge_returns_no_gives_reward_zero(self):
        """If judge returns no, should return 0.0."""
        mock_parser = MagicMock()
        mock_parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="hypertension")
        mock_parser.parse.return_value = parsed

        async def judge_no(**kwargs):
            return "no"

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [
            {"role": "assistant", "content": "The diagnosis is diabetes"}
        ]

        result = await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=sample_info,
            parser=mock_parser,
            judge=judge_no,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 0.0
        feedback = sample_info["judge_feedback"][0]
        assert feedback["prediction"] == "hypertension"
        assert feedback["parsed"] is False
        assert feedback["is_correct"] is False

    @pytest.mark.asyncio
    async def test_judge_returns_unparsable_gives_reward_zero(self):
        """If judge returns unparsable response, should return 0.0."""
        mock_parser = MagicMock()
        mock_parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="diabetes")
        mock_parser.parse.return_value = parsed

        async def judge_unclear(**kwargs):
            return "I'm not sure about this one."

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [
            {"role": "assistant", "content": "The diagnosis is diabetes"}
        ]

        result = await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=sample_info,
            parser=mock_parser,
            judge=judge_unclear,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 0.0
        feedback = sample_info["judge_feedback"][0]
        assert feedback["parsed"] is None
        assert feedback["is_correct"] is False

    @pytest.mark.asyncio
    async def test_info_judge_feedback_populated(self):
        """Verify info['judge_feedback'] is populated with all fields."""
        mock_parser = MagicMock()
        mock_parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="diabetes")
        mock_parser.parse.return_value = parsed

        async def judge_yes(**kwargs):
            return "Yes, the diagnosis matches."

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [
            {"role": "assistant", "content": "The diagnosis is diabetes"}
        ]

        await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=sample_info,
            parser=mock_parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert "judge_feedback" in sample_info
        feedback = sample_info["judge_feedback"][0]
        assert "prediction" in feedback
        assert "answer" in feedback
        assert "raw_judge" in feedback
        assert "parsed" in feedback
        assert "is_correct" in feedback
        assert feedback["answer"] == "diabetes"
        assert feedback["raw_judge"] == "Yes, the diagnosis matches."

    @pytest.mark.asyncio
    async def test_template_formatting(self):
        """Verify template is correctly formatted with prediction and ground_truth."""
        mock_parser = MagicMock()
        mock_parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="hypertension")
        mock_parser.parse.return_value = parsed

        received_prompt = None

        async def capturing_judge(prompt, **kwargs):
            nonlocal received_prompt
            received_prompt = prompt
            return "yes"

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [
            {"role": "assistant", "content": "The diagnosis is diabetes"}
        ]

        await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=sample_info,
            parser=mock_parser,
            judge=capturing_judge,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert received_prompt is not None
        assert "hypertension" in received_prompt
        assert "diabetes" in received_prompt

    @pytest.mark.asyncio
    async def test_parser_with_parse_answer_fallback(self):
        """Parser without XMLParser-style parse should fallback to parse_answer."""
        parser = MagicMock()
        parser.parse.return_value = None  # parse returns None
        parser.parse_answer.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [
            {"role": "assistant", "content": "The diagnosis is diabetes"}
        ]

        result = await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=sample_info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_multiple_calls_append_to_feedback(self):
        """Multiple calls should append to judge_feedback list."""
        mock_parser = MagicMock()
        mock_parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="diabetes")
        mock_parser.parse.return_value = parsed

        async def judge_yes(**kwargs):
            return "yes"

        sample_state = {"trajectory": []}
        info = {}
        sample_completion = [{"role": "assistant", "content": "diabetes"}]

        await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="diabetes",
            state=sample_state,
            info=info,
            parser=mock_parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="hypertension",
            state=sample_state,
            info=info,
            parser=mock_parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert len(info["judge_feedback"]) == 2

    @pytest.mark.asyncio
    async def test_pubhealthbench_template(self):
        """Test with PUBHEALTHBENCH_JUDGE_TEMPLATE."""
        mock_parser = MagicMock()
        mock_parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="true")
        mock_parser.parse.return_value = parsed

        received_prompt = None

        async def capturing_judge(prompt, **kwargs):
            nonlocal received_prompt
            received_prompt = prompt
            return "yes"

        sample_state = {"trajectory": []}
        sample_info = {}
        sample_completion = [{"role": "assistant", "content": "true"}]

        await binary_judge_reward_from_template(
            completion=sample_completion,
            answer="true",
            state=sample_state,
            info=sample_info,
            parser=mock_parser,
            judge=capturing_judge,
            template=PUBHEALTHBENCH_JUDGE_TEMPLATE,
        )

        assert received_prompt is not None
        assert "Predicted answer:" in received_prompt
        assert "Correct answer:" in received_prompt


class TestMakeBinaryJudgeReward:
    """Tests for make_binary_judge_reward factory function."""

    def test_returns_callable(self):
        """Factory should return a callable."""
        reward_func = make_binary_judge_reward(MEDCASEREASONING_JUDGE_TEMPLATE)
        assert callable(reward_func)

    def test_function_has_descriptive_name(self):
        """Returned function should have a descriptive name."""
        reward_func = make_binary_judge_reward(MEDCASEREASONING_JUDGE_TEMPLATE)
        assert "binary_judge_reward" in reward_func.__name__

    def test_function_has_docstring(self):
        """Returned function should have a docstring."""
        reward_func = make_binary_judge_reward(MEDCASEREASONING_JUDGE_TEMPLATE)
        assert reward_func.__doc__ is not None
        assert "template" in reward_func.__doc__.lower()

    @pytest.mark.asyncio
    async def test_created_function_works_correctly(self):
        """Created function should work the same as direct call."""
        reward_func = make_binary_judge_reward(MEDCASEREASONING_JUDGE_TEMPLATE)

        parser = MagicMock()
        parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="diabetes")
        parser.parse.return_value = parsed

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await reward_func(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
        )

        assert result == 1.0
        assert "judge_feedback" in info

    @pytest.mark.asyncio
    async def test_template_is_baked_in(self):
        """Template should be baked into the returned function."""
        custom_template = "Custom: {prediction} vs {ground_truth}"
        reward_func = make_binary_judge_reward(custom_template)

        parser = MagicMock()
        parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="test")
        parser.parse.return_value = parsed

        received_prompt = None

        async def capturing_judge(prompt, **kwargs):
            nonlocal received_prompt
            received_prompt = prompt
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "test"}]

        await reward_func(
            completion=completion,
            answer="expected",
            state=state,
            info=info,
            parser=parser,
            judge=capturing_judge,
        )

        assert received_prompt is not None
        assert received_prompt == "Custom: test vs expected"


class TestTemplateConstants:
    """Tests for template constants."""

    def test_medcasereasoning_template_has_placeholders(self):
        """MEDCASEREASONING_JUDGE_TEMPLATE should have required placeholders."""
        assert "{prediction}" in MEDCASEREASONING_JUDGE_TEMPLATE
        assert "{ground_truth}" in MEDCASEREASONING_JUDGE_TEMPLATE

    def test_pubhealthbench_template_has_placeholders(self):
        """PUBHEALTHBENCH_JUDGE_TEMPLATE should have required placeholders."""
        assert "{prediction}" in PUBHEALTHBENCH_JUDGE_TEMPLATE
        assert "{ground_truth}" in PUBHEALTHBENCH_JUDGE_TEMPLATE

    def test_templates_can_be_formatted(self):
        """Templates should be formattable without errors."""
        result1 = MEDCASEREASONING_JUDGE_TEMPLATE.format(
            prediction="diabetes", ground_truth="diabetes"
        )
        assert "diabetes" in result1

        result2 = PUBHEALTHBENCH_JUDGE_TEMPLATE.format(
            prediction="true", ground_truth="true"
        )
        assert "true" in result2


class TestEdgeCases:
    """Edge case tests for robustness."""

    @pytest.mark.asyncio
    async def test_parser_raises_typeerror_without_last(self):
        """Parser that doesn't support last= should be handled gracefully."""
        parser = MagicMock()
        parser.answer_field = "answer"

        # First call with last= raises TypeError
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")
            return SimpleNamespace(answer="diabetes")

        parser.parse.side_effect = parse_side_effect

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_typeerror_fallback_with_string_result(self, mocker):
        """Cover lines 121-130: TypeError fallback when parsed is a string."""
        parser = MagicMock()
        parser.answer_field = "answer"

        # First call raises TypeError, fallback returns a string directly
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")
            # Fallback returns string directly (lines 129-130)
            return "diabetes"

        parser.parse.side_effect = parse_side_effect

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "diabetes"

    @pytest.mark.asyncio
    async def test_typeerror_fallback_with_namespace_result(self, mocker):
        """Cover lines 124-128: TypeError fallback with SimpleNamespace result."""
        parser = MagicMock()
        parser.answer_field = "answer"

        # First call raises TypeError, fallback returns SimpleNamespace
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")
            # Fallback returns SimpleNamespace with answer attribute
            return SimpleNamespace(answer="diabetes")

        parser.parse.side_effect = parse_side_effect

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "diabetes"

    @pytest.mark.asyncio
    async def test_typeerror_fallback_no_answer_field(self, mocker):
        """Cover lines 125-130: TypeError fallback when answer_field check fails."""
        parser = MagicMock()
        # Set answer_field to a MagicMock (not a string) to fail isinstance check
        parser.answer_field = mocker.MagicMock()

        # First call raises TypeError, fallback returns SimpleNamespace
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")
            return SimpleNamespace(answer="diabetes")

        parser.parse.side_effect = parse_side_effect

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        # Should still work because parsed is a string via fallback (line 129-130)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_typeerror_fallback_missing_attribute(self, mocker):
        """TypeError fallback when hasattr check fails but parsed is a string."""
        parser = MagicMock()
        parser.answer_field = "nonexistent_field"

        # First call raises TypeError, fallback returns string
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")
            # Return string directly when answer field doesn't exist
            return "diabetes"

        parser.parse.side_effect = parse_side_effect

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "diabetes"

    @pytest.mark.asyncio
    async def test_typeerror_fallback_returns_none(self, mocker):
        """TypeError fallback returns None and triggers parse_answer fallback."""
        parser = MagicMock()
        parser.answer_field = "answer"

        # First call raises TypeError, fallback returns None
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")

        parser.parse.side_effect = parse_side_effect
        # parse_answer fallback returns the prediction
        parser.parse_answer.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        parser.parse_answer.assert_called_once_with(completion)

    @pytest.mark.asyncio
    async def test_typeerror_fallback_with_namespace_no_hasattr(self, mocker):
        """TypeError fallback with namespace but missing answer attribute."""
        parser = MagicMock()
        parser.answer_field = "answer"

        # First call raises TypeError, fallback returns SimpleNamespace without answer
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")
            # Return namespace without the answer attribute
            return SimpleNamespace(other_field="diabetes")

        parser.parse.side_effect = parse_side_effect
        # parse_answer fallback
        parser.parse_answer.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_parsed_result_is_string_directly(self):
        """Parser returning string directly should work."""
        parser = MagicMock()
        parser.parse.return_value = "diabetes"  # Direct string, not SimpleNamespace
        # Don't set answer_field so hasattr check fails

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_non_string_prediction_converted(self):
        """Non-string prediction should be converted to string."""
        parser = MagicMock()
        parser.answer_field = "answer"
        # Return an integer as the answer
        parsed = SimpleNamespace(answer=42)
        parser.parse.return_value = parsed

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "42"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="42",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "42"

    @pytest.mark.asyncio
    async def test_judge_arguments_passed_correctly(self):
        """Verify judge receives correct arguments."""
        parser = MagicMock()
        parser.answer_field = "answer"
        parsed = SimpleNamespace(answer="test_prediction")
        parser.parse.return_value = parsed

        received_kwargs = {}

        async def capturing_judge(**kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            return "yes"

        state = {"key": "value"}
        info = {}
        completion = [{"role": "assistant", "content": "test"}]

        await binary_judge_reward_from_template(
            completion=completion,
            answer="test_answer",
            state=state,
            info=info,
            parser=parser,
            judge=capturing_judge,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert "prompt" in received_kwargs
        assert "completion" in received_kwargs
        assert received_kwargs["completion"] == "test_prediction"
        assert "answer" in received_kwargs
        assert received_kwargs["answer"] == "test_answer"
        assert "state" in received_kwargs
        assert received_kwargs["state"] == state


class TestTryBlockSuccessPaths:
    """Tests covering the try block success paths (lines 108-118).

    These tests cover the missing branch coverage gaps 108->133 and 119->133.
    The existing tests trigger the except TypeError block; these tests ensure
    the try block succeeds and exercises all paths within it.
    """

    @pytest.mark.asyncio
    async def test_try_block_success_with_namespace(self):
        """Cover lines 108-118: try block succeeds with SimpleNamespace result.

        Tests the path where parser.parse(completion, last=True) succeeds
        without raising TypeError, and returns a SimpleNamespace with the
        answer_field attribute.
        """
        parser = MagicMock()
        parser.answer_field = "answer"
        # parse returns SimpleNamespace successfully (no TypeError)
        parsed = SimpleNamespace(answer="diabetes")
        parser.parse.return_value = parsed

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "diabetes"
        # Verify parse was called with last=True (try block path)
        parser.parse.assert_called_once_with(completion, last=True)

    @pytest.mark.asyncio
    async def test_try_block_success_with_string_result(self):
        """Cover lines 108-118, 119-130: try block succeeds with string result.

        Tests the path where parser.parse returns a string directly
        and isinstance(parsed, str) is checked in the try block.
        """
        parser = MagicMock()
        parser.answer_field = "answer"
        # parse returns string directly
        parser.parse.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "diabetes"

    @pytest.mark.asyncio
    async def test_try_block_success_parsed_is_none(self):
        """Cover lines 108-118, 132-147: try block succeeds but returns None.

        Tests the path where parser.parse succeeds but returns None,
        triggering the parse_answer fallback.
        """
        parser = MagicMock()
        parser.answer_field = "answer"
        # parse returns None (but no TypeError raised)
        parser.parse.return_value = None
        # parse_answer fallback returns the prediction
        parser.parse_answer.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        parser.parse_answer.assert_called_once_with(completion)

    @pytest.mark.asyncio
    async def test_try_block_success_missing_answer_attribute(self):
        """Cover lines 108-118, 127-130: try block succeeds but missing answer attr.

        Tests the path where parsed is a SimpleNamespace but doesn't have
        the answer_field attribute, so it falls through to the string check.
        """
        parser = MagicMock()
        parser.answer_field = "answer"
        # parse returns SimpleNamespace without the 'answer' attribute
        parsed = SimpleNamespace(wrong_field="diabetes")
        parser.parse.return_value = parsed
        # parse_answer fallback
        parser.parse_answer.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_try_block_success_string_parsed_with_nonstring_answer_field(
        self, mocker
    ):
        """Try block with string parsed and non-string answer_field.

        Tests the path where parsed is a string and answer_field is not a string,
        so the isinstance(answer_field, str) check fails but isinstance(parsed, str)
        succeeds.
        """
        parser = MagicMock()
        # Set answer_field to a MagicMock (not a string)
        parser.answer_field = mocker.MagicMock()
        # parse returns a string directly
        parser.parse.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "diabetes"

    @pytest.mark.asyncio
    async def test_typeerror_branch_with_namespace_no_answer_attr(self):
        """TypeError branch with namespace missing answer.

        Tests the except TypeError path where parsed is a SimpleNamespace but
        doesn't have the answer_field attribute, falling through to parse_answer.
        """
        parser = MagicMock()
        parser.answer_field = "answer"

        # First call raises TypeError, fallback returns SimpleNamespace without answer
        def parse_side_effect(*args, **kwargs):
            if "last" in kwargs:
                raise TypeError("unexpected keyword argument 'last'")
            return SimpleNamespace(other_field="diabetes")

        parser.parse.side_effect = parse_side_effect
        parser.parse_answer.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        parser.parse.assert_any_call(completion, last=True)
        parser.parse.assert_any_call(completion)

    @pytest.mark.asyncio
    async def test_no_parse_method_uses_parse_answer(self):
        """Cover lines 108->133: parser without parse method uses parse_answer fallback.

        Tests the path where parser doesn't have a 'parse' method at all,
        so hasattr(parser, 'parse') returns False and we skip to line 133.
        """
        parser = MagicMock()
        # Remove parse method entirely
        del parser.parse
        # parse_answer fallback returns the prediction
        parser.parse_answer.return_value = "diabetes"

        async def judge_yes(**kwargs):
            return "yes"

        state = {"trajectory": []}
        info = {}
        completion = [{"role": "assistant", "content": "diabetes"}]

        result = await binary_judge_reward_from_template(
            completion=completion,
            answer="diabetes",
            state=state,
            info=info,
            parser=parser,
            judge=judge_yes,
            template=MEDCASEREASONING_JUDGE_TEMPLATE,
        )

        assert result == 1.0
        assert info["judge_feedback"][0]["prediction"] == "diabetes"
        parser.parse_answer.assert_called_once_with(completion)
