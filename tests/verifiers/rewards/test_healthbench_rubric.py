"""Tests for HealthBench rubric reward function."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from med_reason_evals.verifiers.rewards.judge_rubric import (
    HEALTHBENCH_JUDGE_TEMPLATE,
    format_conversation,
    healthbench_rubric_reward,
    parse_json_response,
)


class TestParseJsonResponse:
    """Tests for parse_json_response function."""

    def test_json_in_markdown_code_block(self):
        """JSON wrapped in markdown code block should be parsed."""
        text = '```json\n{"criteria_met": true, "explanation": "Good answer"}\n```'
        result = parse_json_response(text)
        assert result == {"criteria_met": True, "explanation": "Good answer"}

    def test_json_in_markdown_code_block_without_language(self):
        """JSON in markdown code block without json tag should be parsed."""
        text = '```\n{"criteria_met": false, "explanation": "Poor answer"}\n```'
        result = parse_json_response(text)
        assert result == {"criteria_met": False, "explanation": "Poor answer"}

    def test_plain_json_object(self):
        """Plain JSON object should be parsed."""
        text = '{"criteria_met": true, "explanation": "Correct"}'
        result = parse_json_response(text)
        assert result == {"criteria_met": True, "explanation": "Correct"}

    def test_json_extraction_from_mixed_text(self):
        """JSON embedded in surrounding text should be extracted."""
        text = 'Based on my analysis, the result is {"criteria_met": true, "explanation": "Valid"} which means it passes.'
        result = parse_json_response(text)
        assert result.get("criteria_met") is True
        assert result.get("explanation") == "Valid"

    def test_json_with_nested_quotes(self):
        """JSON with escaped quotes should be handled."""
        text = '{"criteria_met": true, "explanation": "The answer \\"works\\" well"}'
        result = parse_json_response(text)
        assert result.get("criteria_met") is True

    def test_malformed_json_returns_empty_dict(self):
        """Malformed JSON should return empty dict."""
        text = "{not valid json"
        result = parse_json_response(text)
        assert result == {}

    def test_no_json_returns_empty_dict(self):
        """Text without JSON should return empty dict."""
        text = "This is just plain text without any JSON."
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

    def test_json_with_newlines(self):
        """JSON with newlines should be parsed."""
        text = """```json
{
    "criteria_met": true,
    "explanation": "Multi-line explanation"
}
```"""
        result = parse_json_response(text)
        assert result.get("criteria_met") is True


class TestFormatConversation:
    """Tests for format_conversation function."""

    def test_single_user_message(self):
        """Single user message should be formatted correctly."""
        prompt = [{"role": "user", "content": "What is the diagnosis?"}]
        completion_text = "The diagnosis is diabetes."
        result = format_conversation(prompt, completion_text)
        assert "user: What is the diagnosis?" in result
        assert "assistant: The diagnosis is diabetes." in result

    def test_multiple_messages(self):
        """Multiple messages should be formatted in order."""
        prompt = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": "Patient has high blood sugar."},
        ]
        completion_text = "Consider testing for diabetes."
        result = format_conversation(prompt, completion_text)
        assert "system: You are a helpful medical assistant." in result
        assert "user: Patient has high blood sugar." in result
        assert "assistant: Consider testing for diabetes." in result
        # Verify order
        system_idx = result.index("system:")
        user_idx = result.index("user:")
        assistant_idx = result.index("assistant:")
        assert system_idx < user_idx < assistant_idx

    def test_completion_appended_as_assistant(self):
        """Completion text should be appended as assistant role."""
        prompt = [{"role": "user", "content": "Question"}]
        completion_text = "Answer"
        result = format_conversation(prompt, completion_text)
        assert result.endswith("assistant: Answer")

    def test_messages_separated_by_double_newline(self):
        """Messages should be separated by double newlines."""
        prompt = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ]
        completion_text = "Third"
        result = format_conversation(prompt, completion_text)
        assert "\n\n" in result

    def test_empty_prompt_list(self):
        """Empty prompt list should only have assistant message."""
        prompt: list[dict[str, str]] = []
        completion_text = "Just the completion"
        result = format_conversation(prompt, completion_text)
        assert result == "assistant: Just the completion"

    def test_message_with_missing_role(self):
        """Message without role should be skipped."""
        prompt = [
            {"content": "No role here"},
            {"role": "user", "content": "Has role"},
        ]
        completion_text = "Response"
        result = format_conversation(prompt, completion_text)
        assert "No role here" not in result
        assert "user: Has role" in result

    def test_message_with_missing_content(self):
        """Message without content should be skipped."""
        prompt = [
            {"role": "user"},
            {"role": "user", "content": "Has content"},
        ]
        completion_text = "Response"
        result = format_conversation(prompt, completion_text)
        # Count user occurrences - should only have one
        assert result.count("user:") == 1
        assert "user: Has content" in result

    def test_non_list_prompt(self):
        """Non-list prompt should result in just the completion."""
        prompt = "Just a string"  # type: ignore
        completion_text = "Response"
        result = format_conversation(prompt, completion_text)
        assert result == "assistant: Response"


class TestHealthBenchRubricReward:
    """Tests for healthbench_rubric_reward function."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_criteria(self):
        """Should return 0.0 when no criteria in info."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {"points_list": [1, 2, 3]}
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_points_list(self):
        """Should return 0.0 when no points_list in info."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {"criteria": ["Criterion 1", "Criterion 2"]}
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_criteria_empty(self):
        """Should return 0.0 when criteria list is empty."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {"criteria": [], "points_list": []}
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_total_positive_is_zero(self):
        """Should return 0.0 when total positive points is 0."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Negative criterion 1", "Negative criterion 2"],
            "points_list": [-1, -2],  # All negative points
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_normalization_uses_positive_points_only(self):
        """Score normalization should only use positive points."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        # Mix of positive and negative points
        # Positive total = 3 + 2 = 5
        info: dict[str, Any] = {
            "criteria": ["Positive 1", "Negative 1", "Positive 2"],
            "points_list": [3, -2, 2],
        }
        state: dict[str, Any] = {}

        # All criteria met
        async def mock_judge(**kwargs):
            return '{"criteria_met": true, "explanation": "Met"}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # Earned: 3 + (-2) + 2 = 3 (negative points count when met)
        # But normalization uses positive_total = 5
        # Score = 3/5 = 0.6
        # Wait, let's check the code - it sums points_possible if criteria_met
        # So earned = 3 + (-2) + 2 = 3, but total_positive = 5
        # Actually, looking at the code:
        # earned_points = sum(j["points_possible"] if j["criteria_met"] else 0
        #                     for j in judgments)
        # So if all met: 3 + (-2) + 2 = 3
        # Score = 3/5 = 0.6
        assert result == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_correct_score_calculation_all_pass(self):
        """Should calculate correct score when all criteria pass."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Criterion 1", "Criterion 2", "Criterion 3"],
            "points_list": [1, 2, 3],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true, "explanation": "Good"}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # All pass: earned = 1 + 2 + 3 = 6, total = 6
        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_correct_score_calculation_partial_pass(self):
        """Should calculate correct score when some criteria pass."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Criterion 1", "Criterion 2", "Criterion 3"],
            "points_list": [2, 3, 5],
        }
        state: dict[str, Any] = {}

        call_count = 0

        async def mock_judge(**kwargs):
            nonlocal call_count
            call_count += 1
            # First and third pass, second fails
            if call_count == 2:
                return '{"criteria_met": false, "explanation": "Failed"}'
            return '{"criteria_met": true, "explanation": "Passed"}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # Earned: 2 + 0 + 5 = 7, total = 10
        assert result == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_correct_score_calculation_all_fail(self):
        """Should return 0.0 when all criteria fail."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Criterion 1", "Criterion 2"],
            "points_list": [5, 5],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": false, "explanation": "Failed"}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_handles_negative_points_correctly(self):
        """Negative points should be included in earned calculation."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        # Scenario: positive criterion fails, negative criterion met
        info: dict[str, Any] = {
            "criteria": ["Must mention X", "Must not mention Y"],
            "points_list": [5, -2],  # -2 is a penalty criterion
        }
        state: dict[str, Any] = {}

        call_count = 0

        async def mock_judge(**kwargs):
            nonlocal call_count
            call_count += 1
            # First (positive) fails, second (negative/penalty) is met
            if call_count == 1:
                return '{"criteria_met": false}'
            return '{"criteria_met": true}'  # Penalty criterion met

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # Earned: 0 + (-2) = -2, total_positive = 5
        # Score = -2/5 = -0.4, but clamped to [0, 1]
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_zero_one(self):
        """Score should be clamped to [0, 1] range."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        # All negative points met = negative total
        info: dict[str, Any] = {
            "criteria": ["Positive", "Negative penalty"],
            "points_list": [2, -5],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # Earned: 2 + (-5) = -3, total_positive = 2
        # Score = -3/2 = -1.5, clamped to 0.0
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_rubric_results_populated(self):
        """info['rubric_results'] should be populated with judgment details."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["First criterion", "Second criterion"],
            "points_list": [3, 5],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true, "explanation": "Good job"}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )

        assert "rubric_results" in info
        assert len(info["rubric_results"]) == 2

        # Check structure of each result
        for result in info["rubric_results"]:
            assert "idx" in result
            assert "points_possible" in result
            assert "criteria_met" in result
            assert "judge_explanation" in result

    @pytest.mark.asyncio
    async def test_rubric_results_shape_consistency(self):
        """rubric_results should have consistent shape across criteria."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2", "Crit3"],
            "points_list": [1, 2, 3],
        }
        state: dict[str, Any] = {}

        call_count = 0

        async def mock_judge(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return '{"criteria_met": false}'
            return '{"criteria_met": true, "explanation": "Explanation"}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )

        results = info["rubric_results"]
        assert len(results) == 3

        # All results should have same keys
        expected_keys = {"idx", "points_possible", "criteria_met", "judge_explanation"}
        for r in results:
            assert set(r.keys()) == expected_keys

        # Check indices are correct
        indices = {r["idx"] for r in results}
        assert indices == {0, 1, 2}

        # Check points match
        points_by_idx = {r["idx"]: r["points_possible"] for r in results}
        assert points_by_idx[0] == 1
        assert points_by_idx[1] == 2
        assert points_by_idx[2] == 3

    @pytest.mark.asyncio
    async def test_semaphore_limits_parallel_judges(self):
        """Semaphore should limit concurrent judge calls."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": [f"Criterion {i}" for i in range(10)],
            "points_list": [1] * 10,
        }
        state: dict[str, Any] = {}

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_judge(**kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)  # Simulate some work
            async with lock:
                current_concurrent -= 1
            return '{"criteria_met": true}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=tracking_judge,
            max_parallel_judges=3,
        )

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_default_max_parallel_judges(self):
        """Default max_parallel_judges should be 5."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": [f"Criterion {i}" for i in range(15)],
            "points_list": [1] * 15,
        }
        state: dict[str, Any] = {}

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracking_judge(**kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return '{"criteria_met": true}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=tracking_judge,
            # Not passing max_parallel_judges, should use default of 5
        )

        assert max_concurrent <= 5

    @pytest.mark.asyncio
    async def test_completion_as_list_with_content(self):
        """Completion as list should extract last message content."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [
            {"role": "assistant", "content": "First response"},
            {"role": "assistant", "content": "Final response"},
        ]
        info: dict[str, Any] = {
            "criteria": ["Check response"],
            "points_list": [1],
        }
        state: dict[str, Any] = {}

        received_prompt = None

        async def capturing_judge(prompt, **kwargs):
            nonlocal received_prompt
            received_prompt = prompt
            return '{"criteria_met": true}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=capturing_judge,
        )

        # The formatted conversation should include "Final response"
        assert received_prompt is not None
        prompt_content = received_prompt[0]["content"]
        assert "Final response" in prompt_content

    @pytest.mark.asyncio
    async def test_completion_as_string(self):
        """Completion as string should be used directly."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = "Direct string completion"  # type: ignore
        info: dict[str, Any] = {
            "criteria": ["Check response"],
            "points_list": [1],
        }
        state: dict[str, Any] = {}

        received_prompt = None

        async def capturing_judge(prompt, **kwargs):
            nonlocal received_prompt
            received_prompt = prompt
            return '{"criteria_met": true}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=capturing_judge,
        )

        assert received_prompt is not None
        prompt_content = received_prompt[0]["content"]
        assert "Direct string completion" in prompt_content

    @pytest.mark.asyncio
    async def test_judge_receives_correct_arguments(self):
        """Judge should receive correct prompt, completion, answer, and state."""
        prompt = [{"role": "user", "content": "Medical question"}]
        completion = [{"role": "assistant", "content": "Medical answer"}]
        info: dict[str, Any] = {
            "criteria": ["Criterion with details"],
            "points_list": [5],
        }
        state: dict[str, Any] = {"key": "value"}

        received_kwargs: dict[str, Any] = {}

        async def capturing_judge(**kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            return '{"criteria_met": true}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=capturing_judge,
        )

        assert "prompt" in received_kwargs
        assert "completion" in received_kwargs
        assert "answer" in received_kwargs
        assert "state" in received_kwargs
        assert received_kwargs["state"] == state
        assert received_kwargs["completion"] == ""
        assert received_kwargs["answer"] == ""

    @pytest.mark.asyncio
    async def test_judge_prompt_contains_template_elements(self):
        """Judge prompt should contain conversation and rubric item."""
        prompt = [{"role": "user", "content": "What is the diagnosis?"}]
        completion = [{"role": "assistant", "content": "Diabetes"}]
        info: dict[str, Any] = {
            "criteria": ["Response should mention diabetes"],
            "points_list": [3],
        }
        state: dict[str, Any] = {}

        received_prompt = None

        async def capturing_judge(prompt, **kwargs):
            nonlocal received_prompt
            received_prompt = prompt
            return '{"criteria_met": true}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=capturing_judge,
        )

        assert received_prompt is not None
        prompt_content = received_prompt[0]["content"]
        # Should contain conversation elements
        assert "What is the diagnosis?" in prompt_content
        assert "Diabetes" in prompt_content
        # Should contain rubric item with points
        assert "[3] Response should mention diabetes" in prompt_content

    @pytest.mark.asyncio
    async def test_rubric_results_appended_not_overwritten(self):
        """Multiple calls should append to rubric_results, not overwrite."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["First"],
            "points_list": [1],
            "rubric_results": [{"existing": "data"}],  # Pre-existing results
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )

        assert len(info["rubric_results"]) == 2
        assert info["rubric_results"][0] == {"existing": "data"}

    @pytest.mark.asyncio
    async def test_handles_unparsable_judge_response(self):
        """Unparsable judge response should result in criteria_met=False."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Criterion"],
            "points_list": [5],
        }
        state: dict[str, Any] = {}

        async def bad_judge(**kwargs):
            return "This is not JSON at all"

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=bad_judge,
        )

        # Should treat as not met when response is unparsable
        assert result == 0.0
        assert info["rubric_results"][0]["criteria_met"] is False

    @pytest.mark.asyncio
    async def test_handles_partial_json_response(self):
        """JSON response without criteria_met should default to False."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Criterion"],
            "points_list": [5],
        }
        state: dict[str, Any] = {}

        async def partial_judge(**kwargs):
            return '{"explanation": "Some explanation but no criteria_met"}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=partial_judge,
        )

        assert result == 0.0
        assert info["rubric_results"][0]["criteria_met"] is False


class TestHealthBenchJudgeTemplate:
    """Tests for HEALTHBENCH_JUDGE_TEMPLATE constant."""

    def test_template_has_conversation_placeholder(self):
        """Template should have {conversation} placeholder."""
        assert "{conversation}" in HEALTHBENCH_JUDGE_TEMPLATE

    def test_template_has_rubric_item_placeholder(self):
        """Template should have {rubric_item} placeholder."""
        assert "{rubric_item}" in HEALTHBENCH_JUDGE_TEMPLATE

    def test_template_can_be_formatted(self):
        """Template should be formattable without errors."""
        result = HEALTHBENCH_JUDGE_TEMPLATE.format(
            conversation="user: Hello\nassistant: Hi",
            rubric_item="[5] Be polite",
        )
        assert "user: Hello" in result
        assert "[5] Be polite" in result

    def test_template_mentions_json_format(self):
        """Template should instruct to return JSON."""
        assert "json" in HEALTHBENCH_JUDGE_TEMPLATE.lower()

    def test_template_mentions_criteria_met(self):
        """Template should mention criteria_met field."""
        assert "criteria_met" in HEALTHBENCH_JUDGE_TEMPLATE

    def test_template_mentions_explanation(self):
        """Template should mention explanation field."""
        assert "explanation" in HEALTHBENCH_JUDGE_TEMPLATE


class TestEdgeCases:
    """Edge case tests for robustness."""

    @pytest.mark.asyncio
    async def test_empty_completion_list(self):
        """Empty completion list should be handled."""
        prompt = [{"role": "user", "content": "Question"}]
        completion: list[dict[str, str]] = []
        info: dict[str, Any] = {
            "criteria": ["Criterion"],
            "points_list": [1],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        # Should not raise, completion_text will be empty string
        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_completion_message_without_content(self):
        """Completion message without content key should be handled."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant"}]  # No content key
        info: dict[str, Any] = {
            "criteria": ["Criterion"],
            "points_list": [1],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_single_criterion(self):
        """Single criterion should work correctly."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Single criterion"],
            "points_list": [10],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_large_number_of_criteria(self):
        """Large number of criteria should be handled."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        num_criteria = 50
        info: dict[str, Any] = {
            "criteria": [f"Criterion {i}" for i in range(num_criteria)],
            "points_list": [1] * num_criteria,
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        assert result == 1.0
        assert len(info["rubric_results"]) == num_criteria

    @pytest.mark.asyncio
    async def test_zero_points_criterion(self):
        """Criterion with zero points should be handled."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Important", "Zero weight", "Also important"],
            "points_list": [5, 0, 5],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # total_positive = 5 + 5 = 10 (0 is not positive)
        # earned = 5 + 0 + 5 = 10
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_mismatched_criteria_points_length(self):
        """Mismatched criteria and points_list lengths should work with zip."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        # zip will stop at shorter list
        info: dict[str, Any] = {
            "criteria": ["Crit1", "Crit2", "Crit3"],
            "points_list": [1, 2],  # Shorter than criteria
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            return '{"criteria_met": true}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # Only first 2 criteria evaluated
        assert len(info["rubric_results"]) == 2
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_criteria_met_as_string_true(self):
        """String 'true' for criteria_met should be handled as True."""
        prompt = [{"role": "user", "content": "Question"}]
        completion = [{"role": "assistant", "content": "Answer"}]
        info: dict[str, Any] = {
            "criteria": ["Criterion"],
            "points_list": [5],
        }
        state: dict[str, Any] = {}

        async def mock_judge(**kwargs):
            # Some judges might return string "true" instead of boolean
            return '{"criteria_met": "true"}'

        result = await healthbench_rubric_reward(
            prompt=prompt,
            completion=completion,
            info=info,
            state=state,
            judge=mock_judge,
        )
        # bool("true") is True since non-empty string
        assert result == 1.0
