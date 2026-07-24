"""Tests for Verl LLM-as-judge reward functions.

This module provides comprehensive test coverage for the verl rewards module
that uses LLM judges to evaluate model responses against ground truth answers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from med_reason_evals.verl.rewards.llm_as_judge import (
    DEFAULT_JUDGE_PROMPT,
    compute_score,
    judge_answer,
    make_judge_reward,
)


# Allow Unix sockets for asyncio event loop (required for async tests)
pytestmark = pytest.mark.allow_hosts(["localhost"])


class MockAsyncOpenAI:
    """Mock AsyncOpenAI client for testing judge functions.

        This mock enables deterministic testing of LLM judge interactions by allowing
    tests to pre-register expected prompts and their corresponding responses.
    """

    def __init__(self):
        self.chat_completions: dict[tuple[str, ...], str] = {}
        self.default_response = "yes"
        self.base_url = "http://localhost/v1/"

        # Mirror OpenAI client structure
        self.chat = MagicMock()
        self.completions = MagicMock()
        self.chat.completions = MagicMock()

        # Wire async methods
        self.chat.completions.create = AsyncMock(
            side_effect=self._handle_chat_completion
        )

    def add_response(self, prompt_pattern: str, response: str) -> None:
        """Add a mapped response for a prompt pattern."""
        self.chat_completions[prompt_pattern] = response

    def set_default_response(self, response: str) -> None:
        """Set default response when no pattern matches."""
        self.default_response = response

    async def _handle_chat_completion(
        self, *, model: str, messages: list[dict], **kwargs
    ) -> MagicMock:
        """Handle chat completion requests."""
        content = messages[0]["content"] if messages else ""

        # Find matching response
        response_text = self.default_response
        for pattern, response in self.chat_completions.items():
            if pattern in content:
                response_text = response
                break

        # Create mock response structure
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = response_text
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        return mock_response


@pytest.fixture
def mock_judge_client():
    """Return a mocked AsyncOpenAI client for judge testing."""
    return MockAsyncOpenAI()


class TestJudgeAnswer:
    """Tests for judge_answer function."""

    @pytest.mark.asyncio
    async def test_judge_returns_yes(self, mock_judge_client):
        """Judge returning 'yes' should return True."""
        mock_judge_client.set_default_response("yes")

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_judge_returns_no(self, mock_judge_client):
        """Judge returning 'no' should return False."""
        mock_judge_client.set_default_response("no")

        result = await judge_answer(
            prediction="hypertension",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_judge_returns_yes_with_explanation(self, mock_judge_client):
        """Judge returning 'yes' with explanation should return True."""
        mock_judge_client.set_default_response("Yes, the answers match correctly.")

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_judge_returns_no_with_explanation(self, mock_judge_client):
        """Judge returning 'no' with explanation should return False."""
        mock_judge_client.set_default_response("No, the predictions do not match.")

        result = await judge_answer(
            prediction="hypertension",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_judge_returns_json_true(self, mock_judge_client):
        """Judge returning JSON with correct: true should return True."""
        mock_judge_client.set_default_response('{"correct": true}')

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_judge_returns_json_false(self, mock_judge_client):
        """Judge returning JSON with correct: false should return False."""
        mock_judge_client.set_default_response('{"correct": false}')

        result = await judge_answer(
            prediction="hypertension",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_judge_returns_json_criteria_met(self, mock_judge_client):
        """Judge returning JSON with criteria_met should work."""
        mock_judge_client.set_default_response('{"criteria_met": true}')

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_judge_returns_unparsable_defaults_false(self, mock_judge_client):
        """Unparsable judge response should default to False."""
        mock_judge_client.set_default_response("I'm not sure about this one.")

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_judge_returns_none_content_defaults_false(self, mock_judge_client):
        """None content in response should default to False."""

        async def mock_create(*, model, messages, **kwargs):
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = mock_create

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_judge_returns_empty_string_defaults_false(self, mock_judge_client):
        """Empty string content should default to False."""
        mock_judge_client.set_default_response("")

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_custom_judge_prompt(self, mock_judge_client):
        """Custom judge prompt should be used when provided."""
        custom_prompt = "Compare: {prediction} vs {ground_truth}. Is this correct?"
        mock_judge_client.add_response("Compare: diabetes vs diabetes", "yes")

        result = await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
            judge_prompt=custom_prompt,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_default_judge_prompt_used(self, mock_judge_client):
        """Default judge prompt should be used when no custom prompt provided."""
        received_content = None

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_content
            received_content = messages[0]["content"]
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
        )

        assert received_content is not None
        assert "diabetes" in received_content
        assert "Predicted answer:" in received_content
        assert "Correct answer:" in received_content

    @pytest.mark.asyncio
    async def test_custom_model_passed_to_api(self, mock_judge_client):
        """Custom judge model should be passed to API call."""
        received_model = None

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_model
            received_model = model
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
            judge_model="custom/model-v1",
        )

        assert received_model == "custom/model-v1"

    @pytest.mark.asyncio
    async def test_max_tokens_passed_to_api(self, mock_judge_client):
        """Max tokens parameter should be passed to API call."""
        received_kwargs = {}

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
            max_tokens=50,
        )

        assert received_kwargs.get("max_tokens") == 50

    @pytest.mark.asyncio
    async def test_temperature_passed_to_api(self, mock_judge_client):
        """Temperature parameter should be passed to API call."""
        received_kwargs = {}

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
            temperature=0.5,
        )

        assert received_kwargs.get("temperature") == 0.5

    @pytest.mark.asyncio
    async def test_additional_kwargs_passed_to_api(self, mock_judge_client):
        """Additional kwargs should be passed to API call."""
        received_kwargs = {}

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_kwargs
            received_kwargs = kwargs
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await judge_answer(
            prediction="diabetes",
            ground_truth="diabetes",
            judge_client=mock_judge_client,
            top_p=0.9,
            presence_penalty=0.1,
        )

        assert received_kwargs.get("top_p") == 0.9
        assert received_kwargs.get("presence_penalty") == 0.1


class TestComputeScore:
    """Tests for compute_score function."""

    @pytest.mark.asyncio
    async def test_correct_answer_returns_full_score(self, mock_judge_client):
        """Correct answer should return full score."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_incorrect_answer_returns_format_score(self, mock_judge_client):
        """Incorrect answer should return format_score."""
        mock_judge_client.set_default_response("no")

        result = await compute_score(
            solution_str="<answer>hypertension</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0  # default format_score

    @pytest.mark.asyncio
    async def test_custom_scores_used(self, mock_judge_client):
        """Custom score and format_score should be used."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
            score=0.8,
            format_score=0.2,
        )

        assert result == 0.8

    @pytest.mark.asyncio
    async def test_incorrect_with_custom_format_score(self, mock_judge_client):
        """Incorrect answer with custom format_score."""
        mock_judge_client.set_default_response("no")

        result = await compute_score(
            solution_str="<answer>hypertension</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
            score=1.0,
            format_score=0.25,
        )

        assert result == 0.25

    @pytest.mark.asyncio
    async def test_no_answer_extracted_returns_zero(self, mock_judge_client):
        """No answer extracted should return 0.0."""
        result = await compute_score(
            solution_str="This is just some text without an answer",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_empty_solution_returns_zero(self, mock_judge_client):
        """Empty solution string should return 0.0."""
        result = await compute_score(
            solution_str="",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_none_solution_returns_zero(self, mock_judge_client):
        """None solution should return 0.0."""
        result = await compute_score(
            solution_str=None,  # type: ignore[arg-type]
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_empty_ground_truth_returns_format_score(self, mock_judge_client):
        """Empty ground truth should return format_score."""
        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={},
            judge_client=mock_judge_client,
            format_score=0.5,
        )

        assert result == 0.5

    @pytest.mark.asyncio
    async def test_none_ground_truth_values_return_format_score(
        self, mock_judge_client
    ):
        """None ground truth values should return format_score."""
        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": None, "answer": None},
            judge_client=mock_judge_client,
            format_score=0.5,
        )

        assert result == 0.5

    @pytest.mark.asyncio
    async def test_ground_truth_with_answer_key(self, mock_judge_client):
        """Ground truth with 'answer' key should work."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"answer": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_ground_truth_target_preferred_over_answer(self, mock_judge_client):
        """Ground truth 'target' should be preferred over 'answer'."""
        received_prompt = None

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_prompt
            received_prompt = messages[0]["content"]
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "target_value", "answer": "answer_value"},
            judge_client=mock_judge_client,
        )

        assert "target_value" in received_prompt
        assert "answer_value" not in received_prompt

    @pytest.mark.asyncio
    async def test_ground_truth_as_list_first_element_used(self, mock_judge_client):
        """Ground truth as list should use first element."""
        mock_judge_client.add_response("first_answer", "yes")

        result = await compute_score(
            solution_str="<answer>first_answer</answer>",
            ground_truth={"target": ["first_answer", "second_answer"]},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_ground_truth_empty_list_returns_format_score(
        self, mock_judge_client
    ):
        """Empty list ground truth should return format_score."""
        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": []},
            judge_client=mock_judge_client,
            format_score=0.5,
        )

        assert result == 0.5

    @pytest.mark.asyncio
    async def test_custom_judge_prompt_passed_through(self, mock_judge_client):
        """Custom judge prompt should be passed through to judge_answer."""
        custom_prompt = "Custom evaluation: {prediction} vs {ground_truth}"
        received_prompt = None

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_prompt
            received_prompt = messages[0]["content"]
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
            judge_prompt=custom_prompt,
        )

        assert "Custom evaluation:" in received_prompt


class TestMakeJudgeReward:
    """Tests for make_judge_reward factory function."""

    def test_returns_callable(self):
        """Factory should return a callable."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        assert callable(reward_fn)

    def test_returned_function_is_async(self):
        """Returned function should be async."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        import inspect

        assert inspect.iscoroutinefunction(reward_fn)

    @pytest.mark.asyncio
    async def test_created_function_works_correctly(self, mock_judge_client):
        """Created reward function should work correctly."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        mock_judge_client.set_default_response("yes")

        result = await reward_fn(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_template_is_baked_in(self, mock_judge_client):
        """Template should be baked into the returned function."""
        custom_template = "Custom judge prompt: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(custom_template)

        received_prompt = None

        async def capture_create(*, model, messages, **kwargs):
            nonlocal received_prompt
            received_prompt = messages[0]["content"]
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await reward_fn(
            solution_str="<answer>test_value</answer>",
            ground_truth={"target": "expected_value"},
            judge_client=mock_judge_client,
        )

        assert received_prompt == "Custom judge prompt: test_value vs expected_value"

    @pytest.mark.asyncio
    async def test_custom_extractor_function(self, mock_judge_client):
        """Custom extractor function should be used."""
        template = "Compare: {prediction} vs {ground_truth}"

        def custom_extractor(text: str) -> str | None:
            """Custom extractor that looks for boxed answers."""
            import re

            match = re.search(r"\\boxed\{([^}]+)\}", text)
            return match.group(1) if match else None

        reward_fn = make_judge_reward(template, custom_extractor)
        mock_judge_client.set_default_response("yes")

        result = await reward_fn(
            solution_str=r"The answer is \boxed{42}",
            ground_truth={"target": "42"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_custom_extractor_returns_none(self, mock_judge_client):
        """Custom extractor returning None should return 0.0."""
        template = "Compare: {prediction} vs {ground_truth}"

        def failing_extractor(text: str) -> str | None:
            return None

        reward_fn = make_judge_reward(template, failing_extractor)

        result = await reward_fn(
            solution_str="Some text without the expected pattern",
            ground_truth={"target": "answer"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_default_extractor_used_when_none_provided(self, mock_judge_client):
        """Default extract_answer should be used when no custom extractor provided."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        mock_judge_client.set_default_response("yes")

        result = await reward_fn(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_baked_function_uses_custom_scores(self, mock_judge_client):
        """Baked function should use custom score and format_score."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        mock_judge_client.set_default_response("yes")

        result = await reward_fn(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
            score=0.9,
            format_score=0.1,
        )

        assert result == 0.9

    @pytest.mark.asyncio
    async def test_baked_function_handles_incorrect_answer(self, mock_judge_client):
        """Baked function should handle incorrect answers."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        mock_judge_client.set_default_response("no")

        result = await reward_fn(
            solution_str="<answer>wrong_answer</answer>",
            ground_truth={"target": "correct_answer"},
            judge_client=mock_judge_client,
            score=1.0,
            format_score=0.2,
        )

        assert result == 0.2

    @pytest.mark.asyncio
    async def test_baked_function_uses_answer_key_in_ground_truth(
        self, mock_judge_client
    ):
        """Baked function should work with 'answer' key in ground truth."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        mock_judge_client.set_default_response("yes")

        result = await reward_fn(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"answer": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_baked_function_handles_list_ground_truth(self, mock_judge_client):
        """Baked function should handle list ground truth."""
        template = "Compare: {prediction} vs {ground_truth}"
        reward_fn = make_judge_reward(template)

        mock_judge_client.add_response("first", "yes")

        result = await reward_fn(
            solution_str="<answer>first</answer>",
            ground_truth={"target": ["first", "second", "third"]},
            judge_client=mock_judge_client,
        )

        assert result == 1.0


class TestDefaultJudgePrompt:
    """Tests for the default judge prompt constant."""

    def test_has_prediction_placeholder(self):
        """Default prompt should have {prediction} placeholder."""
        assert "{prediction}" in DEFAULT_JUDGE_PROMPT

    def test_has_ground_truth_placeholder(self):
        """Default prompt should have {ground_truth} placeholder."""
        assert "{ground_truth}" in DEFAULT_JUDGE_PROMPT

    def test_can_be_formatted(self):
        """Default prompt should be formattable."""
        formatted = DEFAULT_JUDGE_PROMPT.format(
            prediction="test_prediction",
            ground_truth="test_ground_truth",
        )

        assert "test_prediction" in formatted
        assert "test_ground_truth" in formatted

    def test_contains_yes_no_instruction(self):
        """Default prompt should ask for yes/no answer."""
        assert "yes/no" in DEFAULT_JUDGE_PROMPT.lower()


class TestIntegrationScenarios:
    """Integration-style tests for common scenarios."""

    @pytest.mark.asyncio
    async def test_medical_diagnosis_scenario_correct(self, mock_judge_client):
        """Medical diagnosis scenario with correct answer."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="""<think>
The patient presents with polyuria, polydipsia, and unexplained weight loss.
Random blood glucose is 240 mg/dL.
</think>
<answer>Type 2 Diabetes Mellitus</answer>""",
            ground_truth={"target": "Type 2 Diabetes Mellitus"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_medical_diagnosis_scenario_incorrect(self, mock_judge_client):
        """Medical diagnosis scenario with incorrect answer."""
        mock_judge_client.set_default_response("no")

        result = await compute_score(
            solution_str="""<think>
The patient has chest pain and elevated troponin.
</think>
<answer>Myocardial Infarction</answer>""",
            ground_truth={"target": "Unstable Angina"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_boxed_answer_extraction(self, mock_judge_client):
        """Answer extraction from boxed LaTeX format."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="""Let's solve this step by step:
x + 5 = 10
x = 10 - 5
x = \\boxed{5}""",
            ground_truth={"target": "5"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_final_answer_extraction(self, mock_judge_client):
        """Answer extraction from 'final answer' pattern."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="""After analyzing the data:
Final answer: A""",
            ground_truth={"target": "A"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_no_extractable_answer(self, mock_judge_client):
        """No extractable answer should return 0.0 without calling judge."""
        call_count = 0

        async def counting_create(*, model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = counting_create

        result = await compute_score(
            solution_str="... --- !!! ??? ###",
            ground_truth={"target": "something"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0
        assert call_count == 0  # Judge should not be called

    @pytest.mark.asyncio
    async def test_pubhealthbench_style_true_false(self, mock_judge_client):
        """PubHealthBench style true/false evaluation."""
        template = """Is the predicted answer correct (yes/no)?
Predicted answer: {prediction}
Correct answer: {ground_truth}
Answer [yes/no]."""

        reward_fn = make_judge_reward(template)
        mock_judge_client.set_default_response("yes")

        result = await reward_fn(
            solution_str="<answer>true</answer>",
            ground_truth={"target": "true"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_factory_multiple_instances(self, mock_judge_client):
        """Multiple factory instances should be independent."""
        template1 = "Template 1: {prediction} vs {ground_truth}"
        template2 = "Template 2: {prediction} vs {ground_truth}"

        reward_fn1 = make_judge_reward(template1)
        reward_fn2 = make_judge_reward(template2)

        received_prompts = []

        async def capture_create(*, model, messages, **kwargs):
            received_prompts.append(messages[0]["content"])
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "yes"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            return mock_response

        mock_judge_client.chat.completions.create = capture_create

        await reward_fn1(
            solution_str="<answer>test</answer>",
            ground_truth={"target": "test"},
            judge_client=mock_judge_client,
        )

        await reward_fn2(
            solution_str="<answer>test</answer>",
            ground_truth={"target": "test"},
            judge_client=mock_judge_client,
        )

        assert len(received_prompts) == 2
        assert received_prompts[0].startswith("Template 1:")
        assert received_prompts[1].startswith("Template 2:")


class TestEdgeCases:
    """Edge case tests for robustness."""

    @pytest.mark.asyncio
    async def test_unicode_content_in_prediction(self, mock_judge_client):
        """Unicode content in prediction should be handled."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="<answer>糖尿病</answer>",
            ground_truth={"target": "糖尿病"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_very_long_prediction(self, mock_judge_client):
        """Very long prediction should be handled."""
        mock_judge_client.set_default_response("yes")

        long_answer = "A" * 400
        result = await compute_score(
            solution_str=f"<answer>{long_answer}</answer>",
            ground_truth={"target": long_answer},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_special_characters_in_ground_truth(self, mock_judge_client):
        """Special characters in ground truth should be handled."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="<answer>test-value_123</answer>",
            ground_truth={"target": "test-value_123"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_multiline_prediction(self, mock_judge_client):
        """Multiline prediction should be handled."""
        mock_judge_client.set_default_response("yes")

        multiline_answer = "Line 1\nLine 2\nLine 3"
        result = await compute_score(
            solution_str=f"<answer>{multiline_answer}</answer>",
            ground_truth={"target": multiline_answer},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_judge_case_insensitive_yes(self, mock_judge_client):
        """Judge returning case-insensitive YES should work."""
        mock_judge_client.set_default_response("YES")

        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
        )

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_judge_case_insensitive_no(self, mock_judge_client):
        """Judge returning case-insensitive NO should work."""
        mock_judge_client.set_default_response("NO")

        result = await compute_score(
            solution_str="<answer>wrong</answer>",
            ground_truth={"target": "correct"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_only_answer_returns_zero(self, mock_judge_client):
        """Whitespace-only answer should return 0.0."""
        result = await compute_score(
            solution_str="   ",
            ground_truth={"target": "something"},
            judge_client=mock_judge_client,
        )

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_zero_scores(self, mock_judge_client):
        """Zero scores should work correctly."""
        mock_judge_client.set_default_response("yes")

        result = await compute_score(
            solution_str="<answer>diabetes</answer>",
            ground_truth={"target": "diabetes"},
            judge_client=mock_judge_client,
            score=0.0,
            format_score=0.0,
        )

        assert result == 0.0
