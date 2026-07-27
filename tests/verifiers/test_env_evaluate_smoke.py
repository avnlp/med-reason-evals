"""Smoke test for env.evaluate() to ensure verifiers + reward funcs compatibility."""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from med_reason_evals.verifiers.medqa import MedQAEvaluator


# Allow Unix sockets for asyncio event loop (required for async tests)
pytestmark = pytest.mark.allow_hosts(["localhost"])


def create_mock_medqa_dataset() -> Dataset:
    """Create a tiny mock MedQA dataset with 3 examples."""
    return Dataset.from_dict(
        {
            "question": [
                "A 45-year-old patient presents with chest pain. What is the diagnosis?",
                "What drug is used for hypertension?",
                "A patient has elevated blood glucose. What is the condition?",
            ],
            "options": [
                {"A": "Angina", "B": "GERD", "C": "Pneumonia", "D": "Fracture"},
                {
                    "A": "Aspirin",
                    "B": "Lisinopril",
                    "C": "Metformin",
                    "D": "Omeprazole",
                },
                {
                    "A": "Hypoglycemia",
                    "B": "Diabetes",
                    "C": "Anemia",
                    "D": "Hypertension",
                },
            ],
            "answer_idx": ["A", "B", "B"],
        }
    )


@pytest.fixture
def mock_medqa_dataset():
    """Fixture that mocks load_dataset to return a tiny MedQA dataset."""
    mock_dataset = create_mock_medqa_dataset()

    with patch("med_reason_evals.data.medqa.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        yield mock_load


@pytest.fixture
def mock_openai_client_for_evaluate():
    """Create a mock AsyncOpenAI client that returns XML-formatted answers.

    This uses the MockAsyncOpenAI pattern from conftest.py but configured
    specifically for the evaluate() flow.
    """
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    client = MagicMock()
    client.base_url = "http://localhost/v1/"
    client.chat = MagicMock()
    client.chat.completions = MagicMock()

    # Track call count to return different answers for different questions
    call_count = [0]

    # Answers corresponding to the mock dataset:
    # Q1: correct answer is A (Angina)
    # Q2: correct answer is B (Lisinopril)
    # Q3: correct answer is B (Diabetes)
    responses = [
        "<answer>A</answer>",  # Correct for Q1
        "<answer>B</answer>",  # Correct for Q2
        "<answer>B</answer>",  # Correct for Q3
    ]

    async def mock_create(messages, **kwargs):
        """Return responses with correct answers in XML format."""
        idx = call_count[0] % len(responses)
        call_count[0] += 1

        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = responses[idx]
        mock_message.role = "assistant"
        mock_message.tool_calls = None

        mock_choice = MagicMock(spec=Choice)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_choice.index = 0

        mock_response = MagicMock(spec=ChatCompletion)
        mock_response.choices = [mock_choice]
        mock_response.id = f"test-id-{idx}"
        mock_response.model = "test-model"
        mock_response.object = "chat.completion"

        return mock_response

    client.chat.completions.create = mock_create

    return client


@pytest.fixture
def mock_openai_client_partial_correct():
    """Create a mock client that returns one correct and one incorrect answer."""
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    client = MagicMock()
    client.base_url = "http://localhost/v1/"
    client.chat = MagicMock()
    client.chat.completions = MagicMock()

    call_count = [0]

    # Q1: correct answer is A, we return A (correct)
    # Q2: correct answer is B, we return C (incorrect)
    responses = [
        "<answer>A</answer>",  # Correct
        "<answer>C</answer>",  # Incorrect - expected B
    ]

    async def mock_create(messages, **kwargs):
        idx = call_count[0] % len(responses)
        call_count[0] += 1

        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = responses[idx]
        mock_message.role = "assistant"
        mock_message.tool_calls = None

        mock_choice = MagicMock(spec=Choice)
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_choice.index = 0

        mock_response = MagicMock(spec=ChatCompletion)
        mock_response.choices = [mock_choice]
        mock_response.id = f"test-id-{idx}"
        mock_response.model = "test-model"
        mock_response.object = "chat.completion"

        return mock_response

    client.chat.completions.create = mock_create

    return client


class TestEnvEvaluateSmoke:
    """Smoke tests for env.evaluate() integration."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_expected_structure(
        self, mock_medqa_dataset, mock_openai_client_for_evaluate
    ):
        """Test that env.evaluate() returns GenerateOutputs with expected fields."""
        evaluator = MedQAEvaluator(streaming=False)
        results = await evaluator.evaluate(
            client=mock_openai_client_for_evaluate,
            model="test-model",
            num_examples=2,
        )

        # Verify the result structure (GenerateOutputs TypedDict)
        assert results is not None
        assert "prompt" in results
        assert "completion" in results
        assert "answer" in results
        assert "reward" in results
        assert "metrics" in results
        assert "state" in results
        assert "info" in results
        assert "example_id" in results

        # Verify we got the expected number of examples
        assert len(results["prompt"]) == 2
        assert len(results["completion"]) == 2
        assert len(results["answer"]) == 2
        assert len(results["reward"]) == 2

    @pytest.mark.asyncio
    async def test_evaluate_all_correct_rewards(
        self, mock_medqa_dataset, mock_openai_client_for_evaluate
    ):
        """Test that evaluate() returns 1.0 rewards when answers are correct."""
        evaluator = MedQAEvaluator(streaming=False)
        results = await evaluator.evaluate(
            client=mock_openai_client_for_evaluate,
            model="test-model",
            num_examples=2,
        )

        # All answers should be correct (reward = 1.0)
        assert all(r == 1.0 for r in results["reward"]), (
            f"Expected all rewards to be 1.0, got: {results['reward']}"
        )

        # Average reward should be 1.0
        avg_reward = sum(results["reward"]) / len(results["reward"])
        assert avg_reward == 1.0, f"Expected avg reward 1.0, got {avg_reward}"

    @pytest.mark.asyncio
    async def test_evaluate_partial_correct_rewards(
        self, mock_medqa_dataset, mock_openai_client_partial_correct
    ):
        """Test evaluate() with mixed correct/incorrect answers."""
        evaluator = MedQAEvaluator(streaming=False)
        results = await evaluator.evaluate(
            client=mock_openai_client_partial_correct,
            model="test-model",
            num_examples=2,
        )

        # First answer should be correct (1.0), second incorrect (0.0)
        assert results["reward"][0] == 1.0, (
            f"Expected first reward 1.0, got {results['reward'][0]}"
        )
        assert results["reward"][1] == 0.0, (
            f"Expected second reward 0.0, got {results['reward'][1]}"
        )

        # Average reward should be 0.5
        avg_reward = sum(results["reward"]) / len(results["reward"])
        assert avg_reward == 0.5, f"Expected avg reward 0.5, got {avg_reward}"

    @pytest.mark.asyncio
    async def test_evaluate_metrics_present(
        self, mock_medqa_dataset, mock_openai_client_for_evaluate
    ):
        """Test that evaluate() returns metrics dict."""
        evaluator = MedQAEvaluator(streaming=False)
        results = await evaluator.evaluate(
            client=mock_openai_client_for_evaluate,
            model="test-model",
            num_examples=2,
        )

        # Metrics should be a dict
        assert isinstance(results["metrics"], dict)

    @pytest.mark.asyncio
    async def test_evaluate_answers_are_extracted(
        self, mock_medqa_dataset, mock_openai_client_for_evaluate
    ):
        """Test that answers from the dataset are present in results."""
        evaluator = MedQAEvaluator(streaming=False)
        results = await evaluator.evaluate(
            client=mock_openai_client_for_evaluate,
            model="test-model",
            num_examples=2,
        )

        # Answers should be the correct letters from the dataset
        # Based on mock dataset: first two answers are "A" and "B"
        assert results["answer"][0] == "A"
        assert results["answer"][1] == "B"

    @pytest.mark.asyncio
    async def test_evaluate_completions_contain_xml_answer(
        self, mock_medqa_dataset, mock_openai_client_for_evaluate
    ):
        """Test that completions contain the XML-formatted answers."""
        evaluator = MedQAEvaluator(streaming=False)
        results = await evaluator.evaluate(
            client=mock_openai_client_for_evaluate,
            model="test-model",
            num_examples=2,
        )

        # Completions should be lists of messages (or strings)
        for completion in results["completion"]:
            if isinstance(completion, list):
                # Find the assistant message
                assistant_content = None
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        assistant_content = msg.get("content", "")
                        break
                assert assistant_content is not None
                assert "<answer>" in assistant_content
                assert "</answer>" in assistant_content
            elif isinstance(completion, str):
                assert "<answer>" in completion
                assert "</answer>" in completion

    @pytest.mark.asyncio
    async def test_evaluate_with_think_format(self, mock_medqa_dataset):
        """Test evaluate() with think=True format."""
        from openai.types.chat.chat_completion import ChatCompletion, Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        # Create client that returns think+answer format
        client = MagicMock()
        client.base_url = "http://localhost/v1/"
        client.chat = MagicMock()
        client.chat.completions = MagicMock()

        call_count = [0]
        responses = [
            "<think>The patient has chest pain, likely angina.</think><answer>A</answer>",
            "<think>Lisinopril is an ACE inhibitor for hypertension.</think><answer>B</answer>",
        ]

        async def mock_create(messages, **kwargs):
            idx = call_count[0] % len(responses)
            call_count[0] += 1

            mock_message = MagicMock(spec=ChatCompletionMessage)
            mock_message.content = responses[idx]
            mock_message.role = "assistant"
            mock_message.tool_calls = None

            mock_choice = MagicMock(spec=Choice)
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_choice.index = 0

            mock_response = MagicMock(spec=ChatCompletion)
            mock_response.choices = [mock_choice]
            mock_response.id = f"test-id-{idx}"
            mock_response.model = "test-model"
            mock_response.object = "chat.completion"

            return mock_response

        client.chat.completions.create = mock_create

        evaluator = MedQAEvaluator(use_think=True, streaming=False)
        results = await evaluator.evaluate(
            client=client,
            model="test-model",
            num_examples=2,
        )

        # Should still correctly parse answers from think format
        assert all(r == 1.0 for r in results["reward"]), (
            f"Expected all rewards to be 1.0 with think format, got: {results['reward']}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_info_contains_metadata(
        self, mock_medqa_dataset, mock_openai_client_for_evaluate
    ):
        """Test that info field contains answer_text metadata."""
        evaluator = MedQAEvaluator(streaming=False)
        results = await evaluator.evaluate(
            client=mock_openai_client_for_evaluate,
            model="test-model",
            num_examples=2,
        )

        # Info should contain metadata for each example
        assert len(results["info"]) == 2

        # Each info dict should have answer_text from the dataset mapping
        for info in results["info"]:
            assert isinstance(info, dict)
            # The MedQA dataset maps examples with "answer_text" in info
            assert "answer_text" in info
