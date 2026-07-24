"""Pytest configuration and shared fixtures for verifiers tests.

This module provides:
    - Mock OpenAI client implementations for testing without API calls
    - Parser fixtures (basic, XML, Think, MaybeThink variants)
    - Environment fixtures (SingleTurnEnv, MultiTurnEnv, ToolEnv, StatefulToolEnv)
    - Sample datasets for testing

The MockAsyncOpenAI class is the key testing infrastructure - it simulates
LLM responses by mapping conversation histories to predefined outputs, enabling
deterministic testing of multi-turn interactions and tool use.

Note: All fixtures in this file are scoped to 'function' by default, ensuring
fresh instances for each test to prevent state leakage.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import verifiers as vf
from datasets import Dataset
from verifiers import (
    MaybeThinkParser,
    Messages,
    MultiTurnEnv,
    Parser,
    Rubric,
    SingleTurnEnv,
    State,
    StatefulToolEnv,
    ThinkParser,
    ToolCallError,
    ToolEnv,
    XMLParser,
    stop,
)


@pytest.fixture
def basic_parser():
    """Return a basic Parser instance."""
    return Parser()


@pytest.fixture
def xml_parser():
    """Return an XMLParser instance with common fields."""
    return XMLParser(fields=["reasoning", "answer"], answer_field="answer")


@pytest.fixture
def xml_parser_with_alternatives():
    """Return an XMLParser instance with alternative field names."""
    return XMLParser(fields=["reasoning", ("code", "answer")], answer_field="answer")


@pytest.fixture
def maybe_think_parser():
    """Return a MaybeThinkParser instance."""
    return MaybeThinkParser()


@pytest.fixture
def think_parser():
    """Return a ThinkParser instance."""
    return ThinkParser()


@pytest.fixture
def think_parser_with_extractor():
    """Return a ThinkParser instance with custom extraction function."""

    def extract_boxed(text):
        """Simple boxed answer extractor for testing."""
        import re

        match = re.search(r"\\boxed\{([^}]+)\}", text)
        return match.group(1) if match else text

    return ThinkParser(extract_fn=extract_boxed)


# Async test fixtures for Environment testing


class MockAsyncOpenAI:
    """Mock AsyncOpenAI client that maps conversation inputs to outputs.

        This mock enables deterministic testing of LLM interactions by allowing
    tests to pre-register expected inputs and their corresponding outputs.
        The _messages_to_key method creates hashable keys from message lists,
        enabling lookup-based response simulation rather than fragile string matching.
    """

    def __init__(self):
        # Use dicts for O(1) lookup of predefined responses by conversation history
        self.chat_completions: dict[tuple[str, ...], dict] = {}
        self.text_completions: dict[str, dict] = {}
        self.default_chat_response = "This is a test response"
        self.default_text_response = "This is a test completion"
        self.base_url = "http://localhost/v1/"  # Mirrors real client structure

        # Mirror OpenAI client structure for compatibility with production code
        self.chat = MagicMock()
        self.completions = MagicMock()
        self.chat.completions = MagicMock()

        # Wire async methods to handlers - side_effect ensures proper async behavior
        self.chat.completions.create = AsyncMock(
            side_effect=self._handle_chat_completion
        )
        self.completions.create = AsyncMock(side_effect=self._handle_text_completion)

    def add_chat_response(
        self, messages, response, finish_reason="stop", tool_calls=None
    ):
        """Add a mapped response for specific messages."""
        # Convert messages to a hashable key
        key = self._messages_to_key(messages)
        self.chat_completions[key] = {
            "content": response,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    def add_text_response(self, prompt, response, finish_reason="stop"):
        """Add a mapped response for specific prompt."""
        self.text_completions[prompt] = {
            "text": response,
            "finish_reason": finish_reason,
        }

    def set_default_responses(self, chat_response=None, text_response=None):
        """Set default responses when no mapping found."""
        if chat_response:
            self.default_chat_response = chat_response
        if text_response:
            self.default_text_response = text_response

    async def _handle_chat_completion(self, messages, **kwargs):
        """Handle chat completion requests."""
        key = self._messages_to_key(messages)

        if key in self.chat_completions:
            response_data = self.chat_completions[key]
        else:
            response_data = {
                "content": self.default_chat_response,
                "finish_reason": "stop",
                "tool_calls": None,
            }

        # Create mock response that mimics ChatCompletion
        from openai.types.chat.chat_completion import ChatCompletion, Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        # Create a proper mock that will pass isinstance checks
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock(spec=Choice)
        mock_message = MagicMock(spec=ChatCompletionMessage)

        # Set the attributes
        mock_message.content = response_data["content"]
        mock_message.role = "assistant"
        mock_message.tool_calls = response_data.get("tool_calls")
        mock_choice.message = mock_message
        mock_choice.finish_reason = response_data["finish_reason"]
        mock_choice.index = 0

        mock_response.choices = [mock_choice]
        mock_response.id = "test-id"
        mock_response.model = "test-model"
        mock_response.object = "chat.completion"

        return mock_response

    async def _handle_text_completion(self, prompt, **kwargs):
        """Handle text completion requests."""
        if prompt in self.text_completions:
            response_data = self.text_completions[prompt]
        else:
            response_data = {
                "text": self.default_text_response,
                "finish_reason": "stop",
            }

        # Create mock response that mimics Completion
        from openai.types.completion import Completion
        from openai.types.completion_choice import CompletionChoice

        # Create a proper mock that will pass isinstance checks
        mock_response = MagicMock(spec=Completion)
        mock_choice = MagicMock(spec=CompletionChoice)

        # Set the attributes
        mock_choice.text = response_data["text"]
        mock_choice.finish_reason = response_data["finish_reason"]
        mock_choice.index = 0

        mock_response.choices = [mock_choice]
        mock_response.id = "test-id"
        mock_response.model = "test-model"
        mock_response.object = "text_completion"

        return mock_response

    def _messages_to_key(self, messages):
        """Convert messages list to a hashable key."""
        # Create a simplified representation for hashing
        key_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            key_parts.append(f"{role}:{content}")
        return tuple(key_parts)


@pytest.fixture
def mock_openai_client():
    """Return a mocked AsyncOpenAI client with input-output mapping."""
    return MockAsyncOpenAI()


@pytest.fixture
def sample_dataset():
    """Return a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?", "What is the capital of France?"],
            "answer": ["4", "Paris"],
        }
    )


@pytest.fixture
def sample_chat_dataset():
    """Return a sample dataset with chat format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "What is 2+2?"}],
                [{"role": "user", "content": "What is the capital of France?"}],
            ],
            "answer": ["4", "Paris"],
            "example_id": [0, 1],
        }
    )


@pytest.fixture
def mock_singleturn_env(mock_openai_client, sample_dataset):
    """Return a SingleTurnEnv with mocked client and dataset."""
    return SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_dataset,
        system_prompt="You are a helpful assistant.",
        parser=Parser(),
        rubric=Rubric(),
    )


@pytest.fixture
def mock_singleturn_env_completion(mock_openai_client):
    """Return a SingleTurnEnv for completion format testing."""
    completion_dataset = Dataset.from_dict(
        {
            "prompt": ["Calculate 2+2:", "Name the capital of France:"],
            "answer": ["4", "Paris"],
        }
    )
    return SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=completion_dataset,
        message_type="completion",
        parser=Parser(),
        rubric=Rubric(),
    )


# MultiTurnEnv test fixtures


class SimpleMultiTurnEnv(MultiTurnEnv):
    """Simple concrete implementation of MultiTurnEnv for testing."""

    def __init__(self, completion_condition="answer", **kwargs):
        super().__init__(**kwargs)
        self.completion_condition = (
            completion_condition  # "answer", "max_turns", "error"
        )
        self.env_response_count = 0

    @stop
    async def done_condition(self, state: State) -> bool:
        """Complete when assistant says 'DONE'."""
        if self.completion_condition == "answer" and state["trajectory"]:
            last_completion = state["trajectory"][-1]["completion"]
            if isinstance(last_completion, list) and last_completion:
                return "DONE" in str(last_completion[-1].get("content", ""))
            if isinstance(last_completion, str):
                return "DONE" in last_completion
        return False

    @stop
    async def error_condition(self, state: State) -> bool:
        """Complete on any error."""
        if self.completion_condition == "error" and state["trajectory"]:
            last_completion = state["trajectory"][-1]["completion"]
            if isinstance(last_completion, list) and last_completion:
                return str(last_completion[-1].get("content", "")).startswith("[ERROR]")
            if isinstance(last_completion, str):
                return last_completion.startswith("[ERROR]")
        return False

    async def env_response(self, messages, state, **kwargs) -> Messages:
        """Simple environment response for testing."""
        self.env_response_count += 1

        if self.completion_condition == "answer":
            # Encourage completion after a few turns
            if self.env_response_count >= 2:
                return [{"role": "user", "content": "Please finish with DONE"}]
            return [
                {
                    "role": "user",
                    "content": f"Continue (turn {self.env_response_count})",
                }
            ]
        return [
            {
                "role": "user",
                "content": f"Environment response {self.env_response_count}",
            }
        ]


@pytest.fixture
def mock_multiturn_env(mock_openai_client, sample_chat_dataset):
    """Return a MultiTurnEnv for basic testing."""
    return SimpleMultiTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        max_turns=3,
        completion_condition="answer",
        parser=Parser(),
        rubric=Rubric(),
    )


@pytest.fixture
def mock_multiturn_env_max_turns(mock_openai_client, sample_chat_dataset):
    """Return a MultiTurnEnv that tests max_turns limiting."""
    return SimpleMultiTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        max_turns=2,
        completion_condition="max_turns",  # Never complete naturally
        parser=Parser(),
        rubric=Rubric(),
    )


def square_tool(x: int) -> int:
    return x * x


def faulty_tool() -> None:
    cause = ValueError("failure")
    raise ToolCallError from cause


class BasicToolEnv(ToolEnv):
    """Tool environment that exposes the square tool for tests."""

    def __init__(self, **kwargs):
        super().__init__(tools=[square_tool], **kwargs)


@pytest.fixture
def mock_tool_env(mock_openai_client, sample_chat_dataset):
    return BasicToolEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        parser=Parser(),
        rubric=Rubric(),
    )


def offset_tool(x: int, offset: int) -> int:
    return x + offset


def secret_tool(x: int, secret: int) -> int:
    return x + secret


class ExampleStatefulToolEnv(StatefulToolEnv):
    """Stateful tool environment with an adjustable offset.

    Demonstrates the StatefulToolEnv pattern where tool arguments are dynamically
    modified based on conversation state. This is useful for multi-turn scenarios
    where tool behavior needs to change based on previous interactions.

    The offset parameter is injected into tool calls automatically, simulating
    scenarios like personalized calculations based on user preferences.
    """

    def __init__(self, **kwargs):
        super().__init__(tools=[offset_tool], **kwargs)

    async def setup_state(self, state, **kwargs):
        """Seed state with offset metadata used by tool updates."""
        state = await super().setup_state(state, **kwargs)
        state["offset"] = 3  # Starting offset value
        state["update_calls"] = 0  # Track how often tools are invoked
        return state

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        """Inject the current offset into tool arguments and track usage."""
        state["update_calls"] += 1
        updated_args = {**tool_args, "offset": state["offset"]}
        state["last_tool_args"] = updated_args.copy()
        return updated_args


@pytest.fixture
def mock_stateful_tool_env(mock_openai_client, sample_chat_dataset):
    return ExampleStatefulToolEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        parser=Parser(),
        rubric=Rubric(),
    )


@pytest.fixture
def assert_parser_is_xml():
    """Fixture providing assertion function for XMLParser verification."""

    def _assert_parser_is_xml(parser: vf.Parser, *, has_think: bool = False) -> None:
        assert isinstance(parser, vf.XMLParser), (
            f"Expected XMLParser, got {type(parser).__name__}"
        )
        assert parser.answer_field == "answer"

        field_names = [f[0] for f in parser._fields]

        if has_think:
            assert "think" in field_names, "Expected 'think' field in XMLParser"
            assert "answer" in field_names, "Expected 'answer' field in XMLParser"
            assert field_names == ["think", "answer"], (
                f"Expected fields ['think', 'answer'], got {field_names}"
            )
        else:
            assert "think" not in field_names, (
                "Did not expect 'think' field in XMLParser"
            )
            assert "answer" in field_names, "Expected 'answer' field in XMLParser"
            assert field_names == ["answer"], (
                f"Expected fields ['answer'], got {field_names}"
            )

    return _assert_parser_is_xml


@pytest.fixture
def assert_parser_is_boxed():
    """Fixture providing assertion function for boxed Parser verification."""

    def _assert_parser_is_boxed(parser: vf.Parser, *, has_think: bool = False) -> None:
        assert isinstance(parser, vf.Parser), (
            f"Expected Parser, got {type(parser).__name__}"
        )

        if has_think:
            assert isinstance(parser, vf.ThinkParser), (
                "Expected ThinkParser for boxed format with use_think=True"
            )
        else:
            assert not isinstance(parser, vf.XMLParser), (
                "Did not expect XMLParser for boxed format"
            )
            assert not isinstance(parser, vf.ThinkParser), (
                "Did not expect ThinkParser when use_think=False"
            )

    return _assert_parser_is_boxed


@pytest.fixture
def assert_rubric_has_one_func_weight_one():
    """Fixture providing assertion function for rubric verification."""

    def _assert_rubric_has_one_func_weight_one(
        rubric: vf.Rubric,
        *,
        func_name: str | None = None,
    ) -> None:
        # Handle both Rubric and JudgeRubric (which may be wrapped in RubricGroup)
        evaluator_rubric = rubric.rubrics[0] if hasattr(rubric, "rubrics") else rubric

        assert len(evaluator_rubric.funcs) == 1, (
            f"Expected exactly 1 reward func, got {len(evaluator_rubric.funcs)}"
        )
        assert len(evaluator_rubric.weights) == 1, (
            f"Expected exactly 1 weight, got {len(evaluator_rubric.weights)}"
        )
        assert evaluator_rubric.weights[0] == 1.0, (
            f"Expected weight 1.0, got {evaluator_rubric.weights[0]}"
        )

        if func_name:
            actual_name = evaluator_rubric.funcs[0].__name__
            assert actual_name == func_name, (
                f"Expected func name '{func_name}', got '{actual_name}'"
            )

    return _assert_rubric_has_one_func_weight_one


@pytest.fixture
def assert_env_has_basic_fields():
    """Fixture providing assertion function for environment basic fields."""

    def _assert_env_has_basic_fields(env: vf.Environment) -> None:
        assert isinstance(env, vf.SingleTurnEnv), (
            f"Expected SingleTurnEnv, got {type(env).__name__}"
        )
        assert hasattr(env, "eval_dataset"), "Environment missing eval_dataset"
        assert env.eval_dataset is not None, "eval_dataset should not be None"
        assert hasattr(env, "parser"), "Environment missing parser"
        assert hasattr(env, "rubric"), "Environment missing rubric"
        assert env.system_prompt is not None, "Environment should have system_prompt"

    return _assert_env_has_basic_fields


@pytest.fixture
def assert_judge_rubric_prompt_is_question():
    """Fixture providing assertion function for JudgeRubric prompt verification."""

    def _assert_judge_rubric_prompt_is_question(rubric) -> None:
        assert rubric.judge_prompt == "{question}", (
            f"Expected judge_prompt '{{question}}', got '{rubric.judge_prompt}'"
        )

    return _assert_judge_rubric_prompt_is_question


@pytest.fixture
def assert_async_reward_func():
    """Fixture providing assertion function for async reward function verification."""

    def _assert_async_reward_func(rubric) -> None:
        assert len(rubric.funcs) > 0, "No reward functions in rubric"
        assert asyncio.iscoroutinefunction(rubric.funcs[0]), (
            "Expected async coroutine function for reward"
        )

    return _assert_async_reward_func


@pytest.fixture
def assert_verl_result_shape():
    """Fixture providing assertion function for Verl result shape verification."""

    def _assert_verl_result_shape(
        result: dict,
        *,
        required_keys: list[str],
        expected_dataset: str | None = None,
    ) -> None:
        for key in required_keys:
            assert key in result, f"Missing required key '{key}' in result"

        if expected_dataset:
            assert result.get("dataset") == expected_dataset, (
                f"Expected dataset '{expected_dataset}', got '{result.get('dataset')}'"
            )

    return _assert_verl_result_shape
