"""Tests for Verl base evaluator classes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

from med_reason_evals.verl.base import (
    BaseJudgeEvaluator,
    BaseMCQEvaluator,
    BaseVerlEvaluator,
    GroqGenConfig,
    JudgeConfig,
)


class TestGroqGenConfig:
    """Tests for GroqGenConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GroqGenConfig()
        assert config.api_key_env == "GROQ_API_KEY"
        assert config.base_url == "https://api.groq.com/openai/v1"
        assert config.model == "openai/gpt-oss-120b"
        assert config.max_tokens == 2048
        assert config.temperature == 0.0
        assert config.sampling_args == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GroqGenConfig(
            api_key_env="CUSTOM_KEY",
            base_url="https://custom.api.com",
            model="custom-model",
            max_tokens=1024,
            temperature=0.5,
            sampling_args={"top_p": 0.9},
        )
        assert config.api_key_env == "CUSTOM_KEY"
        assert config.base_url == "https://custom.api.com"
        assert config.model == "custom-model"
        assert config.max_tokens == 1024
        assert config.temperature == 0.5
        assert config.sampling_args == {"top_p": 0.9}


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JudgeConfig()
        assert config.api_key_env == "GROQ_API_KEY"
        assert config.base_url == "https://api.groq.com/openai/v1"
        assert config.model == "openai/gpt-oss-120b"
        assert config.max_tokens == 500
        assert config.temperature == 0.0
        assert config.sampling_args == {}


class ConcreteEvaluator(BaseVerlEvaluator):
    """Concrete implementation for testing BaseVerlEvaluator."""

    def _load_dataset(self):
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "Test"}]],
                "ground_truth": [{"answer": "A"}],
            }
        )

    async def _evaluate_example(self, prompt, ground_truth, metadata=None):
        return 1.0

    def _build_result(self, scores, avg_score):
        return {"scores": scores, "avg": avg_score}


class TestBaseVerlEvaluator:
    """Tests for BaseVerlEvaluator."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteEvaluator()
            assert evaluator.streaming is True
            assert evaluator.gen_config.model == "openai/gpt-oss-120b"

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = GroqGenConfig(model="custom-model")
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteEvaluator(gen_config=config)
            assert evaluator.gen_config.model == "custom-model"

    def test_rollouts_property_without_api_key_raises(self):
        """Test that rollouts property raises without API key."""
        with patch.dict("os.environ", {}, clear=True):
            evaluator = ConcreteEvaluator()
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                _ = evaluator.rollouts

    def test_rollouts_property_with_api_key(self):
        """Test that rollouts property works with API key."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteEvaluator()
            rollouts = evaluator.rollouts
            assert rollouts is not None
            assert rollouts.model == "openai/gpt-oss-120b"

    def test_rollouts_lazy_initialization(self):
        """Test that rollouts are lazily initialized."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteEvaluator()
            assert evaluator._rollouts is None
            _ = evaluator.rollouts
            assert evaluator._rollouts is not None

    @pytest.mark.asyncio
    async def test_evaluate(self):
        """Test the evaluate method."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteEvaluator()
            result = await evaluator.evaluate(num_examples=1)
            assert "scores" in result
            assert "avg" in result
            assert result["avg"] == 1.0


class ConcreteMCQEvaluator(BaseMCQEvaluator):
    """Concrete MCQ evaluator for testing."""

    def _load_dataset(self):
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "Test"}]],
                "ground_truth": [{"answer": "A"}],
            }
        )

    def _build_result(self, scores, avg_score):
        return {"scores": scores, "avg": avg_score}


class TestBaseMCQEvaluator:
    """Tests for BaseMCQEvaluator."""

    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator(system_prompt="Custom prompt")
            assert evaluator.system_prompt == "Custom prompt"

    def test_init_without_system_prompt(self):
        """Test initialization without system prompt."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator()
            assert evaluator.system_prompt is None

    def test_build_messages_with_system_prompt(self):
        """Test building messages with system prompt."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator(system_prompt="System prompt")
            prompt = [{"role": "user", "content": "Question"}]
            messages = evaluator._build_messages(prompt)
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "System prompt"
            assert messages[1]["role"] == "user"

    def test_build_messages_without_system_prompt(self):
        """Test building messages without system prompt."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator()
            prompt = [{"role": "user", "content": "Question"}]
            messages = evaluator._build_messages(prompt)
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_evaluate_example(self):
        """Test evaluating a single example."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator()

            # Mock rollouts.generate
            mock_response = "<answer>A</answer>"
            evaluator._rollouts = MagicMock()
            evaluator._rollouts.generate = AsyncMock(return_value=mock_response)

            prompt = [{"role": "user", "content": "Question"}]
            ground_truth = {"answer": "A"}

            score = await evaluator._evaluate_example(prompt, ground_truth)
            assert score == 1.0


class ConcreteJudgeEvaluator(BaseJudgeEvaluator):
    """Concrete judge evaluator for testing."""

    def _load_dataset(self):
        from datasets import Dataset

        return Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "Test"}]],
                "ground_truth": [{"answer": "A"}],
            }
        )

    async def _evaluate_example(self, prompt, ground_truth, metadata=None):
        return 1.0

    def _build_result(self, scores, avg_score):
        return {"scores": scores, "avg": avg_score}


class TestBaseJudgeEvaluator:
    """Tests for BaseJudgeEvaluator."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteJudgeEvaluator()
            assert evaluator.judge_config.model == "openai/gpt-oss-120b"

    def test_init_with_custom_judge_config(self):
        """Test initialization with custom judge config."""
        judge_config = JudgeConfig(model="judge-model")
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteJudgeEvaluator(judge_config=judge_config)
            assert evaluator.judge_config.model == "judge-model"

    def test_judge_client_property_without_api_key_raises(self):
        """Test that judge_client property raises without API key."""
        with patch.dict("os.environ", {}, clear=True):
            evaluator = ConcreteJudgeEvaluator()
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                _ = evaluator.judge_client

    def test_judge_client_property_with_api_key(self):
        """Test that judge_client property works with API key."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteJudgeEvaluator()
            client = evaluator.judge_client
            assert client is not None

    def test_judge_client_lazy_initialization(self):
        """Test that judge_client is lazily initialized."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteJudgeEvaluator()
            assert evaluator._judge_client is None
            _ = evaluator.judge_client
            assert evaluator._judge_client is not None

    def test_judge_client_property_with_custom_env_var_raises(self):
        """Test that judge_client raises with custom env var when not set."""
        judge_config = JudgeConfig(api_key_env="CUSTOM_JUDGE_KEY")
        with patch.dict("os.environ", {}, clear=True):
            evaluator = ConcreteJudgeEvaluator(judge_config=judge_config)
            with pytest.raises(ValueError, match="CUSTOM_JUDGE_KEY"):
                _ = evaluator.judge_client


class TestEvaluateEdgeCases:
    """Tests for evaluate() method edge cases."""

    async def test_evaluate_skips_empty_prompt(self, capsys):
        """Test that evaluate skips examples with empty prompt (line 167, 174)."""

        class EmptyPromptEvaluator(BaseVerlEvaluator):
            def _load_dataset(self):
                return Dataset.from_dict(
                    {
                        "prompt": [[], [{"role": "user", "content": "Valid"}]],
                        "ground_truth": [{"answer": "A"}, {"answer": "B"}],
                    }
                )

            async def _evaluate_example(self, prompt, ground_truth, metadata=None):
                return 1.0

            def _build_result(self, scores, avg_score):
                return {"scores": scores, "avg": avg_score}

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = EmptyPromptEvaluator()
            result = await evaluator.evaluate(num_examples=2)
            # Should only process 1 example (the one with non-empty prompt)
            assert len(result["scores"]) == 1
            assert result["avg"] == 1.0

    async def test_evaluate_skips_none_ground_truth(self, capsys):
        """Test that evaluate skips examples with None ground_truth (line 174)."""

        class NoneGroundTruthEvaluator(BaseVerlEvaluator):
            def _load_dataset(self):
                # Use a custom iterable that returns None for ground_truth
                class MockDataset:
                    def __iter__(self):
                        yield {
                            "prompt": [{"role": "user", "content": "Q1"}],
                            "ground_truth": None,
                        }
                        yield {
                            "prompt": [{"role": "user", "content": "Q2"}],
                            "ground_truth": {"answer": "B"},
                        }

                return MockDataset()

            async def _evaluate_example(self, prompt, ground_truth, metadata=None):
                return 1.0

            def _build_result(self, scores, avg_score):
                return {"scores": scores, "avg": avg_score}

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = NoneGroundTruthEvaluator()
            result = await evaluator.evaluate(num_examples=2)
            # Should only process 1 example (the one with non-None ground_truth)
            assert len(result["scores"]) == 1
            assert result["avg"] == 1.0

    async def test_evaluate_skips_both_empty(self, capsys):
        """Test that evaluate skips examples with both empty prompt and ground_truth."""

        class BothEmptyEvaluator(BaseVerlEvaluator):
            def _load_dataset(self):
                class MockDataset:
                    def __iter__(self):
                        yield {
                            "prompt": [],
                            "ground_truth": {"answer": "A"},
                        }  # empty prompt
                        yield {
                            "prompt": [{"role": "user", "content": "Valid"}],
                            "ground_truth": {"answer": "B"},
                        }  # valid
                        yield {
                            "prompt": [{"role": "user", "content": "Q3"}],
                            "ground_truth": None,
                        }  # None ground_truth

                return MockDataset()

            async def _evaluate_example(self, prompt, ground_truth, metadata=None):
                return 1.0

            def _build_result(self, scores, avg_score):
                return {"scores": scores, "avg": avg_score}

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = BothEmptyEvaluator()
            result = await evaluator.evaluate(num_examples=3)
            # Should only process 1 example (the middle valid one)
            assert len(result["scores"]) == 1
            assert result["avg"] == 1.0

    async def test_evaluate_progress_reporting(self, capsys):
        """Test progress reporting at intervals (lines 181-182)."""

        class ProgressEvaluator(BaseVerlEvaluator):
            def _load_dataset(self):
                # Create 15 examples to trigger progress at 10
                prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(15)]
                ground_truths = [{"answer": str(i)} for i in range(15)]
                return Dataset.from_dict(
                    {
                        "prompt": prompts,
                        "ground_truth": ground_truths,
                    }
                )

            async def _evaluate_example(self, prompt, ground_truth, metadata=None):
                return 0.5

            def _build_result(self, scores, avg_score):
                return {"scores": scores, "avg": avg_score}

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ProgressEvaluator()
            result = await evaluator.evaluate(num_examples=15, progress_interval=10)
            # Check that progress was reported
            captured = capsys.readouterr()
            assert "Processed 10 examples" in captured.out
            assert "avg score: 0.500" in captured.out
            assert len(result["scores"]) == 15

    async def test_evaluate_no_progress_for_small_dataset(self, capsys):
        """Test that progress is not reported when dataset is smaller than interval."""

        class SmallEvaluator(BaseVerlEvaluator):
            def _load_dataset(self):
                return Dataset.from_dict(
                    {
                        "prompt": [[{"role": "user", "content": "Q1"}]],
                        "ground_truth": [{"answer": "A"}],
                    }
                )

            async def _evaluate_example(self, prompt, ground_truth, metadata=None):
                return 1.0

            def _build_result(self, scores, avg_score):
                return {"scores": scores, "avg": avg_score}

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = SmallEvaluator()
            result = await evaluator.evaluate(num_examples=1, progress_interval=10)
            captured = capsys.readouterr()
            # Should not print progress since we only processed 1 example
            assert "Processed" not in captured.out
            assert len(result["scores"]) == 1

    async def test_evaluate_all_skipped_returns_zero_avg(self):
        """Test that evaluate returns avg_score of 0 when all examples are skipped."""

        class AllSkippedEvaluator(BaseVerlEvaluator):
            def _load_dataset(self):
                class MockDataset:
                    def __iter__(self):
                        yield {
                            "prompt": [],
                            "ground_truth": {"answer": "A"},
                        }  # empty prompt
                        yield {
                            "prompt": [{"role": "user", "content": "Q2"}],
                            "ground_truth": None,
                        }  # None ground_truth

                return MockDataset()

            async def _evaluate_example(self, prompt, ground_truth, metadata=None):
                return 1.0

            def _build_result(self, scores, avg_score):
                return {"scores": scores, "avg": avg_score}

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = AllSkippedEvaluator()
            result = await evaluator.evaluate(num_examples=2)
            assert result["scores"] == []
            assert result["avg"] == 0


class TestBaseMCQEvaluatorMessages:
    """Tests for BaseMCQEvaluator._build_messages() method."""

    def test_build_messages_preserves_original_prompt(self):
        """Test that _build_messages doesn't modify original prompt list."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator(system_prompt="System")
            original_prompt = [{"role": "user", "content": "Question"}]
            messages = evaluator._build_messages(original_prompt)
            # Original should not be modified
            assert len(original_prompt) == 1
            assert original_prompt[0]["role"] == "user"
            # New messages should have system prompt
            assert len(messages) == 2
            assert messages[0]["role"] == "system"

    def test_build_messages_with_multiple_user_messages(self):
        """Test building messages with multiple user messages and system prompt."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator(system_prompt="You are helpful")
            prompt = [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second"},
            ]
            messages = evaluator._build_messages(prompt)
            assert len(messages) == 4
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
            assert messages[3]["role"] == "user"

    def test_build_messages_empty_system_prompt_treated_as_none(self):
        """Test that empty string system prompt is treated as falsy."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            evaluator = ConcreteMCQEvaluator(system_prompt="")
            prompt = [{"role": "user", "content": "Question"}]
            messages = evaluator._build_messages(prompt)
            # Empty string should be falsy, so no system prompt added
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
