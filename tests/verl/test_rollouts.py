"""Tests for Verl rollouts module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from med_reason_evals.verl.rollouts import GroqRollouts, get_default_rollouts


class TestGroqRollouts:
    """Tests for GroqRollouts class."""

    def test_init_without_api_key_raises(self):
        """Init without API key should raise ValueError."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="GROQ_API_KEY"),
        ):
            GroqRollouts()

    def test_init_with_api_key(self):
        """Init with API key should work."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = GroqRollouts()
            assert rollouts.api_key == "test-key"
            assert rollouts.model == GroqRollouts.DEFAULT_MODEL

    def test_init_with_custom_model(self):
        """Custom model should be stored."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = GroqRollouts(model="custom-model")
            assert rollouts.model == "custom-model"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_calls_client(self):
        """Generate should call the OpenAI client."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = GroqRollouts()

            # Mock the client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"

            rollouts._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            result = await rollouts.generate([{"role": "user", "content": "Hello"}])

            assert result == "Test response"
            rollouts._client.chat.completions.create.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """Generate batch should process multiple messages."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = GroqRollouts()

            # Mock the client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"

            rollouts._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            messages_batch = [
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "World"}],
            ]

            results = await rollouts.generate_batch(messages_batch)

            assert len(results) == 2
            assert all(r == "Response" for r in results)

    @pytest.mark.asyncio
    async def test_generate_with_overrides(self):
        """Generate should use override parameters."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = GroqRollouts()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"

            with patch(
                "med_reason_evals.verl.rollouts.call_with_retry",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_retry:
                result = await rollouts.generate(
                    [{"role": "user", "content": "Hi"}],
                    model="custom-model",
                    max_tokens=512,
                    temperature=0.5,
                )

                assert result == "Response"
                call_kwargs = mock_retry.call_args
                assert call_kwargs.kwargs["model"] == "custom-model"
                assert call_kwargs.kwargs["max_tokens"] == 512
                assert call_kwargs.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_temperature_zero(self):
        """Temperature=0 should be passed explicitly (not substituted)."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = GroqRollouts(temperature=0.7)

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"

            with patch(
                "med_reason_evals.verl.rollouts.call_with_retry",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_retry:
                await rollouts.generate(
                    [{"role": "user", "content": "Hi"}],
                    temperature=0,
                )

                call_kwargs = mock_retry.call_args
                assert call_kwargs.kwargs["temperature"] == 0

    def test_init_custom_base_url(self):
        """Custom base URL should be stored."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = GroqRollouts(base_url="http://custom:8080/v1")
            assert rollouts.base_url == "http://custom:8080/v1"

    def test_get_default_rollouts(self):
        """get_default_rollouts should return a GroqRollouts instance."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            rollouts = get_default_rollouts()
            assert isinstance(rollouts, GroqRollouts)
            assert rollouts.model == GroqRollouts.DEFAULT_MODEL
