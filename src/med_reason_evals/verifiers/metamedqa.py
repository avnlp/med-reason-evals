"""MetaMedQA Evaluation.

Dataset: HuggingFace `maximegmd/MetaMedQA` dataset.

- Parser: Extracts first letter A-Z from completions
- Reward Functions:
    - Correct answer reward
    - Format reward
"""

import os
from typing import Any

import verifiers as vf
from dotenv import load_dotenv
from openai import OpenAI

from med_reason_evals.data.metamedqa import MetaMedQADataset
from med_reason_evals.verifiers.answer_correctness_reward import (
    correct_answer_reward_func,
)


class LetterParser:
    """Parser that extracts the first letter (A-Z) from completions."""

    def __init__(self) -> None:
        """Initialize the LetterParser."""
        pass

    def parse_answer(self, completion: Any) -> str:
        """Parse the completion to extract the first letter A-Z."""
        text = self._get_text_from_completion(completion)
        return self._first_letter(text) or ""

    def get_format_reward_func(self) -> Any:
        """Return a format reward function (simple placeholder)."""

        def format_reward(
            parser: Any, completion: str, answer: str, **kwargs: Any
        ) -> float:
            # Basic format reward - just check if we were able to extract a letter
            parsed = self.parse_answer(completion)
            return 1.0 if parsed != "" else 0.0

        return format_reward

    def _get_text_from_completion(self, completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion:
            last = completion[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
            return str(last)
        return str(completion)

    def _first_letter(self, text: str) -> str:
        t = (text or "").upper()
        for ch in t:
            if "A" <= ch <= "Z":
                return ch
        return ""


def main() -> None:
    """Run the evaluation on the MetaMedQA dataset."""
    # Load environment variables
    load_dotenv()

    # Create an instance of the processor
    dataset = MetaMedQADataset(split="test", num_examples=-1)

    # Construct prompts
    system_prompt = (
        "Think step-by-step inside think> tags, then give only the letter "
        "of the correct answer. Do not include option text; only the letter."
    )

    parser = LetterParser()

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
        parser=parser,
    )

    env = vf.SingleTurnEnv(
        dataset=dataset.dataset,
        eval_dataset=dataset.dataset,  # Using same dataset for both train and eval as in original
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    # Run the evaluation
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/v1",
    )
    results = env.evaluate(
        client=client,
        model="llama-3.3-70b-versatile",
        num_examples=2,
        rollouts_per_example=5,
    )
    print(results)


if __name__ == "__main__":
    main()
