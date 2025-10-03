"""Reward Functions for the Medbullets dataset.

- Parser extracts \\boxed{A|B|C|D|E} from completions
- Reward looks for exact match between parsed letter and answer letter
"""

from typing import Any


def correct_answer_reward_func(
    parser: Any, completion: str, answer: str, **kwargs
) -> float:
    """Reward function for correct answer.

    Args:
        parser: Parser object
        completion: Completion string
        answer: Answer string
        **kwargs: Additional keyword arguments
    Returns:
        float: Reward value
    """
    response = parser.parse_answer(completion) or ""

    return 1.0 if response == answer else 0.0
