"""MedQA Evaluation.

Dataset: HuggingFace `GBaker/MedQA-USMLE-4-options` dataset.

- Parser: Extracts \\boxed{A|B|C|D} from completions
- Reward Functions:
    - Correct answer reward
    - Format reward
"""

import os
import verifiers as vf
from dotenv import load_dotenv
from openai import OpenAI
from verifiers.utils.data_utils import extract_boxed_answer

from med_reason_evals.data.medqa import MedQADataset
from med_reason_evals.verifiers.exact_match_reward import (
    exact_match_reward_func,
)


def load_environment(
    use_think: bool = True,
    num_train_examples: int = -1,
    num_test_examples: int = -1,
) -> vf.SingleTurnEnv:
    """MedQA-USMLE-4-options multiple-choice evaluation.

    Args:
        use_think: Whether to require step-by-step reasoning (default: True)
        num_train_examples: Number of training examples to use (-1 for all)
        num_test_examples: Number of test examples to use (-1 for all)

    Returns:
        vf.SingleTurnEnv configured with MedQA dataset
    """
    dataset = MedQADataset(
        num_train_examples=num_train_examples,
        num_test_examples=num_test_examples,
    )

    options = "(A, B, C, or D)"  # MedQA has 4 options

    system_prompt = (
        f"Think step-by-step inside <think> tags, then give only the letter "
        f"of the correct answer inside \\boxed{{...}} {options}. Do not include option "
        f"text in the box; only the letter."
    )

    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    rubric = vf.Rubric(
        funcs=[exact_match_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],  
        parser=parser,
    )

    return vf.SingleTurnEnv(
        dataset=dataset.train_ds,
        eval_dataset=dataset.test_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )


def main() -> None:
    """Run the evaluation on the MedQA dataset."""
    # Load environment variables
    load_dotenv()

    # Load environment
    env = load_environment(
        use_think=True,
        num_train_examples=-1,
        num_test_examples=-1,
    )

    # Initialize OpenAI-compatible client (e.g., Groq)
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",  # Fixed URL (removed extra spaces)
    )

    # Run evaluation
    results = env.evaluate(
        client=client,
        model="llama-3.3-70b-versatile",
        num_examples=2,
        rollouts_per_example=5,
    )
    print(results)


if __name__ == "__main__":
    main()