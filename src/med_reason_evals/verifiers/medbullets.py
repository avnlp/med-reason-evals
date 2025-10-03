"""Medbullets Evaluation.

Dataset: HuggingFace `mkieffer/Medbullets` dataset.

- Parser: Extracts \boxed{A|B|C|D|E} from completions
- Reward Functions:
    - Correct answer reward
    - Format reward
"""

import os

import verifiers as vf
from dotenv import load_dotenv
from openai import OpenAI
from verifiers.utils.data_utils import extract_boxed_answer

from med_reason_evals.data.medbullets import MedBulletsDataset
from med_reason_evals.verifiers.answer_correctness_reward import (
    correct_answer_reward_func,
)


def main() -> None:
    """Run the evaluation on the Medbullets dataset."""
    # Load environment variables
    load_dotenv()

    # Create an instance of the processor
    dataset = MedBulletsDataset(
        num_train_examples=-1, num_eval_examples=-1, num_options=4
    )

    # Construct prompts
    options = "(A, B, C, or D)" if dataset.num_options == 4 else "(A, B, C, D, or E)"

    system_prompt = (
        f"Think step-by-step inside think> tags, then give only the letter "
        f"of the correct answer inside \\boxed{{...}} {options}. Do not include option "
        f"text in the box; only the letter."
    )

    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
        parser=parser,
    )

    env = vf.SingleTurnEnv(
        dataset=dataset.train_ds,
        eval_dataset=dataset.eval_ds,
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
