"""PubMedQA Evaluation.

Dataset: HuggingFace `qiaojin/PubMedQA` (pqa_labeled and pqa_artificial splits).

- Parser: Extracts \\boxed{A|B|C} from completions (A=yes, B=no, C=maybe)
- Reward Functions:
    - Exact match classification reward
"""

import json
import os

import verifiers as vf
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI


def map_row_to_mcq_prompt(row):
    """Map PubMedQA row to MCQ-style prompt with A/B/C answers."""
    question_text = row.get("question", "")
    context_dict = row.get("context", {})
    labels = context_dict.get("labels", [])
    contexts = context_dict.get("contexts", [])
    final_decision = row.get("final_decision", "").lower()

    choices_map = {"yes": "A", "no": "B", "maybe": "C"}
    correct_answer_letter = choices_map.get(final_decision, "C")  # default to maybe

    formatted_contexts = [
        f"{label}. {context}" for label, context in zip(labels, contexts)
    ]
    context_text = "\n".join(formatted_contexts)

    complete_prompt = (
        f"Answer A for yes, B for no or C for maybe.\n\n"
        f"Context: {context_text}\n\n"
        f"Question: {question_text}\nAnswer: "
    )

    return {
        "question": complete_prompt,
        "answer": correct_answer_letter,
        "task": "pubmedqa",
    }


def classification_reward_func(prompt, completion, answer, state, **kwargs) -> float:
    """Exact match reward: 1.0 if predicted letter matches ground truth."""
    # Extract content from chat completion
    if isinstance(completion, list) and len(completion) > 0:
        content = completion[0].get("content", "")
    else:
        content = str(completion)

    # Parse using the rubric's parser
    parser = kwargs.get("parser")
    if parser is None:
        return 0.0

    parsed = parser.parse(content)
    predicted_letter = parsed.strip().rstrip(".") if parsed else None

    return 1.0 if predicted_letter == answer else 0.0


def main() -> None:
    """Run evaluation on PubMedQA."""
    load_dotenv()

    # Load datasets
    DATASET_PATH = "qiaojin/PubMedQA"
    dataset_train = load_dataset(DATASET_PATH, name="pqa_artificial", split="train")
    dataset_test = load_dataset(DATASET_PATH, name="pqa_labeled", split="train")

    # Filter test set to human-annotated 500 examples
    here = os.path.dirname(__file__)
    file_path = os.path.join(here, "data", "test_ground_truth.json")
    with open(file_path) as f:
        test_ids = set(json.load(f))  # use set for O(1) lookup

    dataset_test = dataset_test.filter(
        lambda sample: str(sample["pubid"]) in test_ids, load_from_cache_file=False
    )

    # Map to standard format
    mapped_train = dataset_train.map(
        map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True
    )
    mapped_test = dataset_test.map(
        map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True
    )

    # Use boxed-only system prompt (no chain-of-thought)
    system_prompt = vf.utils.data_utils.BOXED_SYSTEM_PROMPT
    parser = vf.parsers.parser.Parser(extract_fn=vf.extract_boxed_answer)

    # Build rubric
    rubric = vf.Rubric(
        funcs=[classification_reward_func],
        weights=[1.0],
        parser=parser,
    )

    # Create environment
    env = vf.SingleTurnEnv(
        dataset=mapped_train,
        eval_dataset=mapped_test,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    # Initialize client (Groq via OpenAI-compatible API)
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
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
