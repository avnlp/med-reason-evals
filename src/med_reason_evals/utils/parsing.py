"""Parse model responses into structured values used by evaluators.

The parsers favor robustness over strict validation because judge models often
emit loosely formatted answers. The helpers strip reasoning tags, recover JSON
payloads from fenced or partial outputs, and normalize yes/no signals into a
binary outcome when possible.
"""

import json
import re
from typing import Any


_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_PAIR_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str | None) -> str:
    """Extract the answer section from text, removing think tags.

    Behavior is intentionally conservative:
    - If there is exactly one well-formed <think>...</think> pair AND no unclosed
      <think> later, return everything after that closing tag.
    - If there's only a closing </think> tag (some models emit this), return
      everything after it.
    - Otherwise, return the full response.

    Args:
        text: The model's response text.

    Returns:
        The text with think tags removed, or the original text if no valid
        think tags are found.
    """
    text = text or ""

    # Some judge models only emit a closing tag; treat it as the reasoning/answer
    # boundary to avoid discarding the final response fragment.
    if _THINK_OPEN_RE.search(text) is None:
        closes = list(_THINK_CLOSE_RE.finditer(text))
        if closes:
            return text[closes[-1].end() :].strip()
        return text.strip()

    # Count properly closed pairs, but stop early once we know there are 2+.
    it = _THINK_PAIR_RE.finditer(text)
    first = next(it, None)
    if first is None:
        return text.strip()
    if next(it, None) is not None:
        # Multiple pairs imply interleaved reasoning; preserve the full text.
        return text.strip()

    # Exactly one pair means the answer follows the closing tag.
    return text[first.end() :].strip()


def parse_json_response(text: str) -> dict[str, Any] | list[Any]:
    """Extract and parse JSON from a judge response.

    Args:
        text: The judge's response text, optionally wrapped in Markdown fences.

    Returns:
        The parsed JSON payload, or an empty dict if parsing fails.
    """
    json_match = re.search(
        r"
    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first shallow JSON object to salvage partial outputs.
        # The shallow pattern avoids matching nested structures that might be
        # truncated mid-stream, which is common in incomplete model responses.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}


def parse_yes_no(text: str) -> bool | None:
    """Parse a yes/no response from the judge.

    Handles various formats:
    - Plain "yes" or "no" at the start of text (case insensitive)
    - JSON with "criteria_met" or "correct" boolean fields

    Args:
        text: The judge's response text.

    Returns:
        True if yes/correct, False if no/incorrect, None if unparsable.
    """
    if not text:
        return None

    text_stripped = text.strip()

    # Prefer an explicit leading yes/no before attempting JSON parsing.
    yes_no_match = re.match(r"^\s*(yes|no)\b", text_stripped, re.IGNORECASE)
    if yes_no_match:
        return yes_no_match.group(1).lower() == "yes"

    # Accept small JSON snippets used by some judge prompts.
    json_match = re.search(r"\{[^{}]*\}", text_stripped)
    if json_match:
        try:
            data = json.loads(json_match.group())
            # Honor common judge schema fields in priority order.
            for field in ("criteria_met", "correct"):
                if field in data:
                    value = data[field]
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.lower() in ("yes", "true")
        except json.JSONDecodeError:
            pass

    return None
