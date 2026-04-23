import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


class ToolCallParseError(Exception):
    """Raised when tool call parsing fails."""
    pass


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: Dict[str, Any]


def extract_json_blocks(text: str) -> List[str]:
    """
    Extract ALL JSON blocks from text using stack-based parsing.

    Handles:
    - nested JSON
    - multiple JSON objects
    - noisy model output
    """

    stack = []
    start = None
    candidates = []

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start = i
            stack.append(char)

        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None

    return candidates


def try_parse_json(candidate: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse JSON safely.
    """
    try:
        return json.loads(candidate)
    except Exception:
        return None


def is_valid_tool_call(obj: Dict[str, Any]) -> bool:
    """
    Check if dict matches tool call structure.
    """
    if not isinstance(obj, dict):
        return False

    if "tool" not in obj or "arguments" not in obj:
        return False

    if not isinstance(obj["tool"], str):
        return False

    if not isinstance(obj["arguments"], dict):
        return False

    return True


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Extract a tool call from model output.

    Strategy:
    - Extract ALL JSON blocks
    - Parse each
    - Return FIRST valid tool-call structure

    This avoids:
    - picking inner JSON (like arguments only)
    - picking incomplete blocks
    - dependency on size heuristics

    Returns:
        ToolCall or None
    """

    candidates = extract_json_blocks(text)

    for candidate in candidates:
        parsed = try_parse_json(candidate)

        if parsed is None:
            continue

        if not is_valid_tool_call(parsed):
            continue

        return ToolCall(
            tool=parsed["tool"],
            arguments=parsed["arguments"],
        )

    return None
