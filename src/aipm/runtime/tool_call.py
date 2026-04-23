import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


class ToolCallParseError(Exception):
    """Raised when tool call parsing fails."""
    pass


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: Dict[str, Any]


def extract_json_blocks(text: str) -> list[str]:
    """
    Extract candidate JSON objects from text.

    This is a heuristic:
    - finds {...} blocks
    - does NOT guarantee validity
    """
    pattern = r"\{.*?\}"
    return re.findall(pattern, text, re.DOTALL)


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

    Returns:
        ToolCall or None (if no valid call found)

    Never raises unless something fundamentally breaks.
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
