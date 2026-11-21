"""Compare a pretokenized corpus producer's vocabulary with the training tokenizer."""

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    """Hash the IDs and token strings, including added tokens, for a tokenized corpus."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else None
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("The tokenizer must expose a non-empty vocabulary.")
    if any(
        not isinstance(token, str) or type(token_id) is not int or token_id < 0
        for token, token_id in vocab.items()
    ):
        raise ValueError("Vocabulary entries must map token strings to nonnegative integer IDs.")
    payload = json.dumps(dict(vocab), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def validate_vocab_expectation(data_format: str, expected: str | None) -> None:
    if expected is None:
        return
    if data_format != "tokens":
        raise ValueError("expected_tokenizer_vocab_sha256 applies to tokenized inputs.")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected_tokenizer_vocab_sha256 must be a lowercase SHA-256 digest.")


def verify_tokenizer_vocab(tokenizer: Any, expected: str | None) -> None:
    if expected is not None and tokenizer_vocab_sha256(tokenizer) != expected:
        raise ValueError("The selected tokenizer vocabulary differs from the tokenized corpus fingerprint.")
