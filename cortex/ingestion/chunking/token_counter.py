"""Token counting utilities for chunk sizing."""

from __future__ import annotations

import functools

import tiktoken

_ENCODING = "cl100k_base"


@functools.lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = _encoder().encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoder().decode(tokens[:max_tokens])
