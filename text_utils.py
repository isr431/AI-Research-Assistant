"""Shared lightweight text helpers for search, coverage, and extraction."""

from __future__ import annotations

import re


STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "did",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "latest",
    "more",
    "new",
    "not",
    "now",
    "the",
    "their",
    "this",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def tokenize_terms(text: str, *, stop_words: set[str] | None = None) -> set[str]:
    """Return lowercase search-relevant terms from free text."""
    ignored = STOP_WORDS if stop_words is None else stop_words
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower())
        if term not in ignored
    }
