"""Shared lightweight text helpers for search, coverage, and extraction."""

from __future__ import annotations

import re


STOP_WORDS = {
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "whose",
    # Verbs / Auxiliaries
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must",
    # Prepositions / Conjunctions
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "yet", "also", "now",
    # Query noise
    "find", "get", "search", "show", "tell", "explain", "describe", "list",
    "latest", "new", "recent", "current", "update", "updates", "information",
    "detail", "details",
}


def stem_word(word: str) -> str:
    word = word.lower().strip()
    
    def is_valid_stem(stem: str) -> bool:
        return len(stem) >= 3 and bool(re.search(r'[aeiouy]', stem))

    if word.endswith("ing"):
        stem = word[:-3]
        if is_valid_stem(stem):
            return stem
    elif word.endswith("ed"):
        stem = word[:-2]
        if is_valid_stem(stem):
            return stem
    elif word.endswith("es"):
        stem_es = word[:-2]
        if stem_es.endswith(('s', 'x', 'z', 'ch', 'sh')):
            if is_valid_stem(stem_es):
                return stem_es
        else:
            stem_s = word[:-1]
            if is_valid_stem(stem_s):
                return stem_s
    elif word.endswith("s") and not word.endswith("ss"):
        stem = word[:-1]
        if is_valid_stem(stem):
            return stem
    elif word.endswith("ly"):
        stem = word[:-2]
        if is_valid_stem(stem):
            return stem
    elif word.endswith("ment"):
        stem = word[:-4]
        if is_valid_stem(stem):
            return stem
            
    return word


def tokenize_terms(text: str, *, stop_words: set[str] | None = None) -> set[str]:
    """Return lowercase search-relevant terms from free text."""
    ignored = STOP_WORDS if stop_words is None else stop_words
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9-]{1,}", text.lower())
        if term not in ignored
    }
