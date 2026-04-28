from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\u00a0", " ")
    text = text.replace("â€”", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"[^a-z0-9$%.\-()]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text))


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$%.\-()]+", normalize_text(text)))


def token_recall(needle: str, haystack: str) -> float:
    needle_tokens = token_set(needle)
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & token_set(haystack)) / len(needle_tokens)


def sequence_ratio(needle: str, haystack: str, max_haystack_chars: int = 12000) -> float:
    needle_norm = compact_text(needle)
    haystack_norm = compact_text(haystack)
    if not needle_norm or not haystack_norm:
        return 0.0
    if len(haystack_norm) > max_haystack_chars:
        start = 0
        for token in sorted(token_set(needle_norm), key=len, reverse=True):
            pos = haystack_norm.find(token)
            if pos >= 0:
                start = max(0, pos - max_haystack_chars // 3)
                break
        haystack_norm = haystack_norm[start : start + max_haystack_chars]
    return SequenceMatcher(None, needle_norm, haystack_norm).ratio()


def doc_key(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value).replace("\\", "/")).stem.lower()
