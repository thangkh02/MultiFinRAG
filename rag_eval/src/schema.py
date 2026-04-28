from __future__ import annotations

from typing import Any


def chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or chunk.get("summary") or "")


def chunk_doc_name(chunk: dict[str, Any]) -> str:
    from .text_utils import doc_key

    for key in ("doc_name", "source_doc", "source_pdf", "source_html"):
        value = chunk.get(key)
        if value:
            return doc_key(str(value))
    return ""


def page_range(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    start = chunk.get("page_start", chunk.get("page"))
    end = chunk.get("page_end", chunk.get("page"))
    try:
        start_i = int(start) if start is not None else None
        end_i = int(end) if end is not None else start_i
    except (TypeError, ValueError):
        return None, None
    return start_i, end_i


def page_overlaps(chunk: dict[str, Any], one_indexed_page: int | None) -> bool:
    if one_indexed_page is None:
        return False
    start, end = page_range(chunk)
    if start is None or end is None:
        return False
    return start <= one_indexed_page <= end
