from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import fitz
from tqdm import tqdm

from .io_utils import load_jsonl, write_jsonl
from .load_financebench import required_doc_names
from .schema import chunk_text
from .text_utils import doc_key

LOGGER = logging.getLogger(__name__)


def stable_id(prefix: str, *parts: object) -> str:
    raw = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_existing_chunk(row: dict[str, Any]) -> dict[str, Any]:
    chunk = dict(row)
    chunk.setdefault("chunk_id", chunk.get("id") or stable_id("chunk", chunk_text(chunk)[:200]))
    chunk.setdefault("id", chunk["chunk_id"])
    chunk.setdefault("doc_name", doc_key(chunk.get("doc_name") or chunk.get("source_pdf") or chunk.get("source_html")))
    chunk.setdefault("chunk_type", chunk.get("chunk_type") or chunk.get("modality") or "text")
    chunk.setdefault("text", chunk_text(chunk))
    if "page_start" not in chunk and "page" in chunk:
        chunk["page_start"] = chunk["page"]
    if "page_end" not in chunk and "page" in chunk:
        chunk["page_end"] = chunk["page"]
    chunk.setdefault("metadata", {})
    return chunk


def load_existing_chunks(source_path: Path, chunk_types: set[str] | None = None) -> list[dict[str, Any]]:
    rows = [normalize_existing_chunk(row) for row in load_jsonl(source_path)]
    if chunk_types:
        rows = [row for row in rows if str(row.get("chunk_type") or row.get("modality")) in chunk_types]
    LOGGER.info("Loaded %d existing chunks from %s", len(rows), source_path)
    return rows


def pdf_path_for_doc(pdf_dir: Path, doc_name: str) -> Path | None:
    exact = pdf_dir / f"{doc_name}.pdf"
    if exact.exists():
        return exact
    matches = sorted(pdf_dir.glob(f"{doc_name}*.pdf"))
    return matches[0] if matches else None


def build_page_chunks(samples: list[dict[str, Any]], pdf_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    doc_names = sorted(required_doc_names(samples))
    missing = []
    for doc_name in tqdm(doc_names, desc="Chunk PDFs by page"):
        pdf_path = pdf_path_for_doc(pdf_dir, doc_name)
        if not pdf_path:
            missing.append(doc_name)
            continue
        with fitz.open(pdf_path) as doc:
            for page_idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if not text.strip():
                    continue
                chunk_id = stable_id("fb_page", doc_name, page_idx, text[:120])
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "id": chunk_id,
                        "doc_name": doc_name.lower(),
                        "source_pdf": str(pdf_path).replace("\\", "/"),
                        "text": text,
                        "page_start": page_idx,
                        "page_end": page_idx,
                        "section_title": None,
                        "chunk_type": "text",
                        "metadata": {"adapter": "page_level_pymupdf"},
                    }
                )
    if missing:
        LOGGER.warning("Missing %d PDFs referenced by FinanceBench. First few: %s", len(missing), missing[:10])
    LOGGER.info("Built %d page chunks from %s", len(chunks), pdf_dir)
    return chunks


def prepare_chunks(config: dict[str, Any], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = config.get("source_chunks_path")
    output_path = Path(config["output_chunks_path"])
    chunk_types = set(config.get("chunk_types") or []) or None
    if source:
        chunks = load_existing_chunks(Path(source), chunk_types=chunk_types)
    else:
        chunks = build_page_chunks(samples, Path(config["pdf_dir"]))
    write_jsonl(output_path, chunks)
    LOGGER.info("Wrote chunks to %s", output_path)
    return chunks
