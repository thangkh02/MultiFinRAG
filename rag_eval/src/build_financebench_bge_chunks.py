from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(PROJECT_SRC))

from chunking.rebuild_text_chunks_bge import (  # noqa: E402
    clean_text,
    extract_sentence_rows,
    infer_ticker,
    load_bge_model,
    semantic_chunks_from_bge,
    stable_id,
)
from common.bge_embedder import DEFAULT_BGE_MODEL  # noqa: E402

from .io_utils import load_jsonl, write_jsonl
from .load_financebench import required_doc_names

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def pdf_path_for_doc(pdf_dir: Path, doc_name: str) -> Path | None:
    exact = pdf_dir / f"{doc_name}.pdf"
    if exact.exists():
        return exact
    matches = sorted(pdf_dir.glob(f"{doc_name}*.pdf"))
    return matches[0] if matches else None


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            import json

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_bge_text_chunks(
    samples: list[dict[str, Any]],
    pdf_dir: Path,
    model_name: str,
    batch_size: int,
    device: str | None,
    percentile: float,
    window_size: int,
    overlap: int,
    min_sentences: int,
    max_sentences: int,
    merge_threshold: float,
    max_merge_sentences: int,
    max_merge_words: int,
    max_merge_pages: int,
    limit_docs: int | None,
    output_path: Path | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    doc_names = sorted(required_doc_names(samples))
    if limit_docs is not None:
        doc_names = doc_names[:limit_docs]

    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    all_chunks: list[dict[str, Any]] = []
    completed_docs: set[str] = set()
    if output_path and resume and output_path.exists():
        existing_chunks = load_jsonl(output_path)
        all_chunks.extend(existing_chunks)
        completed_docs = {
            str((chunk.get("metadata") or {}).get("source_doc_name") or chunk.get("doc_name") or "").lower()
            for chunk in existing_chunks
        }
        completed_docs.discard("")
        LOGGER.info("Resume enabled: loaded %d existing chunks for %d docs", len(existing_chunks), len(completed_docs))
    elif output_path and output_path.exists() and not resume:
        output_path.unlink()

    missing_docs: list[str] = []

    for doc_name in tqdm(doc_names, desc="BGE semantic chunk FinanceBench PDFs"):
        if doc_name.lower() in completed_docs:
            continue
        pdf_path = pdf_path_for_doc(pdf_dir, doc_name)
        if not pdf_path:
            missing_docs.append(doc_name)
            continue

        sentence_rows = extract_sentence_rows(pdf_path)
        if not sentence_rows:
            LOGGER.warning("No sentences extracted: %s", pdf_path)
            continue

        sentence_vectors = embedder.encode_documents([row["sentence"] for row in sentence_rows])
        chunks = semantic_chunks_from_bge(
            sentence_rows=sentence_rows,
            sentence_vectors=sentence_vectors,
            percentile=percentile,
            window_size=window_size,
            overlap=overlap,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            merge_threshold=merge_threshold,
            max_merge_sentences=max_merge_sentences,
            max_merge_words=max_merge_words,
            max_merge_pages=max_merge_pages,
        )

        doc_chunks: list[dict[str, Any]] = []
        for ordinal, chunk in enumerate(chunks, start=1):
            chunk_id = stable_id("fb_bge_text", pdf_path.name, ordinal, chunk["page_start"], chunk["text"][:120])
            doc_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "id": chunk_id,
                    "doc_name": doc_name.lower(),
                    "source_pdf": str(pdf_path).replace("\\", "/"),
                    "ticker": infer_ticker(pdf_path),
                    "text": clean_text(chunk["text"]),
                    "page": chunk["page_start"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "sentence_start": chunk["sentence_start"],
                    "sentence_end": chunk["sentence_end"],
                    "num_sentences": chunk["num_sentences"],
                    "section_title": None,
                    "chunk_type": "text",
                    "modality": "text",
                    "chunking_method": "bge_sentence_semantic",
                    "metadata": {
                        "adapter": "financebench_bge_semantic",
                        "model": model_name,
                        "source_doc_name": doc_name,
                    },
                }
            )
        all_chunks.extend(doc_chunks)
        if output_path:
            append_jsonl(output_path, doc_chunks)
        LOGGER.info("%s: %d sentences -> %d chunks", pdf_path.name, len(sentence_rows), len(chunks))

    if missing_docs:
        LOGGER.warning("Missing %d referenced PDFs. First few: %s", len(missing_docs), missing_docs[:10])
    LOGGER.info("Built %d BGE semantic text chunks", len(all_chunks))
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FinanceBench text chunks using existing BGE semantic chunking.")
    parser.add_argument("--financebench", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--percentile", type=float, default=95)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--min-sentences", type=int, default=3)
    parser.add_argument("--max-sentences", type=int, default=16)
    parser.add_argument("--merge-threshold", type=float, default=0.85)
    parser.add_argument("--max-merge-sentences", type=int, default=32)
    parser.add_argument("--max-merge-words", type=int, default=1000)
    parser.add_argument("--max-merge-pages", type=int, default=6)
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.overlap >= args.window_size:
        raise ValueError("--overlap must be smaller than --window-size")

    setup_logging(args.log_level)
    samples = load_jsonl(args.financebench)
    chunks = build_bge_text_chunks(
        samples=samples,
        pdf_dir=args.pdf_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        percentile=args.percentile,
        window_size=args.window_size,
        overlap=args.overlap,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        merge_threshold=args.merge_threshold,
        max_merge_sentences=args.max_merge_sentences,
        max_merge_words=args.max_merge_words,
        max_merge_pages=args.max_merge_pages,
        limit_docs=args.limit_docs,
        output_path=args.output,
        resume=not args.no_resume,
    )
    if args.no_resume:
        write_jsonl(args.output, chunks)
    LOGGER.info("Wrote %d chunks to %s", len(chunks), args.output)


if __name__ == "__main__":
    main()
