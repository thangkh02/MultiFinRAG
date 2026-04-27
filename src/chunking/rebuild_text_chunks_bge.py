from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model


PDF_DIR = Path("data/pdfs")
RAW_DIR = Path("data/raw_filings")
CHUNK_DIR = Path("data/chunks")


def clean_text(value: str) -> str:
    value = str(value or "").replace("\xa0", " ")
    value = value.replace("−", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def stable_id(prefix: str, *parts: str) -> str:
    raw = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def infer_ticker(path: Path) -> str:
    return path.name.split("_", 1)[0]


def matching_html_path(pdf_path: Path) -> Path | None:
    for suffix in (".htm", ".html"):
        matches = list(RAW_DIR.rglob(f"{pdf_path.stem}{suffix}"))
        if matches:
            return matches[0]
    return None


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    protected = {
        "U.S.": "U<dot>S<dot>",
        "Inc.": "Inc<dot>",
        "Co.": "Co<dot>",
        "Mr.": "Mr<dot>",
        "Ms.": "Ms<dot>",
        "No.": "No<dot>",
    }
    for key, value in protected.items():
        text = text.replace(key, value)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$])", text)
    sentences = []
    for part in parts:
        for key, value in protected.items():
            part = part.replace(value, key)
        part = clean_text(part)
        if len(part) >= 25 and re.search(r"[A-Za-z]", part):
            sentences.append(part)
    return sentences


def extract_sentence_rows(pdf_path: Path) -> list[dict]:
    rows = []
    with fitz.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            for sentence in split_sentences(text):
                rows.append({"page": page_no, "sentence": sentence})
    return rows


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def mean_chunk_vector(sentence_vectors: np.ndarray, start: int, end: int) -> np.ndarray:
    vector = sentence_vectors[start:end].mean(axis=0, keepdims=True)
    return l2_normalize(vector)[0]


def make_chunk(sentence_rows: list[dict], start: int, end: int, sentence_vectors: np.ndarray) -> dict:
    rows = sentence_rows[start:end]
    return {
        "sentence_start": start,
        "sentence_end": end - 1,
        "page_start": rows[0]["page"],
        "page_end": rows[-1]["page"],
        "num_sentences": len(rows),
        "text": clean_text(" ".join(row["sentence"] for row in rows)),
        "_vector": mean_chunk_vector(sentence_vectors, start, end),
    }


def semantic_chunks_from_bge(
    sentence_rows: list[dict],
    sentence_vectors: np.ndarray,
    percentile: float,
    window_size: int,
    overlap: int,
    min_sentences: int,
    max_sentences: int,
    merge_threshold: float,
    max_merge_sentences: int,
    max_merge_words: int,
    max_merge_pages: int,
) -> list[dict]:
    if not sentence_rows:
        return []

    if len(sentence_rows) == 1:
        return [make_chunk(sentence_rows, 0, 1, sentence_vectors)]

    breakpoints = set()
    step = max(1, window_size - overlap)
    for window_start in range(0, len(sentence_rows), step):
        window_end = min(len(sentence_rows), window_start + window_size)
        if window_end - window_start < 2:
            continue

        left = sentence_vectors[window_start : window_end - 1]
        right = sentence_vectors[window_start + 1 : window_end]
        sims = np.sum(left * right, axis=1)
        distances = 1 - sims
        threshold = np.percentile(distances, percentile)
        for local_idx, dist in enumerate(distances):
            if dist >= threshold:
                breakpoints.add(window_start + local_idx)

        if window_end == len(sentence_rows):
            break

    chunks = []
    start = 0
    for idx in range(len(sentence_rows) - 1):
        current_len = idx - start + 1
        should_break = idx in breakpoints and current_len >= min_sentences
        too_long = current_len >= max_sentences
        if should_break or too_long:
            chunks.append(make_chunk(sentence_rows, start, idx + 1, sentence_vectors))
            start = idx + 1

    if start < len(sentence_rows):
        chunks.append(make_chunk(sentence_rows, start, len(sentence_rows), sentence_vectors))

    if len(chunks) >= 2 and chunks[-1]["num_sentences"] < min_sentences:
        tail = chunks.pop()
        chunks[-1]["text"] = clean_text(chunks[-1]["text"] + " " + tail["text"])
        chunks[-1]["page_end"] = tail["page_end"]
        chunks[-1]["sentence_end"] = tail["sentence_end"]
        chunks[-1]["num_sentences"] += tail["num_sentences"]
        chunks[-1]["_vector"] = l2_normalize((chunks[-1]["_vector"] + tail["_vector"]).reshape(1, -1))[0]

    merged = []
    current = chunks[0]
    for nxt in chunks[1:]:
        sim = float(np.dot(current["_vector"], nxt["_vector"]))
        merged_text = clean_text(current["text"] + " " + nxt["text"])
        merged_words = len(re.findall(r"\S+", merged_text))
        merged_sentences = current["num_sentences"] + nxt["num_sentences"]
        merged_pages = nxt["page_end"] - current["page_start"] + 1
        can_merge = (
            merged_sentences <= max_merge_sentences
            and merged_words <= max_merge_words
            and merged_pages <= max_merge_pages
        )
        if sim >= merge_threshold and can_merge:
            current["text"] = merged_text
            current["page_end"] = nxt["page_end"]
            current["sentence_end"] = nxt["sentence_end"]
            current["num_sentences"] = merged_sentences
            current["_vector"] = l2_normalize((current["_vector"] + nxt["_vector"]).reshape(1, -1))[0]
        else:
            merged.append(current)
            current = nxt
    merged.append(current)

    for chunk in merged:
        chunk.pop("_vector", None)
    return merged


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    global PDF_DIR, RAW_DIR, CHUNK_DIR

    parser = argparse.ArgumentParser(description="Rebuild text chunks using BGE sentence semantic chunking.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
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
    args = parser.parse_args()
    if args.overlap >= args.window_size:
        raise ValueError("--overlap must be smaller than --window-size")

    PDF_DIR = args.pdf_dir
    RAW_DIR = args.raw_dir
    CHUNK_DIR = args.chunk_dir

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    embedder = load_bge_model(model_name=args.model, batch_size=args.batch_size, device=args.device)

    all_text_chunks = []
    for doc_idx, pdf_path in enumerate(pdf_paths, start=1):
        sentence_rows = extract_sentence_rows(pdf_path)
        if not sentence_rows:
            print(f"[{doc_idx}/{len(pdf_paths)}] {pdf_path.name}: 0 sentences")
            continue

        sentences = [row["sentence"] for row in sentence_rows]
        sentence_vectors = embedder.encode_documents(sentences)
        chunks = semantic_chunks_from_bge(
            sentence_rows=sentence_rows,
            sentence_vectors=sentence_vectors,
            percentile=args.percentile,
            window_size=args.window_size,
            overlap=args.overlap,
            min_sentences=args.min_sentences,
            max_sentences=args.max_sentences,
            merge_threshold=args.merge_threshold,
            max_merge_sentences=args.max_merge_sentences,
            max_merge_words=args.max_merge_words,
            max_merge_pages=args.max_merge_pages,
        )

        html_path = matching_html_path(pdf_path)
        for ordinal, chunk in enumerate(chunks, start=1):
            chunk_id = stable_id("text", pdf_path.name, ordinal, chunk["page_start"], chunk["text"][:120])
            all_text_chunks.append(
                {
                    "id": chunk_id,
                    "modality": "text",
                    "source_pdf": str(pdf_path).replace("\\", "/"),
                    "source_html": str(html_path or "").replace("\\", "/"),
                    "ticker": infer_ticker(pdf_path),
                    "page": chunk["page_start"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "sentence_start": chunk["sentence_start"],
                    "sentence_end": chunk["sentence_end"],
                    "num_sentences": chunk["num_sentences"],
                    "text": chunk["text"],
                    "chunking_method": "bge_sentence_semantic",
                }
            )
        print(f"[{doc_idx}/{len(pdf_paths)}] {pdf_path.name}: {len(sentence_rows)} sentences -> {len(chunks)} chunks")

    table_chunks = load_jsonl(CHUNK_DIR / "table_chunks.jsonl")
    image_chunks = load_jsonl(CHUNK_DIR / "image_chunks.jsonl")
    all_chunks = all_text_chunks + table_chunks + image_chunks

    write_jsonl(CHUNK_DIR / "text_chunks.jsonl", all_text_chunks)
    write_jsonl(CHUNK_DIR / "all_chunks.jsonl", all_chunks)

    summary = {
        "pdf_count": len(pdf_paths),
        "chunk_counts": dict(Counter(chunk["modality"] for chunk in all_chunks)),
        "total_chunks": len(all_chunks),
        "text_chunking": {
            "method": "BGE sentence semantic chunking",
            "model": args.model,
            "breakpoint_percentile": args.percentile,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "min_sentences": args.min_sentences,
            "max_sentences": args.max_sentences,
            "merge_similarity_threshold": args.merge_threshold,
            "max_merge_sentences": args.max_merge_sentences,
            "max_merge_words": args.max_merge_words,
            "max_merge_pages": args.max_merge_pages,
        },
        "table_chunking": "unchanged from existing table_chunks.jsonl",
        "image_chunking": "unchanged from existing image_chunks.jsonl",
    }
    (CHUNK_DIR / "chunk_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
