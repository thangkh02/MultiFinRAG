from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .chunk_adapter import normalize_existing_chunk
from .io_utils import load_jsonl, write_json, write_jsonl
from .schema import chunk_doc_name, chunk_text, page_overlaps
from .text_utils import compact_text, doc_key, sequence_ratio, token_recall

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def evidence_doc_name(sample: dict[str, Any], evidence: dict[str, Any]) -> str:
    return doc_key(evidence.get("evidence_doc_name") or evidence.get("doc_name") or sample.get("doc_name"))


def evidence_page(evidence: dict[str, Any], evidence_page_base: int) -> int | None:
    raw_page = evidence.get("evidence_page_num")
    if raw_page is None:
        return None
    # FinanceBench evidence_page_num is zero-indexed. Most chunkers store PDF pages as one-indexed.
    return int(raw_page) - evidence_page_base + 1


def make_queries(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queries = []
    for sample in samples:
        query_id = str(sample["financebench_id"])
        queries.append(
            {
                "query_id": query_id,
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "doc_name": sample.get("doc_name"),
                "company": sample.get("company"),
                "question_type": sample.get("question_type"),
                "question_reasoning": sample.get("question_reasoning"),
            }
        )
    return queries


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = [normalize_existing_chunk(row) for row in load_jsonl(path)]
    LOGGER.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


def group_chunks_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk_doc_name(chunk)].append(chunk)
    return grouped


def exact_matches(evidence_text: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    needle = compact_text(evidence_text)
    if not needle:
        return []
    matches = []
    for chunk in chunks:
        if needle in compact_text(chunk_text(chunk)):
            matches.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "match_method": "exact_text",
                    "match_score": 1.0,
                }
            )
    return matches


def fuzzy_matches(
    evidence_text: str,
    chunks: list[dict[str, Any]],
    token_threshold: float,
    sequence_threshold: float,
    max_matches: int,
) -> list[dict[str, Any]]:
    if not compact_text(evidence_text):
        return []
    rows = []
    for chunk in chunks:
        text = chunk_text(chunk)
        tr = token_recall(evidence_text, text)
        sr = sequence_ratio(evidence_text, text)
        score = 0.75 * tr + 0.25 * sr
        if tr >= token_threshold or sr >= sequence_threshold:
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "match_method": "fuzzy_text",
                    "match_score": round(score, 6),
                    "token_recall": round(tr, 6),
                    "sequence_ratio": round(sr, 6),
                }
            )
    return sorted(rows, key=lambda row: row["match_score"], reverse=True)[:max_matches]


def page_fallback_matches(page: int | None, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if page is None:
        return []
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "match_method": "page_fallback",
            "match_score": 1.0,
        }
        for chunk in chunks
        if page_overlaps(chunk, page)
    ]


def map_one_evidence(
    evidence_text: str,
    page: int | None,
    candidates: list[dict[str, Any]],
    token_threshold: float,
    sequence_threshold: float,
    max_fuzzy_matches: int,
    use_page_fallback: bool,
) -> list[dict[str, Any]]:
    matches = exact_matches(evidence_text, candidates)
    if matches:
        return matches

    matches = fuzzy_matches(
        evidence_text=evidence_text,
        chunks=candidates,
        token_threshold=token_threshold,
        sequence_threshold=sequence_threshold,
        max_matches=max_fuzzy_matches,
    )
    if matches:
        return matches

    if use_page_fallback:
        return page_fallback_matches(page, candidates)
    return []


def create_qrels(
    samples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    token_threshold: float,
    sequence_threshold: float,
    max_fuzzy_matches: int,
    use_page_fallback: bool,
    evidence_page_base: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    chunks_by_doc = group_chunks_by_doc(chunks)
    qrels: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    evidence_count = 0
    no_doc_chunk_count = 0

    for sample in tqdm(samples, desc="Create qrels"):
        query_id = str(sample["financebench_id"])
        for evidence_index, evidence in enumerate(sample.get("evidence") or []):
            evidence_count += 1
            ev_doc = evidence_doc_name(sample, evidence)
            ev_page = evidence_page(evidence, evidence_page_base=evidence_page_base)
            ev_text = evidence.get("evidence_text") or evidence.get("evidence_text_full_page") or ""
            candidates = chunks_by_doc.get(ev_doc, [])
            if not candidates:
                no_doc_chunk_count += 1

            matches = map_one_evidence(
                evidence_text=ev_text,
                page=ev_page,
                candidates=candidates,
                token_threshold=token_threshold,
                sequence_threshold=sequence_threshold,
                max_fuzzy_matches=max_fuzzy_matches,
                use_page_fallback=use_page_fallback,
            )
            if not matches:
                unmatched.append(
                    {
                        "query_id": query_id,
                        "financebench_id": query_id,
                        "doc_name": sample.get("doc_name"),
                        "evidence_doc_name": ev_doc,
                        "evidence_page": ev_page,
                        "evidence_page_num_original": evidence.get("evidence_page_num"),
                        "evidence_index": evidence_index,
                        "question": sample.get("question"),
                        "evidence_text_preview": str(ev_text)[:1000],
                        "reason": "no_chunk_match" if candidates else "no_chunks_for_doc",
                    }
                )
                continue

            for match in matches:
                row = {
                    "query_id": query_id,
                    "chunk_id": match["chunk_id"],
                    "relevance": 1,
                    "match_method": match["match_method"],
                    "match_score": match["match_score"],
                    "evidence_index": evidence_index,
                    "evidence_doc_name": ev_doc,
                    "evidence_page": ev_page,
                    "evidence_page_num_original": evidence.get("evidence_page_num"),
                }
                if "token_recall" in match:
                    row["token_recall"] = match["token_recall"]
                    row["sequence_ratio"] = match["sequence_ratio"]
                qrels.append(row)

    method_counts = Counter(row["match_method"] for row in qrels)
    report = {
        "questions": len(samples),
        "chunks": len(chunks),
        "evidence_rows": evidence_count,
        "qrels": len(qrels),
        "unmatched_evidence": len(unmatched),
        "docs_with_chunks": len(chunks_by_doc),
        "evidence_rows_without_doc_chunks": no_doc_chunk_count,
        "match_method_counts": dict(method_counts),
        "settings": {
            "fuzzy_token_threshold": token_threshold,
            "fuzzy_sequence_threshold": sequence_threshold,
            "max_fuzzy_matches": max_fuzzy_matches,
            "use_page_fallback": use_page_fallback,
            "evidence_page_base": evidence_page_base,
        },
    }
    return qrels, unmatched, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create retrieval evaluation files from FinanceBench questions/evidence and local chunks."
    )
    parser.add_argument("--financebench", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fuzzy-token-threshold", type=float, default=0.60)
    parser.add_argument("--fuzzy-sequence-threshold", type=float, default=0.18)
    parser.add_argument("--max-fuzzy-matches", type=int, default=5)
    parser.add_argument("--no-page-fallback", action="store_true")
    parser.add_argument(
        "--evidence-page-base",
        type=int,
        choices=[0, 1],
        default=0,
        help="FinanceBench uses 0. Use 1 only if your benchmark file was already converted.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    samples = load_jsonl(args.financebench)
    chunks = load_chunks(args.chunks)
    queries = make_queries(samples)
    qrels, unmatched, report = create_qrels(
        samples=samples,
        chunks=chunks,
        token_threshold=args.fuzzy_token_threshold,
        sequence_threshold=args.fuzzy_sequence_threshold,
        max_fuzzy_matches=args.max_fuzzy_matches,
        use_page_fallback=not args.no_page_fallback,
        evidence_page_base=args.evidence_page_base,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "queries.jsonl", queries)
    write_jsonl(args.out_dir / "qrels.jsonl", qrels)
    write_jsonl(args.out_dir / "unmatched_evidence.jsonl", unmatched)
    write_json(args.out_dir / "mapping_report.json", report)
    LOGGER.info("Wrote mapping files to %s", args.out_dir)


if __name__ == "__main__":
    main()
