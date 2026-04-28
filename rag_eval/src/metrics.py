from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .schema import chunk_doc_name, page_overlaps


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    samples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    k_values: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    qrels_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    results_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        qrels_by_qid[row["qid"]].append(row)
    for row in retrieval_results:
        results_by_qid[row["qid"]].append(row)

    by_question: list[dict[str, Any]] = []
    for sample in samples:
        qid = str(sample["financebench_id"])
        rel_rows = qrels_by_qid.get(qid, [])
        rel_ids = {row["chunk_id"] for row in rel_rows}
        text_rel_ids = {row["chunk_id"] for row in rel_rows if row["match_method"] in {"exact_text", "fuzzy_text"}}
        evidence_pages = {row.get("evidence_page") for row in rel_rows if row.get("evidence_page") is not None}
        evidence_docs = {row.get("evidence_doc_name") for row in rel_rows if row.get("evidence_doc_name")}
        retrieved = sorted(results_by_qid.get(qid, []), key=lambda row: row["rank"])
        retrieved_ids = [row["chunk_id"] for row in retrieved]
        row = {
            "qid": qid,
            "company": sample.get("company"),
            "doc_name": sample.get("doc_name"),
            "question_type": sample.get("question_type"),
            "question_reasoning": sample.get("question_reasoning"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "num_relevant_chunks": len(rel_ids),
            "qrel_methods": ",".join(sorted({rel["match_method"] for rel in rel_rows})),
        }
        for k in k_values:
            top_ids = retrieved_ids[:k]
            rel_hits = [chunk_id for chunk_id in top_ids if chunk_id in rel_ids]
            page_hit = False
            evidence_hit = False
            for chunk_id in top_ids:
                chunk = chunks_by_id.get(chunk_id, {})
                same_doc = not evidence_docs or chunk_doc_name(chunk) in evidence_docs
                if same_doc and any(page_overlaps(chunk, page) for page in evidence_pages):
                    page_hit = True
                if chunk_id in text_rel_ids:
                    evidence_hit = True
            row[f"precision@{k}"] = len(rel_hits) / k
            row[f"recall@{k}"] = len(set(rel_hits)) / len(rel_ids) if rel_ids else math.nan
            row[f"hit@{k}"] = 1.0 if rel_hits else 0.0
            row[f"mrr@{k}"] = reciprocal_rank(retrieved_ids, rel_ids, k)
            row[f"page_recall@{k}"] = 1.0 if page_hit else 0.0
            row[f"evidence_recall@{k}"] = 1.0 if evidence_hit else 0.0
        by_question.append(row)

    summary: dict[str, Any] = {"questions": len(samples), "k_values": k_values, "metrics": {}}
    for k in k_values:
        for metric in ("precision", "recall", "hit", "mrr", "page_recall", "evidence_recall"):
            key = f"{metric}@{k}"
            values = [row[key] for row in by_question]
            values = [value for value in values if not (isinstance(value, float) and math.isnan(value))]
            summary["metrics"][key] = float(np.mean(values)) if values else None
    return summary, by_question
