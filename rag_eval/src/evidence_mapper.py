from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from .schema import chunk_doc_name, chunk_text, page_overlaps
from .text_utils import compact_text, doc_key, sequence_ratio, token_recall

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MappingConfig:
    fuzzy_token_threshold: float = 0.60
    fuzzy_sequence_threshold: float = 0.18
    max_fuzzy_matches: int = 5
    use_page_fallback: bool = True


def evidence_doc_name(sample: dict[str, Any], evidence: dict[str, Any]) -> str:
    return doc_key(evidence.get("evidence_doc_name") or evidence.get("doc_name") or sample.get("doc_name"))


def evidence_page(evidence: dict[str, Any]) -> int | None:
    page = evidence.get("evidence_page_num")
    if page is None:
        return None
    return int(page) + 1


def match_evidence(
    sample: dict[str, Any],
    evidence: dict[str, Any],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    all_chunks: list[dict[str, Any]],
    config: MappingConfig,
) -> list[dict[str, Any]]:
    ev_text = evidence.get("evidence_text") or evidence.get("evidence_text_full_page") or ""
    ev_doc = evidence_doc_name(sample, evidence)
    ev_page = evidence_page(evidence)
    candidates = chunks_by_doc.get(ev_doc) or all_chunks
    ev_compact = compact_text(ev_text)

    exact = []
    if ev_compact:
        for chunk in candidates:
            if ev_compact in compact_text(chunk_text(chunk)):
                exact.append({"chunk_id": chunk["chunk_id"], "match_method": "exact_text", "match_score": 1.0})
    if exact:
        return exact

    fuzzy = []
    if ev_compact:
        for chunk in candidates:
            text = chunk_text(chunk)
            tr = token_recall(ev_text, text)
            sr = sequence_ratio(ev_text, text)
            score = 0.75 * tr + 0.25 * sr
            if tr >= config.fuzzy_token_threshold or sr >= config.fuzzy_sequence_threshold:
                fuzzy.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "match_method": "fuzzy_text",
                        "match_score": round(score, 6),
                        "token_recall": round(tr, 6),
                        "sequence_ratio": round(sr, 6),
                    }
                )
    if fuzzy:
        return sorted(fuzzy, key=lambda row: row["match_score"], reverse=True)[: config.max_fuzzy_matches]

    if config.use_page_fallback:
        page_matches = [
            {"chunk_id": chunk["chunk_id"], "match_method": "page_fallback", "match_score": 1.0}
            for chunk in candidates
            if page_overlaps(chunk, ev_page)
        ]
        if page_matches:
            return page_matches
    return []


def build_qrels(
    samples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    config: MappingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk_doc_name(chunk)].append(chunk)

    qrels = []
    unmatched = []
    for sample in tqdm(samples, desc="Map evidence to chunks"):
        qid = str(sample["financebench_id"])
        for evidence_index, evidence in enumerate(sample.get("evidence") or []):
            matches = match_evidence(sample, evidence, chunks_by_doc, chunks, config)
            ev_doc = evidence_doc_name(sample, evidence)
            ev_page = evidence_page(evidence)
            if not matches:
                unmatched.append(
                    {
                        "qid": qid,
                        "doc_name": sample.get("doc_name"),
                        "evidence_doc_name": ev_doc,
                        "evidence_page": ev_page,
                        "evidence_index": evidence_index,
                        "question": sample.get("question"),
                        "evidence_text_preview": str(evidence.get("evidence_text") or "")[:500],
                    }
                )
                continue
            for match in matches:
                row = {
                    "qid": qid,
                    "chunk_id": match["chunk_id"],
                    "relevance": 1,
                    "match_method": match["match_method"],
                    "match_score": match["match_score"],
                    "evidence_index": evidence_index,
                    "evidence_doc_name": ev_doc,
                    "evidence_page": ev_page,
                }
                if "token_recall" in match:
                    row["token_recall"] = match["token_recall"]
                    row["sequence_ratio"] = match["sequence_ratio"]
                qrels.append(row)
    LOGGER.info("Built %d qrels; unmatched evidence rows: %d", len(qrels), len(unmatched))
    return qrels, unmatched
