from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$%.\-()]+", str(text or "").lower())


def doc_key(value: Any) -> str:
    return Path(str(value or "").replace("\\", "/")).stem.lower()


def method_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def chunk_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or chunk.get("summary") or "")


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32, copy=False)
    mins = scores.min(axis=1, keepdims=True)
    maxs = scores.max(axis=1, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-8)
    return (scores - mins) / denom


def topk_from_scores(
    method: str,
    scores: np.ndarray,
    query_ids: list[str],
    chunk_ids: list[str],
    chunks_by_id: dict[str, dict[str, Any]],
    top_k: int,
    allowed_by_query: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_chunks = len(chunk_ids)
    for qi, qid in enumerate(query_ids):
        q_scores = scores[qi].copy()
        if allowed_by_query is not None:
            allowed = allowed_by_query.get(qid, set())
            mask = np.ones(n_chunks, dtype=bool)
            if allowed:
                allowed_idx = {idx for idx, cid in enumerate(chunk_ids) if cid in allowed}
                for idx in allowed_idx:
                    mask[idx] = False
            q_scores[mask] = -np.inf
        take = min(top_k, n_chunks)
        idxs = np.argpartition(-q_scores, np.arange(take))[:take]
        idxs = idxs[np.argsort(-q_scores[idxs])]
        for rank, idx in enumerate(idxs, start=1):
            if not np.isfinite(q_scores[idx]):
                continue
            chunk_id = chunk_ids[int(idx)]
            chunk = chunks_by_id[chunk_id]
            rows.append(
                {
                    "method": method,
                    "query_id": qid,
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": float(q_scores[idx]),
                    "doc_name": chunk.get("doc_name"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "chunk_type": chunk.get("chunk_type"),
                    "modality": chunk.get("modality"),
                    "text_preview": chunk_text(chunk)[:300],
                }
            )
    return rows


def rerank_candidates(
    method: str,
    candidate_results: list[dict[str, Any]],
    queries_by_id: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    top_k: int,
    model_name: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    from sentence_transformers import CrossEncoder

    LOGGER.info("Loading reranker: %s", model_name)
    model = CrossEncoder(model_name, max_length=512, device="cpu")
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_results:
        by_query[row["query_id"]].append(row)

    output: list[dict[str, Any]] = []
    for qid, rows in tqdm(by_query.items(), desc="Rerank"):
        rows = sorted(rows, key=lambda row: row["rank"])
        pairs = [(queries_by_id[qid]["question"], chunk_text(chunks_by_id[row["chunk_id"]])) for row in rows]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scored = sorted(zip(rows, scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        for rank, (row, score) in enumerate(scored, start=1):
            chunk = chunks_by_id[row["chunk_id"]]
            output.append(
                {
                    "method": method,
                    "query_id": qid,
                    "rank": rank,
                    "chunk_id": row["chunk_id"],
                    "score": float(score),
                    "base_rank": row["rank"],
                    "base_score": row["score"],
                    "doc_name": chunk.get("doc_name"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "chunk_type": chunk.get("chunk_type"),
                    "modality": chunk.get("modality"),
                    "text_preview": chunk_text(chunk)[:300],
                }
            )
    return output


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    qrels: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    k_values: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qrels_by_qid: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        qrels_by_qid[row["query_id"]].add(row["chunk_id"])

    by_method_qid: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in all_results:
        by_method_qid[row["method"]][row["query_id"]].append(row)

    per_question: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    query_ids = sorted(qrels_by_qid)

    for method, by_qid in sorted(by_method_qid.items()):
        method_rows: list[dict[str, Any]] = []
        for qid in query_ids:
            relevant_ids = qrels_by_qid[qid]
            retrieved = sorted(by_qid.get(qid, []), key=lambda row: row["rank"])
            retrieved_ids = [row["chunk_id"] for row in retrieved]
            metric_row: dict[str, Any] = {
                "method": method,
                "query_id": qid,
                "num_relevant_chunks": len(relevant_ids),
            }
            for k in k_values:
                top_ids = retrieved_ids[:k]
                hits = [chunk_id for chunk_id in top_ids if chunk_id in relevant_ids]
                metric_row[f"precision@{k}"] = len(hits) / k
                metric_row[f"recall@{k}"] = len(set(hits)) / len(relevant_ids) if relevant_ids else float("nan")
                metric_row[f"hit@{k}"] = 1.0 if hits else 0.0
                metric_row[f"mrr@{k}"] = reciprocal_rank(retrieved_ids, relevant_ids, k)
            method_rows.append(metric_row)
            per_question.append(metric_row)

        summary_row: dict[str, Any] = {"method": method, "queries": len(method_rows)}
        for k in k_values:
            for metric in ("precision", "recall", "hit", "mrr"):
                key = f"{metric}@{k}"
                values = [float(row[key]) for row in method_rows if not np.isnan(float(row[key]))]
                summary_row[key] = float(np.mean(values)) if values else None
        summary.append(summary_row)

    return summary, per_question


def markdown_report(summary: list[dict[str, Any]], notes: list[str]) -> str:
    rows = [
        "| Method | Hit@1 | Recall@1 | MRR@1 | Hit@10 | Recall@10 | MRR@10 | Precision@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        rows.append(
            "| {method} | {hit1:.4f} | {recall1:.4f} | {mrr1:.4f} | {hit10:.4f} | {recall10:.4f} | {mrr10:.4f} | {precision10:.4f} |".format(
                method=row["method"],
                hit1=row["hit@1"],
                recall1=row["recall@1"],
                mrr1=row["mrr@1"],
                hit10=row["hit@10"],
                recall10=row["recall@10"],
                mrr10=row["mrr@10"],
                precision10=row["precision@10"],
            )
        )
    note_text = "\n".join(f"- {note}" for note in notes)
    return "# FinanceBench Filtered Retrieval Eval\n\n" + "\n".join(rows) + "\n\n## Notes\n\n" + note_text + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/financebench_eval_bge"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/financebench_eval_bge/retrieval_eval_filtered"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--hybrid-alpha", type=float, default=0.60)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    chunks_path = args.base_dir / "chunks.jsonl"
    queries_path = args.base_dir / "qrels" / "queries.jsonl"
    qrels_path = args.base_dir / "qrels" / "qrels.jsonl"
    cache_dir = args.base_dir / "retrieval_eval"

    chunks = read_jsonl(chunks_path)
    queries = read_jsonl(queries_path)
    qrels = read_jsonl(qrels_path)
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    queries_by_id = {row["query_id"]: row for row in queries}
    query_ids = [row["query_id"] for row in queries]

    chunk_ids = json.loads((cache_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    cached_query_ids = json.loads((cache_dir / "query_ids.json").read_text(encoding="utf-8"))
    if cached_query_ids != query_ids:
        raise ValueError("Cached query embedding order does not match queries.jsonl")

    LOGGER.info("Loaded %d chunks, %d queries, %d qrels", len(chunks), len(queries), len(qrels))
    LOGGER.info("Chunk modalities: %s", dict(Counter(row.get("modality") or row.get("chunk_type") for row in chunks)))

    chunk_emb = np.load(cache_dir / "bge_small_chunk_embeddings.npy").astype(np.float32)
    query_emb = np.load(cache_dir / "bge_small_query_embeddings.npy").astype(np.float32)
    chunk_emb /= np.maximum(np.linalg.norm(chunk_emb, axis=1, keepdims=True), 1e-12)
    query_emb /= np.maximum(np.linalg.norm(query_emb, axis=1, keepdims=True), 1e-12)
    dense_scores = query_emb @ chunk_emb.T

    tokenized_chunks = [tokenize(chunk_text(chunks_by_id[chunk_id])) for chunk_id in tqdm(chunk_ids, desc="Tokenize chunks")]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = np.vstack(
        [bm25.get_scores(tokenize(queries_by_id[qid]["question"])).astype(np.float32) for qid in tqdm(query_ids, desc="BM25")]
    )

    dense_norm = normalize_rows(dense_scores)
    bm25_norm = normalize_rows(bm25_scores)
    hybrid_scores = args.hybrid_alpha * dense_norm + (1.0 - args.hybrid_alpha) * bm25_norm

    dense_results = topk_from_scores("bge_dense", dense_scores, query_ids, chunk_ids, chunks_by_id, args.top_k)
    bm25_results = topk_from_scores("bm25", bm25_scores, query_ids, chunk_ids, chunks_by_id, args.top_k)
    hybrid_method = f"bm25_bge_hybrid_alpha_{args.hybrid_alpha:.2f}"
    hybrid_results = topk_from_scores(hybrid_method, hybrid_scores, query_ids, chunk_ids, chunks_by_id, args.top_k)

    candidate_results = topk_from_scores(
        f"{hybrid_method}_candidates_top{args.candidate_k}",
        hybrid_scores,
        query_ids,
        chunk_ids,
        chunks_by_id,
        args.candidate_k,
    )

    all_results = dense_results + bm25_results + hybrid_results
    notes = [
        f"Benchmark qrels: {qrels_path}",
        "Filtered qrels remove fuzzy_text labels with token_recall < 0.60.",
        "FinanceBench BGE benchmark chunks are text-only; no table-aware score was run.",
    ]

    if not args.skip_reranker:
        try:
            rerank_results = rerank_candidates(
                f"{hybrid_method}_reranker_{method_key(args.reranker_model)}",
                candidate_results,
                queries_by_id,
                chunks_by_id,
                args.top_k,
                args.reranker_model,
                args.reranker_batch_size,
            )
            all_results += rerank_results
            notes.append(f"Reranker: {args.reranker_model}, reranking hybrid top-{args.candidate_k}.")
        except Exception as exc:  # pragma: no cover - model download/runtime failures are environment dependent.
            LOGGER.exception("Reranker failed")
            notes.append(f"Reranker failed and was skipped: {type(exc).__name__}: {exc}")

    allowed_by_query: dict[str, set[str]] = {}
    doc_to_chunk_ids: dict[str, set[str]] = defaultdict(set)
    for chunk_id in chunk_ids:
        doc_to_chunk_ids[doc_key(chunks_by_id[chunk_id].get("doc_name"))].add(chunk_id)
    for query in queries:
        allowed_by_query[query["query_id"]] = doc_to_chunk_ids.get(doc_key(query.get("doc_name")), set())
    metadata_results = topk_from_scores(
        f"metadata_doc_filter_{hybrid_method}",
        hybrid_scores,
        query_ids,
        chunk_ids,
        chunks_by_id,
        args.top_k,
        allowed_by_query=allowed_by_query,
    )
    all_results += metadata_results
    notes.append("Metadata-aware result uses FinanceBench doc_name as an oracle document filter, so it is not directly comparable to open-corpus retrieval.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "bm25_retrieval_results.jsonl", bm25_results)
    write_jsonl(args.out_dir / "bge_dense_retrieval_results.jsonl", dense_results)
    write_jsonl(args.out_dir / "hybrid_retrieval_results.jsonl", hybrid_results)
    write_jsonl(args.out_dir / "hybrid_candidates_top50.jsonl", candidate_results)
    write_jsonl(args.out_dir / "metadata_aware_retrieval_results.jsonl", metadata_results)
    write_jsonl(args.out_dir / "retrieval_results.jsonl", all_results)

    summary, per_question = compute_metrics(qrels, all_results, [1, 3, 5, 10])
    write_json(args.out_dir / "metrics_summary.json", summary)
    write_jsonl(args.out_dir / "metrics_by_question.jsonl", per_question)
    (args.out_dir / "retrieval_eval_report.md").write_text(markdown_report(summary, notes), encoding="utf-8")
    LOGGER.info("Wrote results to %s", args.out_dir)
    print(markdown_report(summary, notes))


if __name__ == "__main__":
    main()
