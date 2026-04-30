from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model  # noqa: E402


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def rerank_score(query: dict[str, Any], chunk: dict[str, Any], dense_score: float) -> float:
    score = float(dense_score)
    qtype = query.get("type")
    modality = chunk.get("modality")
    if qtype == modality:
        score += 0.08
    if qtype == "multimodal" and modality in {"text", "table", "image"}:
        score += 0.03
    source_pdf = query.get("source_pdf")
    if source_pdf and chunk.get("source_pdf") == source_pdf:
        score += 0.06
    return score


def evaluate(
    benchmark_dir: Path,
    chunks_path: Path,
    index_path: Path,
    ids_path: Path,
    output_dir: Path,
    top_k: int,
    k_values: list[int],
    model_name: str,
    batch_size: int,
    device: str | None,
    filter_source: bool,
    rerank: bool,
) -> dict[str, Any]:
    faiss = require_faiss()
    queries = load_jsonl(benchmark_dir / "queries.jsonl")
    qrels = load_jsonl(benchmark_dir / "qrels.jsonl")
    chunks = load_jsonl(chunks_path)
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    index_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        relevant_by_query[row["query_id"]].add(row["chunk_id"])

    index = faiss.read_index(str(index_path))
    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    query_vectors = embedder.encode_queries([row["question"] for row in queries])

    search_k = min(max(top_k * 30, top_k), len(index_ids))
    scores, indices = index.search(query_vectors, search_k)

    by_question = []
    retrieval_rows = []
    for query, query_scores, query_indices in zip(queries, scores, indices):
        qid = query["query_id"]
        source_pdf = query.get("source_pdf")
        relevant_ids = relevant_by_query.get(qid, set())
        retrieved_ids: list[str] = []

        candidates = []
        for score, idx in zip(query_scores, query_indices):
            if idx < 0:
                continue
            chunk_id = index_ids[int(idx)]
            chunk = chunks_by_id.get(chunk_id, {})
            if filter_source and source_pdf and chunk.get("source_pdf") != source_pdf:
                continue
            final_score = rerank_score(query, chunk, float(score)) if rerank else float(score)
            candidates.append((final_score, float(score), chunk_id, chunk))

        candidates.sort(key=lambda row: row[0], reverse=True)
        for final_score, dense_score, chunk_id, chunk in candidates:
            retrieved_ids.append(chunk_id)
            retrieval_rows.append(
                {
                    "query_id": qid,
                    "rank": len(retrieved_ids),
                    "chunk_id": chunk_id,
                    "score": final_score,
                    "dense_score": dense_score,
                    "is_relevant": chunk_id in relevant_ids,
                    "query_type": query.get("type"),
                    "chunk_modality": chunk.get("modality"),
                    "source_pdf": chunk.get("source_pdf"),
                    "page": chunk.get("page"),
                }
            )
            if len(retrieved_ids) >= top_k:
                break

        row = {
            "query_id": qid,
            "type": query.get("type", "unknown"),
            "question": query.get("question"),
            "num_relevant": len(relevant_ids),
            "retrieved_ids": retrieved_ids[:top_k],
        }
        for k in k_values:
            top_ids = retrieved_ids[:k]
            hits = len(set(top_ids) & relevant_ids)
            row[f"hit@{k}"] = 1.0 if hits else 0.0
            row[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0
            row[f"precision@{k}"] = hits / k
            row[f"mrr@{k}"] = reciprocal_rank(top_ids, relevant_ids, k)
        by_question.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_question:
        grouped[row["type"]].append(row)
        grouped["all"].append(row)

    summary = {
        "benchmark_dir": str(benchmark_dir).replace("\\", "/"),
        "queries": len(queries),
        "qrels": len(qrels),
        "top_k": top_k,
        "filter_source": filter_source,
        "rerank": rerank,
        "metrics": {},
    }
    for qtype, rows in grouped.items():
        summary["metrics"][qtype] = {"queries": len(rows)}
        for k in k_values:
            for metric in ("hit", "recall", "precision", "mrr"):
                key = f"{metric}@{k}"
                summary["metrics"][qtype][key] = float(np.mean([row[key] for row in rows])) if rows else 0.0

    write_json(output_dir / "metrics_summary.json", summary)
    write_jsonl(output_dir / "metrics_by_question.jsonl", by_question)
    write_jsonl(output_dir / "retrieval_results.jsonl", retrieval_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunk-level retrieval benchmark.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark_hard"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/all_chunks.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/index_bge/all.faiss"))
    parser.add_argument("--ids", type=Path, default=Path("data/index_bge/all_chunk_ids.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval_benchmark_hard"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--filter-source", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="Apply a lightweight source/modality-aware reranker.")
    args = parser.parse_args()

    summary = evaluate(
        benchmark_dir=args.benchmark_dir,
        chunks_path=args.chunks,
        index_path=args.index,
        ids_path=args.ids,
        output_dir=args.output_dir,
        top_k=args.top_k,
        k_values=args.k_values,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        filter_source=args.filter_source,
        rerank=args.rerank,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
