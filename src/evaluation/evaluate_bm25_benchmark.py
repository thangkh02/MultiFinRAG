from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9._%$-]*", str(text or "").lower())


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.docs = [tokenize(chunk.get("text")) for chunk in chunks]
        self.doc_lens = np.array([len(doc) for doc in self.docs], dtype="float32")
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) else 0.0
        self.term_freqs = [Counter(doc) for doc in self.docs]
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n_docs = len(self.docs)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int, source_pdf: str | None = None, filter_source: bool = False) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.docs), dtype="float32")
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_lens[idx] / max(self.avgdl, 1e-9))
                scores[idx] += idf * (freq * (self.k1 + 1.0) / denom)

        order = np.argsort(-scores)
        rows = []
        for idx in order:
            if scores[idx] <= 0:
                break
            chunk = self.chunks[int(idx)]
            if filter_source and source_pdf and chunk.get("source_pdf") != source_pdf:
                continue
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(scores[idx]),
                    "chunk": chunk,
                }
            )
            if len(rows) >= top_k:
                break
        return rows


def evaluate(
    benchmark_dir: Path,
    output_dir: Path,
    top_k: int,
    k_values: list[int],
    filter_source: bool,
) -> dict[str, Any]:
    queries = load_jsonl(benchmark_dir / "queries.jsonl")
    qrels = load_jsonl(benchmark_dir / "qrels.jsonl")
    corpus = load_jsonl(benchmark_dir / "corpus.jsonl")
    index = BM25Index(corpus)

    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        relevant_by_query[row["query_id"]].add(row["chunk_id"])

    by_question = []
    retrieval_rows = []
    for query in queries:
        qid = query["query_id"]
        relevant_ids = relevant_by_query.get(qid, set())
        hits = index.search(
            query=query["question"],
            top_k=top_k,
            source_pdf=query.get("source_pdf"),
            filter_source=filter_source,
        )
        retrieved_ids = [row["chunk_id"] for row in hits]
        for rank, hit in enumerate(hits, start=1):
            chunk = hit["chunk"]
            retrieval_rows.append(
                {
                    "query_id": qid,
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "score": hit["score"],
                    "is_relevant": hit["chunk_id"] in relevant_ids,
                    "query_type": query.get("type"),
                    "chunk_modality": chunk.get("modality"),
                    "source_pdf": chunk.get("source_pdf"),
                    "page": chunk.get("page"),
                }
            )

        row = {
            "query_id": qid,
            "type": query.get("type", "unknown"),
            "question": query.get("question"),
            "num_relevant": len(relevant_ids),
            "retrieved_ids": retrieved_ids,
        }
        for k in k_values:
            top_ids = retrieved_ids[:k]
            num_hits = len(set(top_ids) & relevant_ids)
            row[f"hit@{k}"] = 1.0 if num_hits else 0.0
            row[f"recall@{k}"] = num_hits / len(relevant_ids) if relevant_ids else 0.0
            row[f"precision@{k}"] = num_hits / k
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
        "method": "bm25",
        "filter_source": filter_source,
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
    parser = argparse.ArgumentParser(description="Evaluate BM25 lexical retrieval on a chunk-level benchmark.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark_report"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval_benchmark_report_bm25"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--filter-source", action="store_true")
    args = parser.parse_args()

    summary = evaluate(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        k_values=args.k_values,
        filter_source=args.filter_source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
