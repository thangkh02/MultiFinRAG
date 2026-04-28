from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if not math.isnan(value) else "nan"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary: dict[str, Any],
    by_question: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    k_rows = []
    for k in summary["k_values"]:
        metrics = summary["metrics"]
        k_rows.append(
            {
                "k": k,
                "precision": metrics.get(f"precision@{k}"),
                "recall": metrics.get(f"recall@{k}"),
                "hit": metrics.get(f"hit@{k}"),
                "mrr": metrics.get(f"mrr@{k}"),
                "page_recall": metrics.get(f"page_recall@{k}"),
                "evidence_recall": metrics.get(f"evidence_recall@{k}"),
            }
        )
    method_rows = [{"method": key, "count": value} for key, value in Counter(q["match_method"] for q in qrels).items()]
    hard_rows = sorted(by_question, key=lambda row: (row.get("hit@10", 0), row.get("recall@10", 0)))[:20]
    hard_display = [
        {
            "qid": row["qid"],
            "doc_name": row.get("doc_name"),
            "hit@10": row.get("hit@10"),
            "recall@10": row.get("recall@10"),
            "qrel_methods": row.get("qrel_methods"),
            "question": str(row.get("question") or "")[:120],
        }
        for row in hard_rows
    ]
    lines = [
        "# FinanceBench Retrieval Evaluation Report",
        "",
        "## Inputs",
        "",
        f"- questions: `{config['financebench']['questions_path']}`",
        f"- pdf_dir: `{config['financebench']['pdf_dir']}`",
        f"- source_chunks_path: `{config['chunking'].get('source_chunks_path')}`",
        f"- chunk_output: `{config['chunking']['output_chunks_path']}`",
        "",
        "## Aggregate Metrics",
        "",
        markdown_table(k_rows, ["k", "precision", "recall", "hit", "mrr", "page_recall", "evidence_recall"]),
        "",
        "## Qrel Construction",
        "",
        markdown_table(method_rows, ["method", "count"]),
        "",
        f"Unmatched evidence rows: {len(unmatched)}",
        "",
        "## Error Analysis Sample",
        "",
        markdown_table(hard_display, ["qid", "doc_name", "hit@10", "recall@10", "qrel_methods", "question"]),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
