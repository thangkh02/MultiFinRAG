from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .chunk_adapter import prepare_chunks
from .evidence_mapper import MappingConfig, build_qrels
from .index_builder import build_index
from .io_utils import read_yaml, write_json, write_jsonl
from .load_financebench import load_financebench
from .metrics import compute_metrics
from .report import write_report
from .retriever import retrieve


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    path_keys = {
        ("financebench", "questions_path"),
        ("financebench", "pdf_dir"),
        ("chunking", "source_chunks_path"),
        ("chunking", "output_chunks_path"),
        ("index", "output_dir"),
        ("mapping", "qrels_path"),
        ("mapping", "unmatched_path"),
        ("retrieval", "results_path"),
        ("metrics", "summary_path"),
        ("metrics", "by_question_path"),
        ("report", "path"),
    }
    for section, key in path_keys:
        value = config.get(section, {}).get(key)
        if not value:
            continue
        path = Path(value)
        config[section][key] = str(path if path.is_absolute() else root / path)
    # Make pdf_dir available to chunk_adapter without coupling it to the whole config.
    config["chunking"]["pdf_dir"] = config["financebench"]["pdf_dir"]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinanceBench retrieval evaluation.")
    parser.add_argument("--config", type=Path, default=Path("rag_eval/config.yaml"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-index", action="store_true", help="Reuse an existing index in config index.output_dir.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = args.config.resolve().parents[1]
    config = resolve_paths(read_yaml(args.config), root)

    samples = load_financebench(Path(config["financebench"]["questions_path"]))
    chunks = prepare_chunks(config["chunking"], samples)

    mapping_cfg = MappingConfig(
        fuzzy_token_threshold=float(config["mapping"].get("fuzzy_token_threshold", 0.60)),
        fuzzy_sequence_threshold=float(config["mapping"].get("fuzzy_sequence_threshold", 0.18)),
        max_fuzzy_matches=int(config["mapping"].get("max_fuzzy_matches", 5)),
        use_page_fallback=bool(config["mapping"].get("use_page_fallback", True)),
    )
    qrels, unmatched = build_qrels(samples, chunks, mapping_cfg)
    write_jsonl(Path(config["mapping"]["qrels_path"]), qrels)
    write_jsonl(Path(config["mapping"]["unmatched_path"]), unmatched)

    if not args.skip_index:
        build_index(chunks, config["index"])

    retrieval_rows = retrieve(
        samples=samples,
        chunks=chunks,
        index_dir=Path(config["index"]["output_dir"]),
        top_k=int(config["retrieval"]["top_k"]),
        model_name=str(config["index"]["model_name"]),
        batch_size=int(config["index"].get("batch_size") or 32),
        device=config["index"].get("device"),
    )
    write_jsonl(Path(config["retrieval"]["results_path"]), retrieval_rows)

    summary, by_question = compute_metrics(
        samples=samples,
        chunks=chunks,
        qrels=qrels,
        retrieval_results=retrieval_rows,
        k_values=[int(k) for k in config["retrieval"]["k_values"]],
    )
    write_json(Path(config["metrics"]["summary_path"]), summary)
    write_jsonl(Path(config["metrics"]["by_question_path"]), by_question)
    write_report(Path(config["report"]["path"]), summary, by_question, qrels, unmatched, config)

    logging.getLogger(__name__).info("Evaluation complete. Report: %s", config["report"]["path"])


if __name__ == "__main__":
    main()
