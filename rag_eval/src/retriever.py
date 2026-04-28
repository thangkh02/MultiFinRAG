from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model
from .index_builder import require_faiss
from .schema import chunk_doc_name, chunk_text, page_range

LOGGER = logging.getLogger(__name__)


def retrieve(
    samples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    index_dir: Path,
    top_k: int,
    model_name: str = DEFAULT_BGE_MODEL,
    batch_size: int = 32,
    device: str | None = None,
) -> list[dict[str, Any]]:
    faiss = require_faiss()
    index_path = index_dir / "chunks.faiss"
    ids_path = index_dir / "chunk_ids.json"
    index = faiss.read_index(str(index_path))
    index_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    LOGGER.info("Embedding %d queries with %s", len(samples), model_name)
    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    query_vectors = embedder.encode_queries([sample["question"] for sample in samples])
    scores, indices = index.search(query_vectors, min(top_k, len(index_ids)))

    results: list[dict[str, Any]] = []
    for sample, sample_scores, sample_indices in zip(samples, scores, indices):
        qid = str(sample["financebench_id"])
        for rank, (score, idx) in enumerate(zip(sample_scores, sample_indices), start=1):
            if idx < 0:
                continue
            chunk_id = index_ids[int(idx)]
            chunk = chunks_by_id.get(chunk_id, {})
            start, end = page_range(chunk)
            results.append(
                {
                    "qid": qid,
                    "question": sample.get("question"),
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "doc_name": chunk_doc_name(chunk),
                    "page_start": start,
                    "page_end": end,
                    "chunk_type": chunk.get("chunk_type") or chunk.get("modality"),
                    "text_preview": chunk_text(chunk)[:500],
                }
            )
    LOGGER.info("Retrieved %d rows", len(results))
    return results
