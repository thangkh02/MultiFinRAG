from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model
from .schema import chunk_text

LOGGER = logging.getLogger(__name__)


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def build_index(chunks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    faiss = require_faiss()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.get("model_name") or DEFAULT_BGE_MODEL
    batch_size = int(config.get("batch_size") or 32)
    device = config.get("device")

    texts = [chunk_text(chunk) for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    LOGGER.info("Embedding %d chunks with %s", len(chunks), model_name)
    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    embeddings = embedder.encode_documents(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = output_dir / "chunks.faiss"
    ids_path = output_dir / "chunk_ids.json"
    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "meta.json"
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(embeddings_path, embeddings)
    meta = {
        "model_name": model_name,
        "chunks": len(chunks),
        "dimension": int(embeddings.shape[1]),
        "normalized_embeddings": True,
        "metric": "inner_product_cosine",
        "index_path": str(index_path).replace("\\", "/"),
        "ids_path": str(ids_path).replace("\\", "/"),
        "embeddings_path": str(embeddings_path).replace("\\", "/"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote FAISS index to %s", index_path)
    return meta
