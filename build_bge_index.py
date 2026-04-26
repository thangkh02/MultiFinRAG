from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bge_embedder import DEFAULT_BGE_MODEL, load_bge_model


CHUNK_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index_bge")


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing faiss-cpu. Install it with:\n"
            "  python -m pip install faiss-cpu"
        ) from exc
    return faiss


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_file_for(index_name: str) -> Path:
    if index_name == "all":
        return CHUNK_DIR / "all_chunks.jsonl"
    return CHUNK_DIR / f"{index_name}_chunks.jsonl"


def chunk_text(chunk: dict) -> str:
    if chunk.get("modality") == "table":
        return chunk.get("summary") or chunk.get("text") or ""
    if chunk.get("modality") == "image":
        return chunk.get("summary") or chunk.get("text") or ""
    return chunk.get("text") or chunk.get("summary") or ""


def build_index(
    index_name: str,
    model_name: str,
    batch_size: int,
    device: str | None,
    save_embeddings: bool,
) -> dict:
    faiss = require_faiss()
    chunks = load_jsonl(chunk_file_for(index_name))
    texts = [chunk_text(chunk) for chunk in chunks]

    embedder = load_bge_model(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    embeddings = embedder.encode_documents(texts)

    # encode_documents(normalize_embeddings=True) makes inner product equivalent to cosine similarity.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / f"{index_name}.faiss"
    ids_path = INDEX_DIR / f"{index_name}_chunk_ids.json"
    meta_path = INDEX_DIR / f"{index_name}_meta.json"
    embeddings_path = INDEX_DIR / f"{index_name}_embeddings.npy"

    faiss.write_index(index, str(index_path))
    ids_path.write_text(
        json.dumps([chunk["id"] for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if save_embeddings:
        np.save(embeddings_path, embeddings)

    meta = {
        "index_name": index_name,
        "model_name": model_name,
        "chunks": len(chunks),
        "dimension": dim,
        "normalized_embeddings": True,
        "metric": "inner_product_cosine",
        "index_path": str(index_path).replace("\\", "/"),
        "chunk_ids_path": str(ids_path).replace("\\", "/"),
        "embeddings_path": str(embeddings_path).replace("\\", "/") if save_embeddings else None,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indexes with BAAI/bge-base-en-v1.5 embeddings.")
    parser.add_argument("--index", choices=["text", "table", "image", "all"], default="all")
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Example: cuda, cpu. Leave empty for sentence-transformers auto device.")
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()

    if args.index == "all":
        indexes = ["text", "table", "image", "all"]
    else:
        indexes = [args.index]

    summary = {}
    for index_name in indexes:
        print(f"Building BGE FAISS index: {index_name}")
        summary[index_name] = build_index(
            index_name=index_name,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            save_embeddings=args.save_embeddings,
        )

    summary_path = INDEX_DIR / "bge_index_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
