from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INDEX_DIR = Path("data/index_bge")


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def load_ids(name: str) -> list[str]:
    return json.loads((INDEX_DIR / f"{name}_chunk_ids.json").read_text(encoding="utf-8"))


def main() -> None:
    faiss = require_faiss()
    names = ["text", "table", "image"]
    vectors = []
    chunk_ids = []
    metas = {}

    for name in names:
        index_path = INDEX_DIR / f"{name}.faiss"
        meta_path = INDEX_DIR / f"{name}_meta.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing {index_path}. Build it first.")
        index = faiss.read_index(str(index_path))
        vec = index.reconstruct_n(0, index.ntotal)
        vectors.append(vec)
        chunk_ids.extend(load_ids(name))
        if meta_path.exists():
            metas[name] = json.loads(meta_path.read_text(encoding="utf-8"))

    all_vectors = np.vstack(vectors).astype("float32")
    all_index = faiss.IndexFlatIP(all_vectors.shape[1])
    all_index.add(all_vectors)

    all_index_path = INDEX_DIR / "all.faiss"
    all_ids_path = INDEX_DIR / "all_chunk_ids.json"
    all_meta_path = INDEX_DIR / "all_meta.json"

    faiss.write_index(all_index, str(all_index_path))
    all_ids_path.write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    model_name = next((meta.get("model_name") for meta in metas.values() if meta.get("model_name")), None)
    meta = {
        "index_name": "all",
        "model_name": model_name,
        "chunks": len(chunk_ids),
        "dimension": int(all_vectors.shape[1]),
        "normalized_embeddings": True,
        "metric": "inner_product_cosine",
        "sources": names,
        "index_path": str(all_index_path).replace("\\", "/"),
        "chunk_ids_path": str(all_ids_path).replace("\\", "/"),
    }
    all_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
