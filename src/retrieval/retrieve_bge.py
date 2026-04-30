from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model


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


def load_chunks_by_id(index_name: str) -> dict[str, dict]:
    return {chunk["id"]: chunk for chunk in load_jsonl(chunk_file_for(index_name))}


def chunk_preview_text(chunk: dict) -> str:
    return chunk.get("embed_text") or chunk.get("text") or chunk.get("summary") or ""


def retrieve_bge(
    query: str,
    index_name: str = "all",
    top_k: int = 5,
    source_pdf: str | None = None,
    model_name: str = DEFAULT_BGE_MODEL,
    device: str | None = None,
) -> list[dict]:
    faiss = require_faiss()

    index_path = INDEX_DIR / f"{index_name}.faiss"
    ids_path = INDEX_DIR / f"{index_name}_chunk_ids.json"
    if not index_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Missing BGE index for '{index_name}'. Build it first:\n"
            f"  python src/indexing/build_bge_index.py --index {index_name}"
        )

    index = faiss.read_index(str(index_path))
    chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    chunks_by_id = load_chunks_by_id(index_name)

    embedder = load_bge_model(model_name=model_name, device=device)
    query_vec = embedder.encode_queries([query])

    search_k = min(max(top_k * 20, top_k), len(chunk_ids))
    scores, indices = index.search(query_vec, search_k)

    rows = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks_by_id[chunk_ids[idx]]
        if source_pdf and chunk.get("source_pdf") != source_pdf:
            continue
        row = dict(chunk)
        row["score"] = float(score)
        rows.append(row)
        if len(rows) >= top_k:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve chunks from BGE FAISS indexes.")
    parser.add_argument("query")
    parser.add_argument("--index", choices=["text", "table", "image", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-pdf")
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--device")
    args = parser.parse_args()

    rows = retrieve_bge(
        query=args.query,
        index_name=args.index,
        top_k=args.top_k,
        source_pdf=args.source_pdf,
        model_name=args.model,
        device=args.device,
    )
    for row in rows:
        preview = chunk_preview_text(row)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "modality": row["modality"],
                    "score": round(row["score"], 4),
                    "source_pdf": row.get("source_pdf"),
                    "page": row.get("page"),
                    "preview": preview[:500],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
