from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_BGE_MODEL = "BAAI/bge-base-en-v1.5"


def _missing_dependency_error() -> RuntimeError:
    return RuntimeError(
        "Missing sentence-transformers. Install it with:\n"
        "  python -m pip install sentence-transformers\n\n"
        "For FAISS indexing, also install:\n"
        "  python -m pip install faiss-cpu"
    )


@dataclass
class BGEEmbedder:
    model_name: str = DEFAULT_BGE_MODEL
    device: str | None = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error() from exc

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        self.model = SentenceTransformer(self.model_name, **kwargs)

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        texts = [str(text or "") for text in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        prefixed = [self.query_prefix + str(query or "") for query in queries]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")


def load_bge_model(
    model_name: str = DEFAULT_BGE_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
) -> BGEEmbedder:
    return BGEEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        query_prefix=query_prefix,
    )
