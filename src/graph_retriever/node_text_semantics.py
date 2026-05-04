from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.common.bge_embedder import BGEEmbedder

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def load_json_or_jsonl(path_value: str | Path | None) -> list[dict[str, Any]] | dict[str, Any]:
    path = _resolve_path(path_value)
    if path is None:
        return []
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return []

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Invalid JSON: %s", path)
        return []
    if isinstance(payload, (list, dict)):
        return payload
    return []


def build_chunk_lookup(
    *,
    chunk_text_path: str | Path | None,
    id2node: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    raw = load_json_or_jsonl(chunk_text_path)
    records: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        if isinstance(raw.get("chunks"), list):
            records = [r for r in raw["chunks"] if isinstance(r, dict)]
        else:
            for key, value in raw.items():
                if not isinstance(value, dict):
                    continue
                record = dict(value)
                record.setdefault("chunk_id", key)
                records.append(record)
    elif isinstance(raw, list):
        records = [r for r in raw if isinstance(r, dict)]

    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        keys = (
            record.get("chunk_id"),
            record.get("id"),
            record.get("node_id"),
            record.get("chunk_uid"),
            record.get("source_id"),
            record.get("uid"),
        )
        for key in keys:
            if key is None:
                continue
            text_key = str(key).strip()
            if not text_key:
                continue
            lookup.setdefault(text_key, record)
        chunk_id = record.get("chunk_id") or record.get("id")
        if chunk_id is not None:
            lookup.setdefault(f"chunk:{chunk_id}", record)

    if id2node:
        for node_id, node_uid in id2node.items():
            if node_uid in lookup:
                lookup.setdefault(str(node_id), lookup[node_uid])

    return lookup


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        items: list[str] = []
        for key, val in value.items():
            if isinstance(val, list):
                joined = ", ".join(str(v) for v in val if str(v).strip())
                if joined:
                    items.append(f"{key}: {joined}")
            elif val is not None and str(val).strip():
                items.append(f"{key}: {val}")
        return "; ".join(items)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if str(v).strip())
    return str(value)


def format_node_text(
    *,
    node_id: int,
    node_type: str,
    raw_node_name: str,
    record: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> str:
    include_fields = cfg.get("include_fields", {}) if isinstance(cfg, dict) else {}
    entity_fields = include_fields.get("entity") or ["name", "description"]
    chunk_fields = include_fields.get("chunk") or [
        "contextual_header",
        "company",
        "doc_name",
        "doc_type",
        "doc_period",
        "page",
        "section",
        "text",
        "summary",
        "semantic_tags",
    ]
    document_fields = include_fields.get("document") or [
        "doc_name",
        "company",
        "doc_type",
        "doc_period",
    ]

    parts: list[str] = []
    record = record or {}

    if node_type == "entity":
        name = record.get("name") or record.get("entity") or raw_node_name
        if name:
            parts.append(f"Entity: {name}")
        description = record.get("description") or record.get("summary")
        if description:
            parts.append(f"Description: {_stringify_value(description)}")
        else:
            for field in entity_fields:
                if field in {"name", "description"}:
                    continue
                value = record.get(field)
                if value is not None and str(value).strip():
                    parts.append(f"{field}: {_stringify_value(value)}")

    elif node_type == "chunk":
        mapping = {
            "contextual_header": "Header",
            "company": "Company",
            "doc_name": "Document",
            "doc_type": "Filing",
            "doc_period": "Period",
            "page": "Page",
            "section": "Section",
            "text": "Text",
            "summary": "Summary",
            "semantic_tags": "Tags",
        }
        for field in chunk_fields:
            value = record.get(field)
            if value is None or not str(value).strip():
                continue
            label = mapping.get(field, field)
            parts.append(f"{label}: {_stringify_value(value)}")

    elif node_type == "document":
        mapping = {
            "doc_name": "Document",
            "company": "Company",
            "doc_type": "Filing",
            "doc_period": "Period",
        }
        for field in document_fields:
            value = record.get(field)
            if value is None or not str(value).strip():
                continue
            label = mapping.get(field, field)
            parts.append(f"{label}: {_stringify_value(value)}")

    if not parts:
        if raw_node_name:
            return f"Node: {raw_node_name}"
        return f"Node: {node_id}"

    return ". ".join(parts)


def _infer_node_type(raw_node_name: str) -> str:
    if raw_node_name.startswith("chunk:"):
        return "chunk"
    if raw_node_name.startswith("entity:"):
        return "entity"
    if raw_node_name.startswith("doc:") or raw_node_name.startswith("document:"):
        return "document"
    return "unknown"


def _load_nodes_jsonl(path_value: str | Path | None) -> dict[str, dict[str, Any]]:
    raw = load_json_or_jsonl(path_value)
    lookup: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        for record in raw:
            if not isinstance(record, dict):
                continue
            key = record.get("node_id") or record.get("uid")
            if key is None:
                continue
            lookup[str(key)] = record
    return lookup


def build_node_texts(
    *,
    graph: Any,
    tensor_dir: Path,
    mappings: Any,
    cfg: dict[str, Any],
) -> tuple[list[str], dict[str, int]]:
    num_nodes = int(getattr(graph, "num_nodes"))

    id2node_path = _resolve_path(cfg.get("id2node_path"))
    if id2node_path is None:
        id2node_path = (tensor_dir / "id2node.json").resolve()
    id2node: dict[str, str] = {}
    if id2node_path.exists():
        id2node_raw = json.loads(id2node_path.read_text(encoding="utf-8"))
        id2node = {str(k): str(v) for k, v in id2node_raw.items()}
    else:
        logger.warning("id2node.json not found at %s", id2node_path)

    nodes_jsonl_path = _resolve_path(cfg.get("nodes_jsonl_path"))
    if nodes_jsonl_path is None:
        candidate = tensor_dir.parent / "graph" / "nodes.jsonl"
        nodes_jsonl_path = candidate if candidate.exists() else None
    node_meta = _load_nodes_jsonl(nodes_jsonl_path)

    chunk_lookup = build_chunk_lookup(
        chunk_text_path=cfg.get("chunk_text_path"),
        id2node=id2node,
    )

    node_type_by_id: list[str | None] = [None] * num_nodes
    nodes_by_type = getattr(graph, "nodes_by_type", None)
    if isinstance(nodes_by_type, dict):
        for type_name, node_ids in nodes_by_type.items():
            for idx in node_ids.tolist():
                if 0 <= int(idx) < num_nodes:
                    node_type_by_id[int(idx)] = str(type_name)

    node_type_tensor = getattr(graph, "node_type", None)
    node_type_names = list(getattr(graph, "node_type_names", []) or [])
    if node_type_tensor is not None and node_type_names:
        for idx, type_id in enumerate(node_type_tensor.tolist()):
            if node_type_by_id[idx] is None and 0 <= int(type_id) < len(node_type_names):
                node_type_by_id[idx] = str(node_type_names[int(type_id)])

    node_texts: list[str] = [""] * num_nodes
    stats = {
        "num_nodes": num_nodes,
        "num_entity_texts": 0,
        "num_chunk_texts": 0,
        "num_document_texts": 0,
        "num_fallback_texts": 0,
        "num_empty_texts": 0,
    }

    for node_id in range(num_nodes):
        raw_node_name = id2node.get(str(node_id), str(node_id))
        node_type = node_type_by_id[node_id] or node_meta.get(raw_node_name, {}).get("node_type")
        if not node_type:
            node_type = _infer_node_type(raw_node_name)

        record: dict[str, Any] | None = None
        if node_type == "chunk":
            chunk_id = raw_node_name
            if raw_node_name.startswith("chunk:"):
                chunk_id = raw_node_name.split("chunk:", 1)[1]
            record = (
                chunk_lookup.get(raw_node_name)
                or chunk_lookup.get(chunk_id)
                or node_meta.get(raw_node_name)
            )
        elif node_type == "entity":
            record = node_meta.get(raw_node_name)
        elif node_type == "document":
            record = node_meta.get(raw_node_name)

        text = format_node_text(
            node_id=node_id,
            node_type=str(node_type),
            raw_node_name=raw_node_name,
            record=record,
            cfg=cfg,
        )
        node_texts[node_id] = text

        if node_type == "entity":
            stats["num_entity_texts"] += 1
        elif node_type == "chunk":
            stats["num_chunk_texts"] += 1
        elif node_type == "document":
            stats["num_document_texts"] += 1

        if text.startswith("Node:"):
            stats["num_fallback_texts"] += 1
        if not text.strip():
            stats["num_empty_texts"] += 1

    return node_texts, stats


def encode_node_texts(node_texts: list[str], cfg: dict[str, Any]) -> torch.Tensor:
    model_name = str(cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"))
    device = cfg.get("embedding_device")
    batch_size = int(cfg.get("embedding_batch_size", 32))
    normalize = bool(cfg.get("normalize", True))

    embedder = BGEEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )
    embeddings = embedder.encode_documents(node_texts)
    return torch.from_numpy(embeddings).float().cpu()


def ensure_node_text_features(
    *,
    graph: Any,
    tensor_dir: Path,
    mappings: Any,
    cfg: dict[str, Any],
) -> tuple[Any, int]:
    cache_path = _resolve_path(cfg.get("cache_path"))
    if cache_path is None:
        cache_path = (tensor_dir / "node_text_features.pt").resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    force = bool(cfg.get("force_rebuild", False))
    assign_to = str(cfg.get("assign_to", "x"))

    if cache_path.exists() and not force:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(cached, dict) and isinstance(cached.get("node_features"), torch.Tensor):
            node_features = cached["node_features"]
        elif isinstance(cached, torch.Tensor):
            node_features = cached
        else:
            node_features = None

        if isinstance(node_features, torch.Tensor) and node_features.dim() == 2:
            if int(node_features.size(0)) == int(graph.num_nodes):
                logger.info("Loaded node text features from cache: %s", cache_path)
                if assign_to == "x":
                    graph.x = node_features
                elif assign_to == "node_attr":
                    graph.node_attr = node_features
                return graph, int(node_features.size(1))

        logger.warning("Cached node features invalid, rebuilding: %s", cache_path)

    node_texts, stats = build_node_texts(
        graph=graph,
        tensor_dir=tensor_dir,
        mappings=mappings,
        cfg=cfg,
    )
    logger.info("Node text stats: %s", stats)
    node_features = encode_node_texts(node_texts, cfg)

    payload = {
        "node_features": node_features,
        "model_name": str(cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")),
        "num_nodes": int(graph.num_nodes),
        "feat_dim": int(node_features.size(1)),
        "stats": stats,
    }
    torch.save(payload, cache_path)
    logger.info("Saved node text feature cache: %s", cache_path)

    if assign_to == "x":
        graph.x = node_features
    elif assign_to == "node_attr":
        graph.node_attr = node_features
    else:
        logger.warning("Unknown assign_to=%s; node features not attached", assign_to)

    return graph, int(node_features.size(1))
