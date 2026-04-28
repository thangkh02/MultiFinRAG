from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .io_utils import load_jsonl

LOGGER = logging.getLogger(__name__)


def load_financebench(path: Path) -> list[dict[str, Any]]:
    samples = load_jsonl(path)
    LOGGER.info("Loaded %d FinanceBench samples from %s", len(samples), path)
    return samples


def required_doc_names(samples: list[dict[str, Any]]) -> set[str]:
    doc_names: set[str] = set()
    for sample in samples:
        if sample.get("doc_name"):
            doc_names.add(str(sample["doc_name"]))
        for evidence in sample.get("evidence") or []:
            doc_name = evidence.get("evidence_doc_name") or evidence.get("doc_name")
            if doc_name:
                doc_names.add(str(doc_name))
    return doc_names
