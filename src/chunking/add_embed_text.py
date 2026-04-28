from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHUNK_DIR = Path("data/chunks")

COMPANY_BY_TICKER = {
    "AAPL": "Apple",
    "HD": "Home Depot",
    "INTU": "Intuit",
    "MS": "Morgan Stanley",
    "NVDA": "NVIDIA",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def infer_ticker(source_pdf: str | None) -> str:
    if not source_pdf:
        return ""
    return Path(source_pdf).name.split("_", 1)[0]


def infer_form(source_pdf: str | None) -> str:
    if not source_pdf:
        return ""
    parts = Path(source_pdf).name.split("_")
    if len(parts) >= 2 and parts[1] == "DEF":
        return "DEF 14A proxy statement"
    if len(parts) >= 2:
        return parts[1]
    return ""


def company_name(source_pdf: str | None) -> str:
    ticker = infer_ticker(source_pdf)
    return COMPANY_BY_TICKER.get(ticker, ticker)


def visual_kind(text: str) -> str:
    lower = text.lower()
    if "bar chart" in lower:
        return "bar chart"
    if "line chart" in lower or "line graph" in lower or "multiple lines" in lower:
        return "line chart"
    if "pie chart" in lower:
        return "pie chart"
    if "table" in lower:
        return "table-like chart"
    if "chart" in lower:
        return "chart"
    if "logo" in lower:
        return "logo"
    return "image"


def list_text(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        return ", ".join(clean_text(value) for value in values if clean_text(value))
    return clean_text(values)


def base_context(chunk: dict[str, Any]) -> list[str]:
    source_pdf = chunk.get("source_pdf")
    page = chunk.get("page_start", chunk.get("page"))
    parts = [
        f"Company: {company_name(source_pdf)}" if company_name(source_pdf) else "",
        f"Ticker: {infer_ticker(source_pdf)}" if infer_ticker(source_pdf) else "",
        f"Filing type: {infer_form(source_pdf)}" if infer_form(source_pdf) else "",
        f"Source filing: {Path(source_pdf).name}" if source_pdf else "",
        f"Page: {page}" if page is not None else "",
        f"Modality: {chunk.get('modality')}",
    ]
    return [part for part in parts if part]


def table_embed_text(chunk: dict[str, Any]) -> str:
    parts = base_context(chunk)
    parts.append(f"Table summary: {clean_text(chunk.get('summary'))}")
    if chunk.get("table_markdown"):
        parts.append(f"Table markdown: {clean_text(chunk.get('table_markdown'))}")
    if chunk.get("table_json"):
        parts.append(f"Table JSON: {json.dumps(chunk.get('table_json'), ensure_ascii=False)}")
    return "\n".join(part for part in parts if clean_text(part))


def image_embed_text(chunk: dict[str, Any]) -> str:
    summary = clean_text(chunk.get("summary") or chunk.get("text"))
    vlm = chunk.get("vlm_output") or {}
    key_values = list_text(vlm.get("key_values"))
    evidence = clean_text(vlm.get("evidence"))
    kind = visual_kind(summary)
    image_path = chunk.get("image_path") or chunk.get("crop_path")
    image_name = Path(image_path).name if image_path else ""

    parts = base_context(chunk)
    parts.extend(
        [
            f"Visual type: {kind}",
            f"Image file: {image_name}" if image_name else "",
            f"Image path: {image_path}" if image_path else "",
            f"Visible labels and values: {key_values}" if key_values else "",
            f"Visual evidence: {evidence}" if evidence else "",
            f"Image summary: {summary}",
        ]
    )
    return "\n".join(part for part in parts if clean_text(part))


def text_embed_text(chunk: dict[str, Any]) -> str:
    parts = base_context(chunk)
    parts.append(f"Passage: {clean_text(chunk.get('text') or chunk.get('summary'))}")
    return "\n".join(part for part in parts if clean_text(part))


def enrich_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    row = dict(chunk)
    modality = row.get("modality")
    if modality == "table":
        row["embed_text"] = table_embed_text(row)
    elif modality == "image":
        row["embed_text"] = image_embed_text(row)
    else:
        row["embed_text"] = text_embed_text(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Add retrieval-focused embed_text to chunk JSONL files.")
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    args = parser.parse_args()

    summary = {}
    all_rows: list[dict[str, Any]] = []
    for name in ("text", "table", "image"):
        path = args.chunk_dir / f"{name}_chunks.jsonl"
        rows = [enrich_chunk(row) for row in load_jsonl(path)]
        write_jsonl(path, rows)
        all_rows.extend(rows)
        summary[name] = len(rows)

    write_jsonl(args.chunk_dir / "all_chunks.jsonl", all_rows)
    summary["all"] = len(all_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
