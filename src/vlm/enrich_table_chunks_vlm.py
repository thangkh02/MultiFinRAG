from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vlm.nvidia_vlm_client import TABLE_PROMPT, call_gemma_vision


CHUNK_DIR = Path("data/chunks")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_jsonish(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"summary": stripped, "table_json": None, "evidence": "Raw VLM response was not valid JSON."}


def table_context_prompt(row: dict) -> str:
    return (
        "You are summarizing a financial table from an SEC filing.\n"
        "Return JSON only with these keys:\n"
        "- summary: one precise sentence describing what the table reports\n"
        "- evidence: short note naming visible title/header/page context\n"
        "- table_json: null\n\n"
        "Rules:\n"
        "- Use the parsed table below as the reliable source of row labels, column labels, values, and units.\n"
        "- Do not reproduce the full table_json in your output.\n"
        "- Do not invent missing values.\n"
        "- Do not calculate totals, averages, changes, or derived metrics unless they appear as explicit table cells.\n"
        "- Prefer a summary naming the metric, period/date columns, units, and main row group.\n"
        + "\n\nParsed table from SEC HTML, use this as the reliable source of cell values:\n"
        + json.dumps(row.get("table_json"), ensure_ascii=False)[:12000]
        + "\n\nExisting deterministic summary:\n"
        + str(row.get("summary") or "")[:3000]
    )


def summarize_with_retries(row: dict, image_path: Path, max_image_side: int, retries: int, timeout: int) -> str:
    last_error: Exception | None = None
    side = max_image_side
    sides = []
    while side >= 384:
        sides.append(side)
        side = int(side * 0.75)

    for current_side in sides:
        for attempt in range(1, retries + 1):
            try:
                return call_gemma_vision(
                    table_context_prompt(row),
                    image_path,
                    stream=False,
                    max_tokens=900,
                    max_image_side=current_side,
                    timeout=timeout,
                )
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if "maximum context length" in str(exc):
                    print(f"  table image side {current_side}px too large; trying smaller")
                    break
                if "DEGRADED function cannot be invoked" in str(exc):
                    raise RuntimeError(exc) from exc
                wait = min(30, 2**attempt)
                print(f"  attempt {attempt}/{retries} at {current_side}px failed: {exc}. retrying in {wait}s")
                time.sleep(wait)

    raise RuntimeError(f"VLM failed after {retries} attempts for {image_path}: {last_error}")


def fallback_output(row: dict, error: Exception) -> dict:
    return {
        "summary": row.get("summary") or "",
        "table_json": None,
        "evidence": "Fallback to deterministic HTML table summary because VLM request failed.",
        "error": str(error),
    }


def merge_vlm_into_all_tables(base_rows: list[dict], vlm_rows: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in vlm_rows}
    merged = []
    for row in base_rows:
        replacement = by_id.get(row["id"])
        merged.append(replacement if replacement else row)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Use Gemma Vision to summarize rendered table images.")
    parser.add_argument("--input", type=Path, default=CHUNK_DIR / "table_chunks_assets.jsonl")
    parser.add_argument("--base", type=Path, default=CHUNK_DIR / "table_chunks.jsonl")
    parser.add_argument("--output", type=Path, default=CHUNK_DIR / "table_chunks_vlm.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--replace-current", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is not set. Put it in .env or the environment.")

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    done_ids = set()
    if args.output.exists() and not args.no_resume:
        done = load_jsonl(args.output)
        done_ids = {row.get("id") for row in done}
        rows = [row for row in rows if row.get("id") not in done_ids]
        print(f"Resume enabled: {len(done_ids)} existing rows, {len(rows)} remaining rows")
    elif args.output.exists():
        args.output.unlink()

    new_rows = []
    for idx, row in enumerate(rows, start=1):
        image_path = Path(row["table_image_path"])
        print(f"[{idx}/{len(rows)}] VLM table summary: {image_path}")
        try:
            raw = summarize_with_retries(
                row=row,
                image_path=image_path,
                max_image_side=args.max_image_side,
                retries=args.retries,
                timeout=args.timeout,
            )
            parsed = parse_jsonish(raw)
            vlm_status = "ok"
        except Exception as exc:
            print(f"  VLM failed permanently, using HTML fallback: {exc}")
            parsed = fallback_output(row, exc)
            vlm_status = "fallback"
        summary = parsed.get("summary") or row.get("summary") or ""

        new_row = dict(row)
        new_row["rule_summary"] = row.get("summary")
        new_row["vlm_summary"] = summary
        new_row["summary"] = summary
        new_row["text"] = summary
        new_row["vlm_output"] = parsed
        new_row["vlm_status"] = vlm_status
        new_row["vlm_table_json"] = parsed.get("table_json")
        new_row["table_json_source"] = "html_parse"
        new_row["chunking_method"] = "html_table_object_gemma3_vision_summary"
        append_jsonl(args.output, new_row)
        new_rows.append(new_row)
        time.sleep(args.sleep)

    if args.replace_current:
        base_rows = load_jsonl(args.base)
        vlm_rows = load_jsonl(args.output)
        final_rows = merge_vlm_into_all_tables(base_rows, vlm_rows)
        write_jsonl(CHUNK_DIR / "table_chunks.jsonl", final_rows)
        text_chunks = load_jsonl(CHUNK_DIR / "text_chunks.jsonl")
        image_chunks = load_jsonl(CHUNK_DIR / "image_chunks.jsonl")
        write_jsonl(CHUNK_DIR / "all_chunks.jsonl", text_chunks + final_rows + image_chunks)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "new_rows": len(new_rows),
                "total_rows": len(load_jsonl(args.output)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
