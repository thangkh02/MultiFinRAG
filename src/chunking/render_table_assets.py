from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


CHUNK_DIR = Path("data/chunks")
TABLE_ASSET_DIR = Path("data/visual_chunks/tables")


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


def table_score(row: dict) -> int:
    text = row.get("text") or row.get("summary") or ""
    return sum(ch.isdigit() for ch in text)


def is_meaningful_table(row: dict, min_digits: int, min_words: int) -> bool:
    text = row.get("text") or row.get("summary") or ""
    return table_score(row) >= min_digits and len(text.split()) >= min_words


def render_tables(rows: list[dict], asset_dir: Path, overwrite: bool) -> list[dict]:
    enriched = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["source_html"], []).append(row)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1200}, device_scale_factor=1)

        for source_html, table_rows in grouped.items():
            html_path = Path(source_html).resolve()
            page.goto(html_path.as_uri(), wait_until="load")

            for row in table_rows:
                table_index = int(row["table_index"])
                out_dir = asset_dir / html_path.stem
                out_path = out_dir / f"{row['id']}.png"
                new_row = dict(row)
                new_row["table_image_path"] = str(out_path).replace("\\", "/")

                if out_path.exists() and not overwrite:
                    enriched.append(new_row)
                    continue

                locator = page.locator("table").nth(table_index - 1)
                if locator.count() == 0:
                    new_row["table_render_error"] = f"Missing table nth={table_index}"
                    enriched.append(new_row)
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                locator.scroll_into_view_if_needed(timeout=30_000)
                locator.screenshot(path=str(out_path), timeout=60_000)
                enriched.append(new_row)

        browser.close()

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HTML table chunks to table images.")
    parser.add_argument("--input", type=Path, default=CHUNK_DIR / "table_chunks.jsonl")
    parser.add_argument("--output", type=Path, default=CHUNK_DIR / "table_chunks_assets.jsonl")
    parser.add_argument("--asset-dir", type=Path, default=TABLE_ASSET_DIR)
    parser.add_argument("--min-digits", type=int, default=20)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = [row for row in load_jsonl(args.input) if is_meaningful_table(row, args.min_digits, args.min_words)]
    if args.limit is not None:
        rows = rows[: args.limit]

    enriched = render_tables(rows, asset_dir=args.asset_dir, overwrite=args.overwrite)
    write_jsonl(args.output, enriched)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rendered_rows": len(enriched),
                "asset_dir": str(args.asset_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
