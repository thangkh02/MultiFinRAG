from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image


CHUNK_DIR = Path("data/chunks")
RAW_DIR = Path("data/raw_filings")
IMAGE_ASSET_DIR = Path("data/visual_chunks/sec_images")


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


def filing_url_for_html(source_html: str) -> str | None:
    html_path = Path(source_html)
    metadata_path = html_path.with_suffix(html_path.suffix + ".json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("url")


def default_user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT") or "multifinrag-research local-contact@example.com"


def download_image(session: requests.Session, url: str, out_path: Path, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return True

    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return False
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "image" not in content_type.lower() and not url.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return True


def image_metadata(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "image_width": width,
        "image_height": height,
        "image_area": width * height,
        "image_file_size": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC image assets referenced by HTML image chunks.")
    parser.add_argument("--input", type=Path, default=CHUNK_DIR / "image_chunks.jsonl")
    parser.add_argument("--output", type=Path, default=CHUNK_DIR / "image_chunks_assets.jsonl")
    parser.add_argument("--asset-dir", type=Path, default=IMAGE_ASSET_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-current", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    session = requests.Session()
    session.headers.update({"User-Agent": default_user_agent()})

    enriched = []
    ok = 0
    skipped = 0
    failed = 0
    url_cache: dict[str, str | None] = {}

    for idx, row in enumerate(rows, start=1):
        src = row.get("image_src")
        source_html = row.get("source_html")
        if not src or not source_html:
            failed += 1
            continue

        if source_html not in url_cache:
            url_cache[source_html] = filing_url_for_html(source_html)
        filing_url = url_cache[source_html]
        if not filing_url:
            failed += 1
            continue

        image_url = urljoin(filing_url, src)
        out_path = args.asset_dir / Path(source_html).stem / Path(src).name
        new_row = dict(row)
        new_row["image_url"] = image_url
        new_row["image_path"] = str(out_path).replace("\\", "/")

        try:
            downloaded = download_image(session, image_url, out_path, overwrite=args.overwrite)
        except requests.RequestException as exc:
            new_row["image_download_error"] = str(exc)
            downloaded = False

        if downloaded:
            try:
                new_row.update(image_metadata(out_path))
            except Exception as exc:
                new_row["image_metadata_error"] = str(exc)
            ok += 1
            enriched.append(new_row)
        else:
            skipped += 1

        if idx % 50 == 0:
            print(f"[{idx}/{len(rows)}] downloaded={ok} skipped={skipped} failed={failed}")

    write_jsonl(args.output, enriched)

    if args.replace_current:
        write_jsonl(CHUNK_DIR / "image_chunks.jsonl", enriched)
        text_chunks = load_jsonl(CHUNK_DIR / "text_chunks.jsonl")
        table_chunks = load_jsonl(CHUNK_DIR / "table_chunks.jsonl")
        write_jsonl(CHUNK_DIR / "all_chunks.jsonl", text_chunks + table_chunks + enriched)

    summary = {
        "input_rows": len(rows),
        "downloaded_rows": ok,
        "skipped_rows": skipped,
        "failed_rows": failed,
        "output": str(args.output).replace("\\", "/"),
        "asset_dir": str(args.asset_dir).replace("\\", "/"),
        "replace_current": args.replace_current,
    }
    (CHUNK_DIR / "image_asset_download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
