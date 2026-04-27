from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import fitz
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTContainer, LTFigure, LTImage, LTPage


PDF_DIR = Path("data/pdfs")
RAW_DIR = Path("data/raw_filings")
CHUNK_DIR = Path("data/chunks")
VISUAL_DIR = Path("data/visual_chunks/images")


def stable_id(prefix: str, *parts: object) -> str:
    raw = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def infer_ticker(path: Path) -> str:
    return path.name.split("_", 1)[0]


def matching_html_path(pdf_path: Path) -> Path | None:
    for suffix in (".htm", ".html"):
        matches = list(RAW_DIR.rglob(f"{pdf_path.stem}{suffix}"))
        if matches:
            return matches[0]
    return None


def iter_visual_objects(layout_obj, page_number: int) -> Iterable[dict]:
    if isinstance(layout_obj, (LTImage, LTFigure)):
        x0, y0, x1, y1 = layout_obj.bbox
        yield {
            "page": page_number,
            "object_type": type(layout_obj).__name__,
            "name": getattr(layout_obj, "name", ""),
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "width": float(x1 - x0),
            "height": float(y1 - y0),
        }

    if isinstance(layout_obj, LTContainer):
        for child in layout_obj:
            yield from iter_visual_objects(child, page_number)


def detect_pdf_visuals(pdf_path: Path) -> list[dict]:
    rows: list[dict] = []
    for page_layout in extract_pages(str(pdf_path)):
        assert isinstance(page_layout, LTPage)
        rows.extend(iter_visual_objects(page_layout, page_layout.pageid))
    return rows


def pdfminer_bbox_to_fitz_rect(bbox: list[float], page_height: float, padding: float) -> fitz.Rect:
    x0, y0, x1, y1 = bbox
    rect = fitz.Rect(x0 - padding, page_height - y1 - padding, x1 + padding, page_height - y0 + padding)
    return rect


def crop_visual(pdf_doc: fitz.Document, visual: dict, out_path: Path, zoom: float, padding: float) -> bool:
    page_index = int(visual["page"]) - 1
    if page_index < 0 or page_index >= len(pdf_doc):
        return False

    page = pdf_doc[page_index]
    rect = pdfminer_bbox_to_fitz_rect(visual["bbox"], page.rect.height, padding)
    rect = rect & page.rect
    if rect.is_empty or rect.width <= 1 or rect.height <= 1:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    pix.save(str(out_path))
    return True


def is_reasonable_visual(visual: dict, min_width: float, min_height: float, min_area: float) -> bool:
    width = float(visual["width"])
    height = float(visual["height"])
    return width >= min_width and height >= min_height and width * height >= min_area


def build_image_chunks(
    pdf_paths: list[Path],
    visual_dir: Path,
    min_width: float,
    min_height: float,
    min_area: float,
    zoom: float,
    padding: float,
    limit_per_pdf: int | None,
) -> list[dict]:
    chunks: list[dict] = []
    for pdf_idx, pdf_path in enumerate(pdf_paths, start=1):
        html_path = matching_html_path(pdf_path)
        detected = [
            item
            for item in detect_pdf_visuals(pdf_path)
            if is_reasonable_visual(item, min_width=min_width, min_height=min_height, min_area=min_area)
        ]
        if limit_per_pdf is not None:
            detected = detected[:limit_per_pdf]

        with fitz.open(pdf_path) as doc:
            kept = 0
            for ordinal, visual in enumerate(detected, start=1):
                chunk_id = stable_id(
                    "image",
                    pdf_path.name,
                    visual["page"],
                    ordinal,
                    ",".join(f"{v:.2f}" for v in visual["bbox"]),
                )
                crop_path = visual_dir / pdf_path.stem / f"{chunk_id}.png"
                if not crop_visual(doc, visual, crop_path, zoom=zoom, padding=padding):
                    continue

                summary = (
                    f"{visual['object_type']} detected on page {visual['page']} of {pdf_path.name}. "
                    f"Bounding box: {', '.join(f'{v:.1f}' for v in visual['bbox'])}. "
                    "This is a visual chunk pending VLM summarization."
                )
                chunks.append(
                    {
                        "id": chunk_id,
                        "modality": "image",
                        "source_pdf": str(pdf_path).replace("\\", "/"),
                        "source_html": str(html_path or "").replace("\\", "/"),
                        "ticker": infer_ticker(pdf_path),
                        "page": visual["page"],
                        "image_index": ordinal,
                        "object_type": visual["object_type"],
                        "bbox": visual["bbox"],
                        "crop_path": str(crop_path).replace("\\", "/"),
                        "summary": summary,
                        "text": summary,
                        "chunking_method": "pdfminer_LTImage_LTFigure",
                    }
                )
                kept += 1

        print(f"[{pdf_idx}/{len(pdf_paths)}] {pdf_path.name}: detected {len(detected)} visuals -> kept {kept}")

    return chunks


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image chunks from PDF LTImage/LTFigure regions.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--visual-dir", type=Path, default=VISUAL_DIR)
    parser.add_argument("--min-width", type=float, default=80)
    parser.add_argument("--min-height", type=float, default=60)
    parser.add_argument("--min-area", type=float, default=8000)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--padding", type=float, default=4.0)
    parser.add_argument("--limit-per-pdf", type=int)
    parser.add_argument("--replace-current", action="store_true")
    args = parser.parse_args()

    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
    image_chunks = build_image_chunks(
        pdf_paths=pdf_paths,
        visual_dir=args.visual_dir,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
        zoom=args.zoom,
        padding=args.padding,
        limit_per_pdf=args.limit_per_pdf,
    )

    out_path = args.chunk_dir / "image_chunks_pdfminer.jsonl"
    write_jsonl(out_path, image_chunks)

    if args.replace_current:
        write_jsonl(args.chunk_dir / "image_chunks.jsonl", image_chunks)
        text_chunks = load_jsonl(args.chunk_dir / "text_chunks.jsonl")
        table_chunks = load_jsonl(args.chunk_dir / "table_chunks.jsonl")
        write_jsonl(args.chunk_dir / "all_chunks.jsonl", text_chunks + table_chunks + image_chunks)

    summary = {
        "pdf_count": len(pdf_paths),
        "image_chunks": len(image_chunks),
        "method": "pdfminer LTImage/LTFigure detection + PyMuPDF crop",
        "output": str(out_path).replace("\\", "/"),
        "visual_dir": str(args.visual_dir).replace("\\", "/"),
        "replace_current": args.replace_current,
    }
    summary_path = args.chunk_dir / "image_chunk_pdfminer_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
