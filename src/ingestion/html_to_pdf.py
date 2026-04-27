from pathlib import Path

from playwright.sync_api import sync_playwright


RAW_DIR = Path("data/raw_filings")
PDF_DIR = Path("data/pdfs")


def convert_html_to_pdf(html_path: Path) -> None:
    out_path = PDF_DIR / f"{html_path.stem}.pdf"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.pdf(
            path=str(out_path),
            format="A4",
            print_background=True,
            margin={
                "top": "10mm",
                "right": "10mm",
                "bottom": "10mm",
                "left": "10mm",
            },
        )
        browser.close()

    print("PDF saved:", out_path)


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    html_files = [
        file
        for file in RAW_DIR.rglob("*")
        if file.suffix.lower() in {".htm", ".html"}
    ]

    if not html_files:
        print(f"No HTML files found in {RAW_DIR}")
        return

    for file in html_files:
        convert_html_to_pdf(file)


if __name__ == "__main__":
    main()
