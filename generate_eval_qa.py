import json
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path

import fitz
from bs4 import BeautifulSoup


PDF_DIR = Path("data/pdfs")
RAW_DIR = Path("data/raw_filings")
QA_DIR = Path("data/qa")
TARGET_COUNTS = {
    "text": 75,
    "table": 35,
    "image": 25,
    "multimodal": 15,
}

COMPANY_BY_TICKER = {
    "AAPL": "Apple",
    "HD": "Home Depot",
    "INTU": "Intuit",
    "MS": "Morgan Stanley",
    "NVDA": "NVIDIA",
}


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def infer_ticker(pdf_path):
    return pdf_path.name.split("_", 1)[0]


def infer_form(pdf_path):
    parts = pdf_path.stem.split("_")
    if len(parts) >= 2 and parts[1] == "DEF":
        return "DEF 14A"
    if len(parts) >= 2:
        return parts[1]
    return "filing"


def answer_type(answer):
    text = str(answer)
    if re.search(r"[A-Za-z]", text) and re.search(r"\d", text) and not re.fullmatch(r"[$()%,.\d\s-]+", text):
        return "text"
    if re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", text):
        return "date"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "date"
    if "%" in text or "percent" in text.lower():
        return "percentage"
    if re.search(r"\d", text):
        return "number"
    return "text"


def unit_for(answer, evidence):
    evidence_lower = evidence.lower()
    if "%" in str(answer) or "percent" in evidence_lower:
        if "," not in str(answer) and re.fullmatch(r"\(?-?\d+(?:\.\d+)?%?\)?", str(answer).strip()):
            return "%"
    if "shares" in evidence_lower and "number of shares" not in evidence_lower:
        return "shares"
    if "$" in evidence or " in millions" in evidence_lower or "million" in evidence_lower:
        return "million"
    if "in thousands" in evidence_lower:
        return "thousand"
    if "%" in str(answer):
        return "%"
    if "date" in answer_type(answer):
        return "date"
    return ""


def make_item(qid, source_pdf, page, qtype, question, answer, evidence, note=None):
    evidence = clean_text(evidence)
    item = {
        "id": f"q{qid:03d}",
        "source_pdf": str(source_pdf).replace("\\", "/"),
        "page": page,
        "type": qtype,
        "question": clean_text(question),
        "answer": clean_text(answer),
        "unit": unit_for(answer, evidence),
        "evidence": evidence[:700],
        "answer_type": answer_type(answer),
    }
    if note:
        item["generator_note"] = note
    return item


def extract_pdf_pages(pdf_path):
    pages = []
    with fitz.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text") or ""
            pages.append({"page": idx, "text": clean_text(raw_text), "raw_text": raw_text})
    return pages


def regex_first(pattern, text):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return clean_text(match.group(1))


def first_page_text_questions(pdf_path, pages):
    ticker = infer_ticker(pdf_path)
    form = infer_form(pdf_path)
    company = COMPANY_BY_TICKER.get(ticker, ticker)
    first = pages[0]["text"] if pages else ""
    items = []

    period = regex_first(r"For the (?:quarterly )?period ended\s+([A-Z][a-z]+ \d{1,2}, \d{4})", first)
    if not period:
        period = regex_first(r"For the fiscal year ended\s+([A-Z][a-z]+ \d{1,2}, \d{4})", first)
    if period:
        items.append((
            f"What period ended is reported on the cover page of {company}'s {form} filing?",
            period,
            f"For the period ended {period}",
        ))

    report_date = regex_first(r"Date of Report .*?\):\s*([A-Za-z]+ \d{1,2}, \d{4})", first)
    if report_date:
        items.append((
            f"What is the Date of Report for {company}'s {form} filing?",
            report_date,
            f"Date of Report: {report_date}",
        ))

    commission = regex_first(r"Commission File Number[: ]+([0-9-]+)", first)
    if commission:
        items.append((
            f"What is the SEC Commission File Number shown for {company}?",
            commission,
            f"Commission File Number: {commission}",
        ))

    phone = regex_first(r"(\([0-9]{3}\) [0-9]{3}-[0-9]{4})", first)
    if phone:
        items.append((
            f"What telephone number is listed for {company} on the cover page?",
            phone,
            phone,
        ))

    exchange = regex_first(r"Common Stock.*?\b[A-Z]{1,5}\b\s+(The [A-Za-z ]+ Stock Market LLC|New York Stock Exchange)", first)
    if exchange:
        items.append((
            f"On which exchange is {company}'s common stock registered?",
            exchange,
            f"Common Stock ... {exchange}",
        ))

    if form:
        items.append((
            f"What SEC form type is the {company} filing?",
            form,
            f"FORM {form}",
        ))

    return [
        {
            "page": 1,
            "question": q,
            "answer": a,
            "evidence": e,
            "period": period,
        }
        for q, a, e in items
    ]


NUM_RE = re.compile(r"(?<![A-Za-z])-?\(?\$?\d[\d,]*(?:\.\d+)?%?\)?")


def matching_html_path(pdf_path):
    matches = list(RAW_DIR.rglob(f"{pdf_path.stem}.htm"))
    if matches:
        return matches[0]
    matches = list(RAW_DIR.rglob(f"{pdf_path.stem}.html"))
    return matches[0] if matches else None


def is_header_cell(value):
    return bool(re.search(r"\b20\d{2}\b|[A-Z][a-z]{2,} \d{1,2}, 20\d{2}|Q[1-4]", value))


def is_value_cell(value):
    return bool(re.fullmatch(r"\(?\$?-?\d[\d,]*(?:\.\d+)?%?\)?", value.strip()))


def normalize_cell(value):
    value = clean_text(value).replace("\xa0", " ")
    value = value.replace("−", "-").replace("—", "")
    return clean_text(value)


def table_context_title(table):
    for prev in table.find_all_previous(["p", "div", "span"], limit=12):
        text = normalize_cell(prev.get_text(" ", strip=True))
        if 12 <= len(text) <= 180 and re.search(r"[A-Za-z]", text):
            if text.count("$") > 2 or len(re.findall(r"\d", text)) > 24:
                continue
            if not re.fullmatch(r"\d+", text):
                return text
    return "the financial table"


def find_evidence_page(pages, label, value):
    label_terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z]+", label)[:4]]
    value_norm = re.sub(r"[,$()]", "", value)
    for page in pages:
        text = page["text"].lower()
        text_num = re.sub(r"[,$()]", "", page["text"])
        if value_norm and value_norm not in text_num:
            continue
        if label_terms and all(term in text for term in label_terms[:2]):
            return page["page"]
    return None


def extract_table_like_questions(pdf_path, pages):
    ticker = infer_ticker(pdf_path)
    company = COMPANY_BY_TICKER.get(ticker, ticker)
    candidates = []
    html_path = matching_html_path(pdf_path)
    if not html_path:
        return candidates

    soup = BeautifulSoup(html_path.read_bytes(), "html.parser")

    for table in soup.find_all("table"):
        title = table_context_title(table)
        parsed_rows = []
        for row in table.find_all("tr"):
            cells = [
                normalize_cell(cell.get_text(" ", strip=True))
                for cell in row.find_all(["th", "td"])
            ]
            cells = [cell for cell in cells if cell and cell not in {"$", "(", ")"}]
            if cells:
                parsed_rows.append(cells)

        headers = []
        for cells in parsed_rows[:6]:
            header_cells = [cell for cell in cells if is_header_cell(cell)]
            if len(header_cells) >= 2:
                headers = header_cells[-5:]
                break
        if len(headers) < 2:
            continue

        for cells in parsed_rows:
            if all(is_header_cell(cell) for cell in cells):
                continue

            label_idx = None
            for idx, cell in enumerate(cells):
                if re.search(r"[A-Za-z]", cell) and not is_header_cell(cell) and not is_value_cell(cell):
                    label_idx = idx
                    break
            if label_idx is None:
                continue

            label = cells[label_idx]
            label_lower = label.lower()
            if len(label) < 4 or len(label) > 90:
                continue
            if any(term in label_lower for term in ("table of contents", "commission file", "pursuant", "form 10-", "page ")):
                continue

            values = [cell for cell in cells[label_idx + 1:] if is_value_cell(cell)]
            if not values:
                continue
            columns = headers[-len(values):]
            if len(columns) != len(values):
                continue

            for col, value in zip(columns, values):
                value_clean = value.strip("$").strip()
                if not re.search(r"\d", value_clean):
                    continue
                if re.fullmatch(r"20\d{2}", value_clean.strip("()")):
                    continue
                page_num = find_evidence_page(pages, label, value_clean)
                if not page_num:
                    continue
                question = (
                    f"In the table described as '{title}', what value does {company} "
                    f"report for {label} under {col}?"
                )
                evidence = f"Table context: {title}. HTML table row: {label}; column: {col}; value: {value_clean}"
                candidates.append({
                    "page": page_num,
                    "question": question,
                    "answer": value_clean.strip("()"),
                    "evidence": evidence,
                    "label": label,
                    "column": col,
                })
                break

    deduped = []
    seen = set()
    for c in candidates:
        key = c["question"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def extract_visual_questions(pdf_path, pages):
    ticker = infer_ticker(pdf_path)
    company = COMPANY_BY_TICKER.get(ticker, ticker)
    form = infer_form(pdf_path)
    candidates = []

    for page in pages[:3]:
        text = page["text"]
        if "☒" in text or "☐" in text:
            if "QUARTERLY REPORT" in text and "☒ QUARTERLY REPORT" in text:
                candidates.append({
                    "page": page["page"],
                    "question": f"Which report checkbox is selected on the cover page of {company}'s {form} filing?",
                    "answer": "Quarterly report",
                    "evidence": "The selected checkbox appears next to QUARTERLY REPORT.",
                    "note": "Visual/layout QA from rendered checkbox; verify manually if using as strict image benchmark.",
                })
            elif "ANNUAL REPORT" in text and "☒ ANNUAL REPORT" in text:
                candidates.append({
                    "page": page["page"],
                    "question": f"Which report checkbox is selected on the cover page of {company}'s {form} filing?",
                    "answer": "Annual report",
                    "evidence": "The selected checkbox appears next to ANNUAL REPORT.",
                    "note": "Visual/layout QA from rendered checkbox; verify manually if using as strict image benchmark.",
                })

    for page in pages:
        text = page["text"]
        if re.search(r"\b(chart|graph|image\d+\.jpg|figure)\b", text, flags=re.IGNORECASE):
            marker = regex_first(r"\b(Image\d+\.jpg)\b", text)
            if marker:
                candidates.append({
                    "page": page["page"],
                    "question": f"Which embedded image marker appears on page {page['page']} of {company}'s filing?",
                    "answer": marker,
                    "evidence": f"Rendered page text contains embedded image marker {marker}.",
                    "note": "Image-presence QA; inspect the rendered page before reporting this as chart reasoning.",
                })

    return candidates


def build_dataset():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    qid = 1
    by_type = {key: [] for key in TARGET_COUNTS}
    first_page_meta = {}

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        pages = extract_pdf_pages(pdf_path)
        text_q = first_page_text_questions(pdf_path, pages)
        table_q = extract_table_like_questions(pdf_path, pages)
        visual_q = extract_visual_questions(pdf_path, pages)
        first_page_meta[pdf_path.name] = text_q[0].get("period") if text_q else None

        for c in text_q:
            by_type["text"].append((pdf_path, c))
        for c in table_q:
            by_type["table"].append((pdf_path, c))
        for c in visual_q:
            by_type["image"].append((pdf_path, c))

        period = first_page_meta[pdf_path.name]
        if period:
            for c in table_q[:4]:
                mm = dict(c)
                mm["question"] = (
                    f"For {COMPANY_BY_TICKER.get(infer_ticker(pdf_path), infer_ticker(pdf_path))}'s "
                    f"filing period ended {period}, what value is reported for {c['label']} "
                    f"under {c['column']}?"
                )
                mm["evidence"] = f"Cover page period: {period}. {c['evidence']}"
                by_type["multimodal"].append((pdf_path, mm))

    items = []
    for qtype, target in TARGET_COUNTS.items():
        selected = select_round_robin(by_type[qtype], target)
        for pdf_path, c in selected:
            items.append(make_item(
                qid=qid,
                source_pdf=pdf_path,
                page=c["page"],
                qtype=qtype,
                question=c["question"],
                answer=c["answer"],
                evidence=c["evidence"],
                note=c.get("note"),
            ))
            qid += 1

    return items, {key: len(value) for key, value in by_type.items()}


def select_round_robin(candidates, target):
    grouped = defaultdict(list)
    order = []
    for pdf_path, candidate in candidates:
        key = str(pdf_path)
        if key not in grouped:
            order.append(key)
        grouped[key].append((pdf_path, candidate))

    selected = []
    while len(selected) < target:
        added = False
        for key in order:
            if grouped[key]:
                selected.append(grouped[key].pop(0))
                added = True
                if len(selected) >= target:
                    break
        if not added:
            break
    return selected


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    items, available = build_dataset()
    write_jsonl(QA_DIR / "eval_qa.jsonl", items)

    dev = [row for idx, row in enumerate(items) if idx % 5 == 0]
    test = [row for idx, row in enumerate(items) if idx % 5 != 0]
    write_jsonl(QA_DIR / "dev_qa.jsonl", dev)
    write_jsonl(QA_DIR / "test_qa.jsonl", test)

    summary = {
        "total": len(items),
        "target_counts": TARGET_COUNTS,
        "actual_counts": dict(Counter(row["type"] for row in items)),
        "available_candidates": available,
        "dev_count": len(dev),
        "test_count": len(test),
        "note": (
            "Image questions are generated from visual checkboxes or embedded image markers. "
            "Manually review them before using them as strict chart/image reasoning benchmarks."
        ),
    }
    (QA_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
