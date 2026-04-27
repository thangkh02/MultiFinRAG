# 📊 MultiFinRAG SEC Filing RAG 
pipeline **RAG multimodal** cho các báo cáo SEC. Ý tưởng cốt lõi theo hướng **MultiFinRAG**:

* Xem **text, bảng (table), hình ảnh (image)** là các object độc lập để truy xuất
* Chuyển bảng và ảnh → **semantic text (summary)**
* Embed tất cả bằng **BGE**
* Lưu trữ và truy xuất bằng **FAISS**

Pipeline hiện tại:

* SEC HTML filings
* Convert sang PDF để xử lý text
* Table và image lấy trực tiếp từ HTML để đảm bảo độ chính xác

---

# 🧠 Tổng quan Pipeline

```text
PDF/HTML filings
|
+-- Text
|   +-- PyMuPDF extract text theo page
|   +-- tách câu (sentence split)
|   +-- sliding window + overlap
|   +-- embedding bằng BGE
|   +-- cosine distance giữa các câu
|   +-- breakpoint tại 95 percentile
|   +-- merge nếu similarity >= 0.85
|   +-- embed chunk cuối bằng BAAI/bge-base-en-v1.5
|
+-- Table
|   +-- parse HTML <table> bằng BeautifulSoup
|   +-- 1 bảng = 1 chunk
|   +-- giữ table_json từ HTML (source of truth)
|   +-- render bảng thành ảnh
|   +-- Gemma 3 27B Vision tạo semantic summary
|   +-- embed summary bằng BGE
|
+-- Image/chart
    +-- HTML <img>
    +-- tải ảnh gốc từ SEC
    +-- lọc ảnh meaningful
    +-- Gemma 3 27B Vision tạo summary
    +-- embed summary bằng BGE

FAISS
+-- text.faiss
+-- table.faiss
+-- image.faiss
+-- all.faiss (gộp tất cả)
```

---

# 📁 Cấu trúc Project

```text
src/
  common/
    bge_embedder.py
  ingestion/
    html_to_pdf.py
  chunking/
    rebuild_text_chunks_bge.py
    build_pdfminer_image_chunks.py
    render_table_assets.py
  vlm/
    nvidia_vlm_client.py
    download_sec_image_assets.py
    enrich_image_chunks_vlm.py
    enrich_table_chunks_vlm.py
  indexing/
    build_bge_index.py
    combine_bge_indexes.py
  retrieval/
    retrieve_bge.py
  evaluation/
    generate_eval_qa.py

data/
  raw_filings/
  pdfs/
  chunks/
  qa/
  index_bge/
```


# 📦 Dữ liệu chính

### Chunk files

```text
data/chunks/text_chunks.jsonl
data/chunks/table_chunks.jsonl
data/chunks/image_chunks.jsonl
data/chunks/all_chunks.jsonl
```

### Số lượng chunk

```text
text: 1318
table: 1847
image: 369
total: 3534
```

---

#  Evaluation QA

```text
eval: 150 câu
dev: 30 câu
test: 120 câu
```

Phân loại:

```text
text: 75
table: 35
image: 25
multimodal: 15
```

---

# Rebuild Pipeline

## 1. Convert HTML → PDF

```bash
python src/ingestion/html_to_pdf.py
```

---

## 2. Text Chunking

```bash
python src/chunking/rebuild_text_chunks_bge.py
```

---

## 3. Image Chunking

Download ảnh:

```bash
python src/vlm/download_sec_image_assets.py
```

Summary bằng VLM:

```bash
python src/vlm/enrich_image_chunks_vlm.py --replace-current
```

---

## 4. Table Chunking

Render bảng:

```bash
python src/chunking/render_table_assets.py
```

Summary:

```bash
python src/vlm/enrich_table_chunks_vlm.py --replace-current
```



* Số liệu bảng lấy từ HTML (không dùng VLM)
* VLM chỉ dùng để tạo semantic summary

---

## 5. Build FAISS Index

```bash
python src/indexing/build_bge_index.py --index all --batch-size 64
```

Hoặc build riêng:

```bash
python src/indexing/build_bge_index.py --index table --batch-size 64
python src/indexing/combine_bge_indexes.py
```

---

## 6. Retrieval

```bash
python src/retrieval/retrieve_bge.py "restricted cash" --index table --top-k 3

python src/retrieval/retrieve_bge.py "Daily 95% One-Day Total Management VaR" --index image --top-k 3

python src/retrieval/retrieve_bge.py "net revenue increased" --index all --top-k 5
```

---


