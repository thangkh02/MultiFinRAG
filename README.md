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

## 7. FinanceBench Retrieval Evaluation

FinanceBench khong co ground-truth `chunk_id`. File benchmark chi co `question`, `answer`, `evidence_text`, `evidence_page_num`, `evidence_doc_name/doc_name` va PDF goc. Vi vay evaluation script se tu map evidence vao chunk do pipeline cua minh tao ra:

1. exact match `evidence_text` trong `chunk.text`
2. fuzzy/token-overlap match neu exact match khong duoc
3. fallback theo page neu van khong match duoc

Chay sau khi da chunk va build FAISS index cho PDF FinanceBench:

```bash
python src/evaluation/financebench_retrieval_eval.py ^
  --benchmark benchmark/financebench/data/financebench_open_source.jsonl ^
  --chunks data/chunks/all_chunks.jsonl ^
  --index data/index_bge/all.faiss ^
  --ids data/index_bge/all_chunk_ids.json ^
  --k-values 1,3,5,10 ^
  --out-dir data/eval/financebench_retrieval
```

Output:

```text
data/eval/financebench_retrieval/qrels.jsonl
data/eval/financebench_retrieval/qrel_misses.jsonl
data/eval/financebench_retrieval/runs.jsonl
data/eval/financebench_retrieval/per_question_metrics.jsonl
data/eval/financebench_retrieval/aggregate_metrics.csv
data/eval/financebench_retrieval/report.md
```

Metrics gom `Precision@k`, `Recall@k`, `Hit@k`, `MRR@k`, `Page Recall@k`, va `Evidence Recall@k`.

---

## 8. FinanceBench Semantic BGE Retrieval Result

Da tao benchmark retrieval tu FinanceBench open-source bang semantic text chunking BGE.

Input:

```text
benchmark/financebench/data/financebench_open_source.jsonl
benchmark/financebench/pdfs/
```

Chunking:

```text
PDF -> sentence split -> BGE sentence embedding -> semantic breakpoints -> text chunks
```

Do may hien tai khong co CUDA, da chay CPU voi:

```text
BAAI/bge-small-en-v1.5
```

Output chunks:

```text
outputs/financebench_eval_bge/chunks.jsonl
```

Chay semantic chunking:

```powershell
python -m rag_eval.src.build_financebench_bge_chunks `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --pdf-dir benchmark/financebench/pdfs `
  --output outputs/financebench_eval_bge/chunks.jsonl `
  --model BAAI/bge-small-en-v1.5 `
  --batch-size 32
```

Tao retrieval benchmark tu chunks:

```powershell
python -m rag_eval.src.create_qrels `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --chunks outputs/financebench_eval_bge/chunks.jsonl `
  --out-dir outputs/financebench_eval_bge/qrels
```

File danh gia duoc tao:

```text
outputs/financebench_eval_bge/qrels/queries.jsonl
outputs/financebench_eval_bge/qrels/qrels.jsonl
outputs/financebench_eval_bge/qrels/unmatched_evidence.jsonl
outputs/financebench_eval_bge/qrels/mapping_report.json
```

Thong ke chunks/qrels:

```text
semantic chunks:      9927
docs_with_chunks:     84
questions:            150
evidence_rows:        189
qrels raw:            258
qrels active:         248
unmatched_evidence:   0
match_method_counts:  fuzzy_text=182, exact_text=66
```

Da audit va loc bot qrels dang ngo:

```text
Tieu chi loc: fuzzy_text co token_recall < 0.60
So qrels bi loai: 10
So query bi anh huong: 7
So query con lai trong benchmark: 150/150
remaining low-token fuzzy qrels: 0
```

Khong loai tat ca `page_match=False`, vi semantic chunk co the gom nhieu page hoac page index cua parser PDF lech voi FinanceBench. Neu `token_recall` van du cao thi qrel duoc giu lai.

File qrels:

```text
outputs/financebench_eval_bge/qrels/qrels.jsonl                  # active, da loc
outputs/financebench_eval_bge/qrels/qrels_unfiltered.jsonl       # backup ban raw
outputs/financebench_eval_bge/qrels/removed_qrels_suspicious.jsonl
outputs/financebench_eval_bge/qrels/mapping_report.json
```

Phan bo query -> relevant chunks:

```text
1 chunk relevant : 103 queries
2 chunks relevant: 30 queries
3 chunks relevant: 11 queries
4 chunks relevant: 5 queries
5 chunks relevant: 1 query
```

Evaluation output:

```text
outputs/financebench_eval_bge/retrieval_eval/
```

Chay retrieval evaluation trong notebook:

```text
notebooks/financebench_current_retrieval_hybrid_eval.ipynb
notebooks/financebench_fast_hybrid_retrieval_eval.ipynb
```

Audit qua trinh tao benchmark:

```text
notebooks/financebench_retrieval_benchmark_audit.ipynb
```

File ket qua evaluation:

```text
outputs/financebench_eval_bge/retrieval_eval_filtered/bm25_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/bge_dense_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/hybrid_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/metadata_aware_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/metrics_summary.json
outputs/financebench_eval_bge/retrieval_eval_filtered/metrics_by_question.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/retrieval_eval_report.md
```

Chay lai evaluation filtered:

```powershell
python -m rag_eval.src.run_financebench_filtered_retrieval_eval `
  --base-dir outputs/financebench_eval_bge `
  --out-dir outputs/financebench_eval_bge/retrieval_eval_filtered `
  --top-k 10 `
  --candidate-k 50 `
  --hybrid-alpha 0.60 `
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --reranker-batch-size 8
```

Ket qua retrieval:

| Method | Hit@1 | Recall@1 | MRR@1 | Hit@10 | Recall@10 | MRR@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge_dense | 0.1067 | 0.0650 | 0.1067 | 0.3400 | 0.2694 | 0.1695 | 0.0387 |
| bm25 | 0.0600 | 0.0500 | 0.0600 | 0.1733 | 0.1539 | 0.0947 | 0.0213 |
| bm25_bge_hybrid_alpha_0.60 | 0.1000 | 0.0706 | 0.1000 | 0.3267 | 0.2639 | 0.1563 | 0.0387 |
| bm25_bge_hybrid_alpha_0.60_reranker_cross_encoder_ms_marco_minilm_l_6_v2 | 0.1067 | 0.0722 | 0.1067 | 0.3133 | 0.2522 | 0.1605 | 0.0373 |
| metadata_doc_filter_bm25_bge_hybrid_alpha_0.60 | 0.2067 | 0.1617 | 0.2067 | 0.5333 | 0.4596 | 0.3018 | 0.0680 |

Nhan xet:

```text
bge_dense dang tot nhat trong open-corpus retrieval theo Hit@10/Recall@10.
BM25 thuan thap hon dense.
Hybrid alpha=0.60 gan bang dense nhung chua vuot dense.
Reranker nhe cross-encoder cai thien Recall@1 so voi dense/hybrid, nhung giam Hit@10/Recall@10 vi rerank lai top-50.
metadata_doc_filter la cau hinh oracle biet truoc doc_name, khong so sanh truc tiep voi open-corpus retrieval.
FinanceBench BGE chunks hien tai chi la text chunks, nen chua co table-aware retrieval rieng.
```

File metric chinh:

```text
outputs/financebench_eval_bge/retrieval_eval_filtered/metrics_summary.json
```


