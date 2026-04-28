# FinanceBench Retrieval Evaluation

Pipeline này đánh giá retrieval cho RAG tài chính bằng FinanceBench. FinanceBench không có ground-truth `chunk_id`, nên pipeline tự map `evidence_text`/`evidence_page_num` vào chunks do bạn tạo.

## Output

Theo `config.yaml`, pipeline tạo:

```text
chunks.jsonl
qrels.jsonl
unmatched_evidence.jsonl
retrieval_results.jsonl
metrics_summary.json
metrics_by_question.jsonl
retrieval_eval_report.md
```

## Cài dependency

```powershell
python -m pip install -r requirements-rag.txt
python -m pip install pyyaml tqdm
```

## Chạy nhanh

Mặc định nếu `chunking.source_chunks_path` rỗng, pipeline sẽ tạo chunks page-level từ PDF trong `benchmark/financebench/pdfs`.

```powershell
python -m rag_eval.src.run_eval --config rag_eval/config.yaml
```

## Chỉ tạo file đánh giá retrieval

Nếu bạn đã có `chunks.jsonl` từ chunking pipeline riêng và chỉ cần tạo ground truth retrieval files:

```powershell
python -m rag_eval.src.create_qrels `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --chunks outputs/financebench_eval/chunks.jsonl `
  --out-dir outputs/financebench_eval/qrels
```

## Tạo chunks bằng semantic BGE chunking hiện có

Để dùng đúng text chunking semantic BGE từ `src/chunking/rebuild_text_chunks_bge.py` cho FinanceBench PDFs:

```powershell
python -m rag_eval.src.build_financebench_bge_chunks `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --pdf-dir benchmark/financebench/pdfs `
  --output outputs/financebench_eval_bge/chunks.jsonl `
  --batch-size 32
```

Sau đó tạo qrels từ chunks semantic BGE:

```powershell
python -m rag_eval.src.create_qrels `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --chunks outputs/financebench_eval_bge/chunks.jsonl `
  --out-dir outputs/financebench_eval_bge/qrels
```

Lưu ý: nếu máy không có CUDA, bước BGE semantic chunking có thể chậm vì phải embed câu trong các PDF FinanceBench.

Output:

```text
outputs/financebench_eval/qrels/queries.jsonl
outputs/financebench_eval/qrels/qrels.jsonl
outputs/financebench_eval/qrels/unmatched_evidence.jsonl
outputs/financebench_eval/qrels/mapping_report.json
```

`qrels.jsonl` có schema chính:

```json
{
  "query_id": "financebench_id_03029",
  "chunk_id": "chunk_or_page_id",
  "relevance": 1,
  "match_method": "exact_text",
  "match_score": 1.0,
  "evidence_doc_name": "3m_2018_10k",
  "evidence_page": 60
}
```

FinanceBench lưu `evidence_page_num` dạng zero-indexed. Script mặc định chuyển sang page one-indexed để khớp với `page_start/page_end` của chunk. Nếu chunks của bạn cũng zero-indexed, chạy thêm:

```powershell
python -m rag_eval.src.create_qrels `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --chunks outputs/financebench_eval/chunks.jsonl `
  --out-dir outputs/financebench_eval/qrels `
  --evidence-page-base 1
```

## Plug-in chunking code hiện tại

Nếu bạn đã có chunk file riêng, sửa `rag_eval/config.yaml`:

```yaml
chunking:
  source_chunks_path: data/chunks/all_chunks.jsonl
  output_chunks_path: outputs/financebench_eval/chunks.jsonl
```

Chunk input nên có các field sau:

```text
id hoặc chunk_id
doc_name hoặc source_pdf
text hoặc summary
page_start
page_end
section_title nếu có
chunk_type hoặc modality
metadata nếu có
```

Adapter sẽ normalize thành schema chung và ghi ra `chunks.jsonl`.

## Qrels

`qrels.jsonl` được tạo tự động theo thứ tự:

1. `exact_text`: `evidence_text` nằm trong `chunk.text`
2. `fuzzy_text`: token overlap / sequence ratio đủ ngưỡng
3. `page_fallback`: chunk overlap với `evidence_page_num + 1`

FinanceBench dùng `evidence_page_num` zero-indexed, còn PDF/chunk dùng page one-indexed, nên pipeline tự cộng 1.

## Metrics

Pipeline tính:

```text
Precision@k
Recall@k
Hit@k
MRR@k
Page Recall@k
Evidence Recall@k
```

`Page Recall@k` kiểm tra top-k có chunk nằm đúng evidence page không.  
`Evidence Recall@k` kiểm tra top-k có chunk được match bằng `exact_text` hoặc `fuzzy_text` không.
