# Retrieval Benchmark Summary

## Method Comparison

| Method | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- |
| BM25 | 0.348 | 0.430 | 0.261 |
| Text-only BGE | 0.064 | 0.100 | 0.056 |
| Dense BGE | 0.391 | 0.491 | 0.299 |
| Proposed | 0.473 | 0.673 | 0.395 |

## Proposed Method by Query Type

| Type | #Queries | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| text | 40 | 0.300 | 0.575 | 0.205 |
| table | 26 | 0.577 | 0.808 | 0.415 |
| image | 25 | 0.800 | 0.920 | 0.621 |
| multimodal | 19 | 0.263 | 0.368 | 0.472 |
| all | 110 | 0.473 | 0.673 | 0.395 |
