# Graph retriever — KGC pretraining cục bộ

Module này tách khỏi pipeline RAG/FAISS. Nó tái hiện **negative sampling / ranking** như trong `tasks.py` và **`KGCTrainer`** (BCE + trọng số adversarial) nhưng dùng **`torch_geometric.data.Data`** đã được sinh từ `data/graph_tensor` của project.

## Chuẩn bị graph

1. Chuỗi chuẩn của repo (xem README gốc): `extract_chunk_graph` → `build_graph_nodes_edges.py` sinh CSV trong `data/graph/`.
2. Chuyển sang tensor PyG:

```bash
python src/graph_extraction/build_graph_tensor.py --graph-dir data/graph --output-dir data/graph_tensor
```

Kết quả cần thiết: `graph.pt`, `node2id.json`, `id2node.json`, `rel2id.json`. Inverse edges và `edge_type` mở rộng đã có sẵn trong `graph.pt` — đủ điều kiện cho `strict_negative_mask`.

## Huấn luyện (KGC)

Cấu hình mặc định: `configs/graph_retriever/kgc_training.yaml` (đường dẫn tương đối từ **thư mục gốc project**).

```bash
python -m src.graph_retriever.train_kgc --config configs/graph_retriever/kgc_training.yaml
```

Ghi đè nhanh:

```bash
python -m src.graph_retriever.train_kgc --epochs 5 --tensor-dir data/graph_tensor --output-dir outputs/my_kgc --fast-eval-queries 500
```

- **`num_negative`**: mặc định 128 trong YAML (đổi thành 256 nếu cần).
- **`strict_negative`**: `true` mặc định (không lấy true tail/head làm negative).
- **`graph.build_relation_graph`**: chỉ bật nếu một mô hình sau này yêu cầu relation graph (**DistMult không cần**).

Checkpoint: `{output_dir}/kgc_checkpoint.pt` (embedding DistMult).

## Inference

```bash
python -m src.graph_retriever.inference ^
  --tensor-dir data/graph_tensor ^
  --checkpoint outputs/graph_retriever/kgc_pretrained/kgc_checkpoint.pt ^
  --relation is_mentioned_in ^
  --head entity:YOUR_ENTITY_HASH ^
  --direction tail ^
  --top-k 10 ^
  --candidates entity_only
```

- **`--direction tail`**: cho (head, relation), xếp hạng mọi ứng viên làm đuôi.
- **`--direction head`**: cho (tail, relation), đặt **`--tail chunk:...`** (hoặc entity), bỏ `--head`.
- **`--candidates`**: `all` hoặc `entity_only` (lọc theo `nodes_by_type["entity"]` trên graph).

Đầu ra JSON:

- **`query`**: các trường đã giải mã relation/ids.
- **`top_entities`**: `rank`, `node_id`, `uid`, `score`, optional `linked_documents` (ánh xạ từ cạnh `is_mentioned_in`).
- **`top_documents`**: gộp theo điểm tốt nhất từ các entity trong `top_entities` (độ dài có thể lớn hơn `--top-k` khi một entity có nhiều chunk).

## Tiện ích lập trình

- **`load_graph_bundle(Path)`**: trả **`GraphBundle(data, mappings)`**, gồm `entity_to_documents`.
- **`build_relation_graph`**: trong `tasks.py`, bám file tác giả — gọi khi `graph.build_relation_graph: true`.

## Tiêm mô hình GFM khác

Module hiện huấn luyện **DistMult** (forward `(graph, batch)` giống bước trong `KGCTrainer`). Để dùng NBF/GNN của gfm-rag, giữ adapter + `tasks.py`; thay lớp mô hình và vòng huấn luyện theo repo upstream vẫn tương thích với **`Data`** đã có.
