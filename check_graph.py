#!/usr/bin/env python
"""Check graph.x existence and dimensions."""

import torch
from pathlib import Path

tensor_dir = Path('data/graph_tensor')
graph_pt = tensor_dir / 'graph.pt'

if not graph_pt.exists():
    print(f'ERROR: {graph_pt} NOT FOUND')
    exit(1)

graph = torch.load(graph_pt, map_location='cpu', weights_only=False)
print(f'✓ Graph loaded from {graph_pt}')
print(f'  num_nodes: {graph.num_nodes}')

if graph.x is not None:
    print(f'  ✓ graph.x.shape: {graph.x.shape}')
    print(f'    graph.x.dtype: {graph.x.dtype}')
else:
    print(f'  ⚠️  graph.x is None (NO node embeddings)')

print(f'  graph.feat_dim: {getattr(graph, "feat_dim", None)}')
if hasattr(graph, 'rel_attr') and graph.rel_attr is not None:
    print(f'  graph.rel_attr.shape: {graph.rel_attr.shape}')
else:
    print(f'  graph.rel_attr: None')

print()
print("=" * 60)
if graph.x is None:
    print("ISSUE: graph.x is None - node embeddings NOT embedded")
    print("FIX: Rebuild graph with --embed-features flag")
    print("  python src/graph_extraction/build_graph_tensor.py \\")
    print("    --graph-dir data/graph \\")
    print("    --output-dir data/graph_tensor \\")
    print("    --embed-features \\")
    print("    --embedding-model BAAI/bge-base-en-v1.5")
else:
    print("✓ graph.x EXISTS - ready for distillation loss")
