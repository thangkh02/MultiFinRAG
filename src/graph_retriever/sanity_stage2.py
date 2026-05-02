"""
Sanity test cho Stage 2 GFM-RAG pipeline.

Quy trình:
  1. Load 20 QA samples đầu tiên từ dev_qa_stage2.json
  2. Overfit model trên đúng 20 samples đó trong 300 steps (lr=1e-4, batch=4)
  3. Eval trên chính 20 samples → chunk_mrr và entity_mrr
  4. PASS nếu chunk_mrr > 0.1 và tăng so với step 0
  5. FAIL → in debug info (top-5 predicted chunk vs expected) và raise AssertionError

Chạy:
  python src/graph_retriever/sanity_stage2.py \
      --config configs/graph_retriever/stage2_sft.yaml \
      --sanity-steps 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

SANITY_N = 20       # Số samples để overfit
SANITY_STEPS = 300  # Steps tối đa


def _compute_mrr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MRR trên (B, N) logit vs binary target."""
    if target.sum() == 0:
        return 0.0
    ranks = []
    for i in range(pred.size(0)):
        pos_ids = target[i].bool().nonzero(as_tuple=False).squeeze(1)
        if len(pos_ids) == 0:
            continue
        sorted_ids = torch.argsort(pred[i], descending=True)
        for pos_id in pos_ids:
            rank = (sorted_ids == pos_id).nonzero(as_tuple=False).item() + 1
            ranks.append(1.0 / rank)
    return float(sum(ranks) / len(ranks)) if ranks else 0.0


def run_sanity(cfg: dict, sanity_steps: int = SANITY_STEPS) -> None:
    """Chạy sanity test. Raise AssertionError nếu fail."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    from src.graph_retriever.graph_adapter import build_target_to_other_types, load_graph_bundle
    from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm
    from src.graph_retriever.rel_features import ensure_rel_attr
    from src.graph_retriever.stage2_dataset import Stage2TorchDataset

    import json

    bootstrap_gfmrag(cfg.get("gfmrag_path"))
    if bool(cfg.get("disable_custom_rspmm", True)):
        disable_custom_rspmm()

    device_spec = str(cfg["training"].get("device", "cuda"))
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    logger.info("Sanity device: %s", device)

    tensor_dir = Path(str(cfg["graph"]["tensor_dir"]))

    # Load graph
    bundle = load_graph_bundle(tensor_dir, mention_relation_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")))
    graph = bundle.data

    graph, feat_dim = ensure_rel_attr(
        graph,
        rel2id_path=tensor_dir / "rel2id.json",
        embedding_model=str(cfg["graph"]["relation_embedding_model"]),
        embedding_device=cfg["graph"].get("relation_embedding_device"),
        embedding_batch_size=int(cfg["graph"].get("relation_embedding_batch_size", 32)),
        force=False,
    )

    target_to_other = build_target_to_other_types(
        graph, bundle.mappings.relation2id,
        mention_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    assert target_to_other, "target_to_other_types rỗng — kiểm tra is_mentioned_in edges"
    graph.target_to_other_types = target_to_other
    graph = graph.to(device)

    # Load node mappings
    node2id: dict[str, int] = {str(k): int(v) for k, v in bundle.mappings.entity2id.items()}
    id2node_raw: dict[str, str] = json.loads((tensor_dir / "id2node.json").read_text(encoding="utf-8"))

    # Load 20 samples đầu
    data_cfg = cfg["data"]
    stage2_json = Path(str(data_cfg["stage2_json"]))
    with open(stage2_json, encoding="utf-8") as f:
        all_samples = json.load(f)

    sanity_samples = all_samples[:SANITY_N]
    logger.info("Sanity: lấy %d samples đầu tiên từ %s", len(sanity_samples), stage2_json)

    # Dùng đúng embedding của 20 sample đầu. Không gọi load_stage2_sft_data ở đây
    # vì hàm đó shuffle train split, dễ làm lệch sample ↔ embedding trong sanity.
    from torch.utils.data import DataLoader

    emb_cache = Path(str(data_cfg["emb_cache"])) if data_cfg.get("emb_cache") else None
    if emb_cache is not None and emb_cache.exists():
        all_question_embs = torch.load(emb_cache, map_location="cpu", weights_only=False)
        question_embs = all_question_embs[: len(sanity_samples)].float()
        logger.info("Load %d sanity question embeddings từ cache: %s", len(question_embs), emb_cache)
    else:
        from sentence_transformers import SentenceTransformer

        text_model_name = str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5"))
        logger.info("Encode %d sanity questions bằng %s", len(sanity_samples), text_model_name)
        text_model = SentenceTransformer(text_model_name, device=data_cfg.get("emb_device"))
        embs = text_model.encode(
            [s["question"] for s in sanity_samples],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        question_embs = torch.tensor(embs, dtype=torch.float32)

    tiny_ds = Stage2TorchDataset(sanity_samples, question_embs, int(graph.num_nodes), node2id)
    sanity_batch_size = int(cfg["training"].get("train_batch_size", 1))
    sanity_batch_size = max(1, min(sanity_batch_size, len(tiny_ds)))
    tiny_loader = DataLoader(tiny_ds, batch_size=sanity_batch_size, shuffle=False)
    logger.info("Sanity batch_size=%d", sanity_batch_size)

    # Khởi tạo model
    from gfmrag.models.gfm_rag_v1 import model as gfm_model_module
    from gfmrag.models.gfm_rag_v1.rankers import SimpleRanker
    from gfmrag.losses import BCELoss, ListCELoss

    entity_model = instantiate(OmegaConf.create(cfg["model"]["entity_model"]))
    ranker = SimpleRanker()
    model_cfg = cfg["model"]
    GNNRetriever = gfm_model_module.GNNRetriever
    model = GNNRetriever(
        entity_model=entity_model,
        feat_dim=int(feat_dim),
        ranker=ranker,
        init_nodes_weight=bool(model_cfg.get("init_nodes_weight", True)),
        init_nodes_type=str(model_cfg.get("init_nodes_type", "chunk")),
    ).to(device)

    # Load pretrained
    pretrained_path = cfg["training"].get("pretrained_model_path")
    if pretrained_path:
        pre_path = Path(str(pretrained_path))
        if not pre_path.is_absolute():
            pre_path = (_REPO_ROOT / pre_path).resolve()
        if pre_path.exists():
            payload = torch.load(pre_path, map_location="cpu", weights_only=False)
            state = payload.get("model") if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info("Pretrained load OK: missing=%d, unexpected=%d", len(missing), len(unexpected))
            del payload, state
            if device.type == "cuda":
                torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bce_fn = BCELoss()
    listce_fn = ListCELoss()

    chunk_nodes = graph.nodes_by_type.get("chunk", torch.tensor([], dtype=torch.long))
    entity_nodes = graph.nodes_by_type.get("entity", torch.tensor([], dtype=torch.long))

    # ─── Eval tại step 0 ──────────────────────────────────────
    def eval_model() -> dict[str, float]:
        model.eval()
        all_chunk_pred, all_chunk_tgt = [], []
        all_ent_pred, all_ent_tgt = [], []
        with torch.no_grad():
            for batch in tiny_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(graph, batch)
                tgt = batch["target_nodes_mask"]
                all_chunk_pred.append(pred[:, chunk_nodes].cpu())
                all_chunk_tgt.append(tgt[:, chunk_nodes].cpu())
                all_ent_pred.append(pred[:, entity_nodes].cpu())
                all_ent_tgt.append(tgt[:, entity_nodes].cpu())
        chunk_mrr = _compute_mrr(torch.cat(all_chunk_pred), torch.cat(all_chunk_tgt))
        ent_mrr = _compute_mrr(torch.cat(all_ent_pred), torch.cat(all_ent_tgt))
        return {"chunk_mrr": chunk_mrr, "entity_mrr": ent_mrr}

    logger.info("=== SANITY STEP 0 EVAL ===")
    step0_metrics = eval_model()
    logger.info("  chunk_mrr=%.4f | entity_mrr=%.4f", step0_metrics["chunk_mrr"], step0_metrics["entity_mrr"])
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ─── Overfit loop ─────────────────────────────────────────
    model.train()
    step = 0
    epoch = 0
    loss_history: list[float] = []
    skipped_empty_entity_batches = 0
    # GNNRetriever.map_entities_to_docs uses torch.sparse.mm. CUDA sparse addmm
    # does not support float16 on this PyTorch build, so keep sanity in fp32.
    dtype_name = "float32"
    amp_dtype = torch.float32
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    logger.info("Sanity AMP: enabled=%s dtype=%s", use_amp, amp_dtype if use_amp else "float32")

    while step < sanity_steps:
        epoch += 1
        for batch in tiny_loader:
            if step >= sanity_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(graph, batch)
                tgt = batch["target_nodes_mask"]

                # Loss chỉ trên entity nodes (bám theo official config)
                ent_pred = pred[:, entity_nodes]
                ent_tgt = tgt[:, entity_nodes]
                if ent_tgt.sum() == 0:
                    loss = ent_pred.sum() * 0.0
                else:
                    loss = 0.3 * bce_fn(ent_pred, ent_tgt) + 0.7 * listce_fn(ent_pred, ent_tgt)
            if loss.item() == 0.0 and ent_tgt.sum().item() == 0:
                skipped_empty_entity_batches += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_history.append(loss.item())
            step += 1

            if step % 50 == 0:
                avg_loss = sum(loss_history[-50:]) / len(loss_history[-50:])
                logger.info("  step=%d | avg_loss=%.4f", step, avg_loss)

    # ─── Final eval ───────────────────────────────────────────
    logger.info("=== SANITY FINAL EVAL (step %d) ===", step)
    if skipped_empty_entity_batches:
        logger.info("Skipped %d empty-entity target batches in sanity loss", skipped_empty_entity_batches)
    final_metrics = eval_model()
    logger.info("  chunk_mrr=%.4f | entity_mrr=%.4f", final_metrics["chunk_mrr"], final_metrics["entity_mrr"])

    # ─── Debug: top-5 predicted chunks vs expected ────────────
    logger.info("=== TOP-5 DEBUG (3 samples) ===")
    model.eval()
    debug_loader = DataLoader(tiny_ds, batch_size=1, shuffle=False)
    for debug_idx, batch in enumerate(debug_loader):
        if debug_idx >= 3:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            pred = model(graph, batch)
        tgt = batch["target_nodes_mask"][0]
        chunk_scores = pred[0, chunk_nodes].cpu()
        top5_local = torch.topk(chunk_scores, k=min(5, len(chunk_nodes))).indices
        top5_global = chunk_nodes[top5_local]
        expected_global = (tgt.bool()).nonzero(as_tuple=False).squeeze(1)
        expected_chunk = [g.item() for g in expected_global if g.item() in chunk_nodes.tolist()]

        def node_name(idx: int) -> str:
            return id2node_raw.get(str(idx), f"node_{idx}")

        q = sanity_samples[debug_idx]["question"]
        logger.info("  [%d] Q: %s", debug_idx, q[:80])
        logger.info("    Expected chunks: %s", [node_name(i) for i in expected_chunk[:3]])
        logger.info("    Top-5 predicted: %s", [node_name(i.item()) for i in top5_global])

    # ─── Pass/Fail ────────────────────────────────────────────
    chunk_mrr_0 = step0_metrics["chunk_mrr"]
    chunk_mrr_f = final_metrics["chunk_mrr"]

    logger.info("=== SANITY RESULT ===")
    logger.info("  chunk_mrr: %.4f → %.4f (threshold: 0.1)", chunk_mrr_0, chunk_mrr_f)

    if chunk_mrr_f < 0.1:
        raise AssertionError(
            f"SANITY FAIL: chunk_mrr={chunk_mrr_f:.4f} < 0.1 sau {sanity_steps} steps. "
            "Kiểm tra: data mapping, model loading, loss objective, target_to_other_types."
        )

    if chunk_mrr_f <= chunk_mrr_0:
        raise AssertionError(
            f"SANITY FAIL: chunk_mrr không tăng ({chunk_mrr_0:.4f} → {chunk_mrr_f:.4f}). "
            "Model không học được — kiểm tra gradient flow và loss."
        )

    logger.info("SANITY PASS: chunk_mrr %.4f > 0.1 và tăng từ %.4f", chunk_mrr_f, chunk_mrr_0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity test Stage 2 GFM-RAG.")
    p.add_argument("--config", type=Path, default=Path("configs/graph_retriever/stage2_sft.yaml"))
    p.add_argument("--sanity-steps", type=int, default=SANITY_STEPS)
    p.add_argument("--gfmrag-path", type=str, default=None)
    p.add_argument("--disable-custom-rspmm", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ns = parse_args(argv)
    from omegaconf import OmegaConf
    from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm

    raw = OmegaConf.load(ns.config)
    bootstrap_gfmrag(ns.gfmrag_path or raw.get("gfmrag_path"))
    if ns.disable_custom_rspmm or bool(raw.get("disable_custom_rspmm", True)):
        disable_custom_rspmm()

    cfg_all = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg_all, dict)

    run_sanity(cfg_all, sanity_steps=ns.sanity_steps)


if __name__ == "__main__":
    main()
