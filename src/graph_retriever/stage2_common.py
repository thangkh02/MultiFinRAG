from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def _resolve_existing_or_parent(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()
    return path.parent.resolve() / path.name


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def reject_external_gfmrag_path(gfmrag_path: str | None) -> None:
    """Reject runtime gfmrag paths outside this project."""
    candidates: list[tuple[str, str]] = []
    if gfmrag_path:
        candidates.append(("gfmrag_path", str(gfmrag_path)))
    env_path = os.environ.get("GFM_RAG_PATH")
    if env_path:
        candidates.append(("GFM_RAG_PATH", env_path))

    for source, raw_path in candidates:
        resolved = _resolve_existing_or_parent(raw_path)
        if not _is_relative_to(resolved, _REPO_ROOT):
            raise ValueError(
                f"{source} points outside project root: {resolved}. "
                "Stage 2 must use the current project's gfmrag package only; "
                "the author's clone is reference-only."
            )


def gfmrag_source_path() -> str:
    module = sys.modules.get("gfmrag")
    if module is None:
        try:
            import gfmrag  # noqa:F401

            module = sys.modules.get("gfmrag")
        except Exception as exc:  # pragma: no cover - best effort logging
            return f"unavailable ({exc})"
    if module is None:
        return "unavailable"
    file_path = getattr(module, "__file__", None)
    if file_path:
        return str(Path(file_path).resolve())
    paths = getattr(module, "__path__", None)
    if paths:
        return str(Path(next(iter(paths))).resolve())
    return "unknown"


def log_stage2_runtime_context(config_path: Path) -> None:
    logger.info("=== STAGE2 RUNTIME CONTEXT ===")
    logger.info("  current working directory: %s", Path.cwd())
    logger.info("  project root: %s", _REPO_ROOT)
    logger.info("  gfmrag import source path: %s", gfmrag_source_path())
    logger.info("  config path đang dùng: %s", config_path.resolve())


def validate_graph_x(
    graph: Any,
    *,
    question_embeddings: torch.Tensor | None = None,
    feat_dim: int | None = None,
    context: str = "Stage 2",
) -> int:
    """Validate graph.x for distillation and return its feature dimension."""
    x = getattr(graph, "x", None)
    if x is None:
        raise ValueError(
            f"{context}: graph.x is missing. Run:\n"
            "  python src/graph_retriever/prepare_stage2_node_features.py "
            "--config configs/graph_retriever/stage2_sft.yaml --force"
        )
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{context}: graph.x must be a torch.Tensor, got {type(x)}")
    if x.dim() != 2:
        raise ValueError(f"{context}: graph.x must be 2D [num_nodes, feat_dim], got {tuple(x.shape)}")

    num_nodes = int(getattr(graph, "num_nodes"))
    if int(x.size(0)) != num_nodes:
        raise ValueError(
            f"{context}: graph.x row count {x.size(0)} does not match graph.num_nodes {num_nodes}"
        )

    x_dim = int(x.size(1))
    if feat_dim is not None and x_dim != int(feat_dim):
        raise ValueError(
            f"{context}: graph.x dim {x_dim} does not match relation/model feat_dim {int(feat_dim)}"
        )

    if question_embeddings is not None:
        if question_embeddings.dim() != 2:
            raise ValueError(
                f"{context}: question_embeddings must be 2D, got {tuple(question_embeddings.shape)}"
            )
        q_dim = int(question_embeddings.size(1))
        if q_dim != x_dim:
            raise ValueError(
                f"{context}: question_embeddings dim {q_dim} does not match graph.x dim {x_dim}"
            )

    logger.info("graph.x shape: %s", tuple(x.shape))
    if question_embeddings is not None:
        logger.info("question_embeddings shape: %s", tuple(question_embeddings.shape))
    return x_dim


def build_stage2_loss_functions(loss_cfg_list: list[dict]) -> list[Any]:
    from gfmrag.losses import BCELoss, KLDivLoss, ListCELoss, MSELoss
    from gfmrag.trainers.sft_trainer import SFTLoss

    loss_fn_map = {
        "bce": BCELoss,
        "listce": ListCELoss,
        "mse": MSELoss,
        "kl": KLDivLoss,
    }

    loss_functions: list[SFTLoss] = []
    for lc in loss_cfg_list:
        name = str(lc["name"])
        if "loss_type" not in lc:
            raise ValueError(f"Loss {name!r} is missing required field loss_type")
        loss_type = str(lc["loss_type"]).lower()
        loss_cls = loss_fn_map.get(loss_type)
        if loss_cls is None:
            raise ValueError(
                f"Loss {name!r} has unknown loss_type={loss_type!r}; "
                "expected one of: bce, listce, mse, kl"
            )
        loss_functions.append(
            SFTLoss(
                name=name,
                loss_fn=loss_cls(),
                weight=float(lc["weight"]),
                target_node_type=str(lc["target_node_type"]),
                is_distillation_loss=bool(lc.get("is_distillation_loss", False)),
            )
        )

    log_loss_functions(loss_functions)
    return loss_functions


def log_loss_functions(loss_functions: list[Any]) -> None:
    hard = [loss for loss in loss_functions if not loss.is_distillation_loss]
    distill = [loss for loss in loss_functions if loss.is_distillation_loss]
    distill_types = sorted({str(loss.target_node_type) for loss in distill})

    logger.info("=== LOSS FUNCTIONS LOADED ===")
    for loss in loss_functions:
        logger.info(
            "  %s: fn=%s weight=%.4f target=%s distill=%s",
            loss.name,
            loss.loss_fn.__class__.__name__,
            float(loss.weight),
            loss.target_node_type,
            bool(loss.is_distillation_loss),
        )
    logger.info("hard loss: %s", [loss.name for loss in hard])
    logger.info("distill loss: %s", [loss.name for loss in distill])
    logger.info("distillation target type: %s", distill_types or "none")


class LoggingSFTTrainerMixin:
    """Add grouped loss metrics without changing SFTTrainer loss behavior."""

    def train_step(self, batch: Any, task_dataset: Any) -> dict[str, float | torch.Tensor]:
        step_metrics = super().train_step(batch, task_dataset)  # type: ignore[misc]
        hard_loss = 0.0
        distill_loss = 0.0
        for loss in self.loss_functions:  # type: ignore[attr-defined]
            value = step_metrics.get(loss.name, 0.0)
            if isinstance(value, torch.Tensor):
                value_f = float(value.detach().item())
            else:
                value_f = float(value)
            weighted = float(loss.weight) * value_f
            if loss.is_distillation_loss:
                distill_loss += weighted
            else:
                hard_loss += weighted
        step_metrics["hard_loss"] = hard_loss
        step_metrics["distill_loss"] = distill_loss
        return step_metrics
