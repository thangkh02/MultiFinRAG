from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from gfmrag.models.gfm_rag_v1.model import GNNRetriever

logger = logging.getLogger(__name__)


class NodeSemanticGNNRetriever(GNNRetriever):
    def __init__(
        self,
        entity_model,
        feat_dim: int,
        ranker,
        init_nodes_weight: bool = True,
        init_nodes_type: str | None = None,
        *,
        use_node_text_semantics: bool = True,
        node_feat_dim: int | None = None,
        node_feat_attr: str = "x",
        node_feat_alpha: float = 0.1,
        node_feat_fusion: str = "add",
        use_semantic_residual_score: bool = False,
        semantic_score_weight: float = 0.05,
    ) -> None:
        super().__init__(
            entity_model=entity_model,
            feat_dim=feat_dim,
            ranker=ranker,
            init_nodes_weight=init_nodes_weight,
            init_nodes_type=init_nodes_type,
        )

        self.use_node_text_semantics = bool(use_node_text_semantics)
        self.node_feat_attr = str(node_feat_attr)
        self.node_feat_fusion = str(node_feat_fusion)
        self.use_semantic_residual_score = bool(use_semantic_residual_score)
        self.semantic_score_weight = float(semantic_score_weight)
        self._warned_missing_node_feat = False

        if self.use_node_text_semantics:
            if node_feat_dim is None:
                raise ValueError("node_feat_dim must be set when use_node_text_semantics is True")
            self.node_mlp = nn.Linear(int(node_feat_dim), self.entity_model.dims[0])
            self.node_feat_alpha = float(node_feat_alpha)
        else:
            self.node_mlp = None
            self.node_feat_alpha = float(node_feat_alpha)

    def _get_node_semantic_features(
        self,
        graph: Data,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.use_node_text_semantics:
            return None

        node_feat = getattr(graph, self.node_feat_attr, None)
        if node_feat is None:
            if not self._warned_missing_node_feat:
                logger.warning(
                    "Node semantic features missing: graph.%s not found. Falling back to legacy input.",
                    self.node_feat_attr,
                )
                self._warned_missing_node_feat = True
            return None

        node_feat = node_feat.to(device)
        projected = self.node_mlp(node_feat)
        projected = projected.unsqueeze(0).expand(batch_size, -1, -1)
        return projected

    def forward(
        self,
        graph: Data,
        batch: dict[str, torch.Tensor],
        entities_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        question_emb = batch["question_embeddings"]
        question_entities_mask = batch["start_nodes_mask"]

        question_embedding = self.question_mlp(question_emb)
        batch_size = question_embedding.size(0)
        relation_representations = (
            self.rel_mlp(graph.rel_attr).unsqueeze(0).expand(batch_size, -1, -1)
        )

        if self.init_nodes_weight and entities_weight is None:
            assert self.init_nodes_type is not None, (
                "init_nodes_type must be set if init_nodes_weight is True and entities_weight is None"
            )
            entities_weight = self.get_entities_weight(
                graph.target_to_other_types[self.init_nodes_type]
            )

        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(0)

        input_state = torch.einsum("bn, bd -> bnd", question_entities_mask, question_embedding)

        node_sem = self._get_node_semantic_features(graph, batch_size, question_embedding.device)
        if node_sem is not None:
            if self.node_feat_fusion == "start_only_add":
                input_state = input_state + self.node_feat_alpha * node_sem * question_entities_mask.unsqueeze(-1)
            else:
                input_state = input_state + self.node_feat_alpha * node_sem

        output = self.entity_model(
            graph, input_state, relation_representations, question_embedding
        )
        output = self.map_entities_to_docs(output, graph)

        if self.use_semantic_residual_score and node_sem is not None:
            sem_score = torch.einsum("bnd,bd->bn", node_sem, question_embedding)
            output = output + self.semantic_score_weight * sem_score

        return output
