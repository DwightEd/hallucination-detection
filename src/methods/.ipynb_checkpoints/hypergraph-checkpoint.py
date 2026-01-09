"""Hypergraph-based Hallucination Detection. 

使用超图神经网络（HyperCHARM）检测LLM幻觉。
基于注意力模式构建超图，利用消息传递学习token级别的幻觉概率。

设计原则：
- 与其他方法保持一致的接口（继承BaseMethod）
- 支持两种模式：
  1. 有full_attention时使用GNN
  2. 无full_attention时回退到统计特征+分类器
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import logging
import copy
import pickle
import numpy as np
import torch
import torch. nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction
from .base import BaseMethod

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Utility Functions
# ==============================================================================

def make_mlp(in_dim: int, hidden_dims: List[int], out_dim: int,
             activation=nn.ReLU, dropout:  float = 0.0) -> nn.Sequential:
    """Build MLP with LayerNorm and optional dropout."""
    layers = []
    prev = in_dim
    for h in hidden_dims: 
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ==============================================================================
# HyperCHARM Model Components
# ==============================================================================

class HyperCharmLayer(nn.Module):
    """Hypergraph message passing layer."""

    def __init__(self, node_dim: int, hedge_dim: int, hidden_dim: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.node2edge = make_mlp(node_dim + 2, [hidden_dim], hidden_dim)
        self.edge2node = make_mlp(hedge_dim + hidden_dim, [hidden_dim], node_dim)
        self.ln_out = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, he_index: torch.Tensor,
                he_attr: torch.Tensor, he_mark: torch.Tensor,
                he_count: torch.Tensor) -> torch.Tensor:
        he_ids = he_index[1]
        node_ids = he_index[0]

        # Node -> Edge aggregation
        msg_ne = self.node2edge(torch.cat([x[node_ids], he_mark[he_ids]], dim=-1))
        agg_e = torch.zeros((he_attr.size(0), msg_ne.size(-1)), device=x.device)
        agg_e.index_add_(0, he_ids, msg_ne)
        agg_e = agg_e / (he_count.unsqueeze(-1) + 1e-6)

        # Edge -> Node aggregation
        inc_msg = self.edge2node(torch.cat([he_attr[he_ids], agg_e[he_ids]], dim=-1))
        inc_msg = F.relu(inc_msg)

        out = torch.zeros_like(x)
        out.index_add_(0, node_ids, inc_msg)

        # Normalize by node degree
        num_nodes = x.size(0)
        node_deg = torch.bincount(node_ids, minlength=num_nodes).float().unsqueeze(-1).to(x.device)
        out = out / (node_deg + 1e-6)

        out = self.ln_out(out)
        return x + out if self.residual else out


class HyperCHARMModel(nn.Module):
    """Hypergraph neural network for token-level hallucination detection."""

    def __init__(self, node_dim: int, hedge_dim: int, config: Dict[str, Any]):
        super().__init__()
        hidden_dim = config.get("hidden_dim", 128)
        n_layers = config.get("gnn_layers", 2)
        dropout = config.get("dropout", 0.25)
        residual = config.get("residual_mp", True)

        self.in_proj = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            HyperCharmLayer(
                node_dim=hidden_dim,
                hedge_dim=hedge_dim,
                hidden_dim=hidden_dim,
                residual=residual
            )
            for _ in range(n_layers)
        ])

        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.LayerNorm(max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_dim // 2), 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: 'HypergraphData') -> torch.Tensor:
        h = F.relu(self.in_proj(data.x))
        for layer in self.layers:
            h = layer(h, data.he_index, data.he_attr, data.he_mark, data.he_count)
        return self.pred(h).view(-1)


# ==============================================================================
# Hypergraph Data Structure
# ==============================================================================

class HypergraphData: 
    """Container for hypergraph data."""

    def __init__(self, x: torch.Tensor, he_index:  torch.Tensor,
                 he_attr: torch.Tensor, he_mark: torch.Tensor,
                 he_count: torch.Tensor, y: torch.Tensor,
                 node_pos: torch.Tensor, response_idx: int,
                 sample_id: str = ""):
        self.x = x
        self.he_index = he_index
        self.he_attr = he_attr
        self.he_mark = he_mark
        self.he_count = he_count
        self.y = y
        self.node_pos = node_pos
        self.response_idx = torch.tensor([response_idx])
        self.sample_id = sample_id
        self.batch = torch.zeros(x.size(0), dtype=torch.long)

    def to(self, device:  str) -> 'HypergraphData':
        self.x = self.x.to(device)
        self.he_index = self.he_index.to(device)
        self.he_attr = self.he_attr.to(device)
        self.he_mark = self.he_mark.to(device)
        self.he_count = self.he_count.to(device)
        self.y = self.y.to(device)
        self.node_pos = self.node_pos.to(device)
        self.response_idx = self.response_idx.to(device)
        self.batch = self.batch.to(device)
        return self


# ==============================================================================
# Hypergraph Construction
# ==============================================================================

class HypergraphBuilder:
    """Build hypergraph from attention matrices."""

    def __init__(self, config: Dict[str, Any]):
        self.tau = config.get("attention_threshold", 0.05)
        self.topk_per_row = config.get("topk_per_row", 16)
        self.min_members = config.get("min_members_in_he", 2)
        self.include_center = config.get("include_center_token", True)

    def build_from_attention(
        self,
        attention:  torch.Tensor,
        response_idx: int,
        token_labels: Optional[List[int]] = None,
        sample_id: str = "",
    ) -> HypergraphData: 
        """Build hypergraph from attention matrix."""
        attention = attention.float()
        n_layers, n_heads, seq_len, _ = attention.shape
        num_heads = n_layers * n_heads
        attention_flat = attention.reshape(num_heads, seq_len, seq_len)

        # Node features:  self-attention values across all heads
        diag_idx = torch.arange(seq_len)
        self_att = attention_flat[: , diag_idx, diag_idx].transpose(0, 1)
        self_att = torch.clamp(self_att, 0.0, 1.0)

        # Normalize node features
        if self_att.numel() > 0:
            self_att = torch.clamp(self_att, -5.0, 5.0)
            mean_val = self_att.mean(dim=0, keepdim=True)
            std_val = self_att.std(dim=0, keepdim=True) + 1e-6
            self_att = (self_att - mean_val) / std_val

        x = self_att

        # Masks
        tri_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-1)
        valid_mask = tri_mask.clone()
        valid_mask[: response_idx, :response_idx] = False

        # Build hyperedges
        he_node_list = []
        he_id_list = []
        he_attr_list = []
        he_mark_list = []
        he_count_list = []
        he_counter = 0

        for head_idx in range(num_heads):
            for i in range(response_idx, seq_len):
                row = attention_flat[head_idx, i, : ].clone()
                row[~valid_mask[i]] = 0.0

                valid_indices = (row > self.tau).nonzero(as_tuple=True)[0]
                if len(valid_indices) > self.topk_per_row:
                    topk_vals, topk_idx = torch.topk(row[valid_indices], self.topk_per_row)
                    valid_indices = valid_indices[topk_idx]

                members = valid_indices.tolist()
                if self.include_center and i not in members:
                    members.append(i)

                if len(members) >= self.min_members:
                    max_val = row.max().item()
                    mean_val = row[valid_indices].mean().item() if len(valid_indices) > 0 else 0.0
                    he_attr_list.append([max_val, mean_val])
                    he_count_list.append(len(members))

                    for member in members:
                        he_node_list.append(member)
                        he_id_list.append(he_counter)
                        is_center = 1.0 if member == i else 0.0
                        rel_pos = (member - response_idx) / max(1, seq_len - response_idx)
                        he_mark_list.append([is_center, rel_pos])

                    he_counter += 1

        # Handle empty hypergraph
        if he_counter == 0:
            for i in range(response_idx, seq_len):
                he_attr_list.append([0.1, 0.1])
                he_count_list.append(1)
                he_node_list.append(i)
                he_id_list.append(he_counter)
                he_mark_list.append([1.0, (i - response_idx) / max(1, seq_len - response_idx)])
                he_counter += 1

        he_index = torch.tensor([he_node_list, he_id_list], dtype=torch.long)
        he_attr = torch.tensor(he_attr_list, dtype=torch.float32)
        he_mark = torch.tensor(he_mark_list, dtype=torch.float32)
        he_count = torch.tensor(he_count_list, dtype=torch.float32)

        # Normalize hyperedge attributes
        if he_attr.numel() > 0:
            he_attr = torch.clamp(he_attr, 0.0, 1.0)
            mean_val = he_attr.mean(dim=0, keepdim=True)
            std_val = he_attr.std(dim=0, keepdim=True) + 1e-6
            he_attr = (he_attr - mean_val) / std_val

        if token_labels is not None:
            y = torch.tensor(token_labels, dtype=torch.float32)
        else:
            y = torch.zeros(seq_len, dtype=torch.float32)

        node_pos = torch.arange(seq_len)

        return HypergraphData(
            x=x, he_index=he_index, he_attr=he_attr, he_mark=he_mark,
            he_count=he_count, y=y, node_pos=node_pos,
            response_idx=response_idx, sample_id=sample_id,
        )

    def build_from_features(self, features: ExtractedFeatures) -> Optional[HypergraphData]:
        """Build hypergraph from ExtractedFeatures."""
        # Use lazy loading to get full_attention
        attention = features.get_full_attention()
        
        if attention is None:
            return None

        response_idx = features.prompt_len
        token_labels = features.metadata.get("token_labels", None)

        result = self.build_from_attention(
            attention=attention,
            response_idx=response_idx,
            token_labels=token_labels,
            sample_id=features.sample_id,
        )
        
        # Release large feature after building hypergraph
        features.release_large_features()
        
        return result


# ==============================================================================
# Hypergraph Method - Framework Integration
# ==============================================================================

@METHODS.register("hypergraph", aliases=["hypercharm", "hypergraph_nn"])
class HypergraphMethod(BaseMethod):
    """Hypergraph-based hallucination detection method. 

    支持两种模式：
    1. GNN模式：当有full_attention时，使用HyperCHARM进行训练和预测
    2. 统计模式：当无full_attention时，提取统计特征使用传统分类器

    这确保了方法的普适性和与其他方法的一致性。
    """

    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)

        params = self.config.params or {}

        # Model architecture
        self.hidden_dim = params.get("hidden_dim", 128)
        self.gnn_layers = params.get("gnn_layers", 2)
        self.dropout = params.get("dropout", 0.25)
        self.residual_mp = params.get("residual_mp", True)

        # Training
        self.lr = params.get("lr", 3e-4)
        self.weight_decay = params.get("weight_decay", 0.001)
        self.epochs = params.get("epochs", 50)
        self.patience = params.get("patience", 5)

        # Hypergraph construction
        self.attention_threshold = params.get("attention_threshold", 0.05)
        self.topk_per_row = params.get("topk_per_row", 16)

        # Aggregation
        self.aggregation = params.get("aggregation", "max")
        self.score_threshold = params.get("score_threshold", 0.5)

        # Components
        self.gnn_model:  Optional[HyperCHARMModel] = None
        self.builder = HypergraphBuilder({
            "attention_threshold": self.attention_threshold,
            "topk_per_row": self.topk_per_row,
        })

        # State
        self._node_dim = None
        self._hedge_dim = None
        self._use_gnn = False  # 是否使用GNN模式

    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract features for classification.

        当有full_attention时，使用GNN输出作为特征；
        否则使用注意力统计特征。
        """
        # 尝试从hypergraph提取特征
        hypergraph = self.builder.build_from_features(features)

        if hypergraph is not None and self._use_gnn and self.gnn_model is not None: 
            # 使用GNN提取特征
            self.gnn_model.eval()
            hypergraph = hypergraph.to(DEVICE)

            with torch.no_grad():
                logits = self.gnn_model(hypergraph)
                probs = torch.sigmoid(logits)

                mask = hypergraph.node_pos >= hypergraph.response_idx[hypergraph.batch]
                response_probs = probs[mask]

                # 聚合为样本级特征
                if len(response_probs) > 0:
                    feat_vec = np.array([
                        response_probs.max().item(),
                        response_probs.mean().item(),
                        response_probs.std().item(),
                        (response_probs > 0.5).float().mean().item(),
                        response_probs.median().item(),
                    ], dtype=np.float32)
                else:
                    feat_vec = np.zeros(5, dtype=np.float32)

            return feat_vec

        # 回退到统计特征
        return self._extract_statistical_features(features)

    def _extract_statistical_features(self, features:  ExtractedFeatures) -> np.ndarray:
        """Extract statistical features from attention diagonals."""
        feat_list = []

        # Attention diagonal statistics
        if features.attn_diags is not None: 
            diag = features.attn_diags
            if isinstance(diag, torch.Tensor):
                diag = diag.float().cpu().numpy()

            # Focus on response portion
            if features.response_len > 0 and features.prompt_len < diag.shape[-1]:
                start = features.prompt_len
                end = min(start + features.response_len, diag.shape[-1])
                diag = diag[..., start:end]

            if diag.size > 0:
                feat_list.extend([
                    np.mean(diag), np.std(diag), np.max(diag), np.min(diag),
                    np.median(diag), np.percentile(diag, 25), np.percentile(diag, 75),
                ])
            else:
                feat_list.extend([0.0] * 7)
        else:
            feat_list.extend([0.0] * 7)

        # Laplacian diagonal statistics
        if features.laplacian_diags is not None:
            lap = features.laplacian_diags
            if isinstance(lap, torch.Tensor):
                lap = lap.float().cpu().numpy()

            if features.response_len > 0 and features.prompt_len < lap.shape[-1]:
                start = features.prompt_len
                end = min(start + features.response_len, lap.shape[-1])
                lap = lap[..., start:end]

            if lap.size > 0:
                feat_list.extend([
                    np.mean(lap), np.std(lap), np.max(lap), np.min(lap),
                ])
            else: 
                feat_list.extend([0.0] * 4)
        else: 
            feat_list.extend([0.0] * 4)

        # Attention entropy statistics
        if features.attn_entropy is not None:
            ent = features.attn_entropy
            if isinstance(ent, torch.Tensor):
                ent = ent.float().cpu().numpy()

            if features.response_len > 0 and features.prompt_len < ent.shape[-1]: 
                start = features.prompt_len
                end = min(start + features.response_len, ent.shape[-1])
                ent = ent[..., start:end]

            if ent.size > 0:
                feat_list.extend([
                    np.mean(ent), np.std(ent), np.max(ent), np.min(ent),
                ])
            else:
                feat_list.extend([0.0] * 4)
        else:
            feat_list.extend([0.0] * 4)

        return np.array(feat_list, dtype=np.float32)

    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels:  Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]: 
        """Train the method. 

        自动检测是否有full_attention，决定使用GNN还是统计特征模式。
        """
        # 检查是否有full_attention（支持懒加载模式）
        def has_full_attention_available(f):
            # Direct access
            if hasattr(f, 'full_attention') and f.full_attention is not None:
                return True
            # Lazy loading path
            if hasattr(f, 'metadata') and f.metadata:
                feature_paths = f.metadata.get("_feature_paths", {})
                if "full_attentions" in feature_paths:
                    return True
            return False
        
        has_full_attention = any(has_full_attention_available(f) for f in features_list)

        if has_full_attention: 
            logger.info("Full attention available, using GNN mode")
            self._use_gnn = True
            return self._fit_gnn(features_list, labels)
        else:
            logger.info("No full attention, using statistical features mode")
            self._use_gnn = False
            return super().fit(features_list, labels, cv)

    def _fit_gnn(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Train GNN model."""
        # Build hypergraphs
        hypergraphs = []
        valid_labels = []

        for i, feat in enumerate(features_list):
            hg = self.builder.build_from_features(feat)
            if hg is not None:
                label = labels[i] if labels else feat.label
                if label is not None: 
                    if label == 1:
                        response_mask = hg.node_pos >= hg.response_idx[0]
                        hg.y[response_mask] = 1.0
                    hypergraphs.append(hg)
                    valid_labels.append(label)

        if len(hypergraphs) == 0:
            logger.warning("No valid hypergraphs, falling back to statistical mode")
            self._use_gnn = False
            return super().fit(features_list, labels, cv=True)

        logger.info(f"Built {len(hypergraphs)} hypergraphs for training")

        self._node_dim = hypergraphs[0].x.size(1)
        self._hedge_dim = hypergraphs[0].he_attr.size(1)

        # Split for validation (stratified)
        import random
        random.seed(self.config.random_seed or 42)

        pos_indices = [i for i, l in enumerate(valid_labels) if l == 1]
        neg_indices = [i for i, l in enumerate(valid_labels) if l == 0]

        val_size = max(1, int(len(hypergraphs) * 0.1))
        n_pos_val = max(1, int(len(pos_indices) / len(hypergraphs) * val_size))
        n_neg_val = val_size - n_pos_val

        random.shuffle(pos_indices)
        random.shuffle(neg_indices)

        val_indices = pos_indices[: n_pos_val] + neg_indices[: n_neg_val]
        train_indices = list(set(range(len(hypergraphs))) - set(val_indices))

        train_graphs = [hypergraphs[i] for i in train_indices]
        val_graphs = [hypergraphs[i] for i in val_indices]

        # Train GNN
        self.gnn_model = self._train_gnn(train_graphs, val_graphs)

        # Also train classifier on GNN features for consistency
        self._train_classifier_on_gnn_features(features_list, labels)

        self.is_fitted = True

        # Evaluate
        metrics = self._evaluate_graphs(self.gnn_model, val_graphs)
        metrics["n_samples"] = len(hypergraphs)
        metrics["n_positive"] = sum(valid_labels)
        metrics["n_negative"] = len(valid_labels) - sum(valid_labels)
        metrics["mode"] = "gnn"

        return metrics

    def _train_gnn(
        self,
        train_graphs:  List[HypergraphData],
        val_graphs: List[HypergraphData],
    ) -> HyperCHARMModel:
        """Train HyperCHARM model."""
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup

        config = {
            "hidden_dim": self.hidden_dim,
            "gnn_layers": self.gnn_layers,
            "dropout": self.dropout,
            "residual_mp": self.residual_mp,
        }

        model = HyperCHARMModel(self._node_dim, self._hedge_dim, config).to(DEVICE)

        # Compute class weight
        pos_count = sum((g.y[g.node_pos >= g.response_idx[0]] == 1).sum().item() for g in train_graphs)
        neg_count = sum((g.y[g.node_pos >= g.response_idx[0]] == 0).sum().item() for g in train_graphs)
        pos_weight = min(neg_count / max(pos_count, 1), 10.0)
        logger.info(f"Pos weight: {pos_weight:.3f}")

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self.epochs * len(train_graphs)
        warmup_steps = int(0.05 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_val_score = -1
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            for graph in train_graphs: 
                graph = graph.to(DEVICE)
                logits = model(graph)

                mask = graph.node_pos >= graph.response_idx[graph.batch]
                if mask.sum() == 0:
                    continue

                loss = loss_fn(logits[mask], graph.y[mask])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            val_metrics = self._evaluate_graphs(model, val_graphs)
            val_score = val_metrics.get("aupr", 0.0)

            avg_loss = total_loss / max(1, len(train_graphs))
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, "
                       f"Val AUROC: {val_metrics['auroc']:.4f}, Val AUPR: {val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def _train_classifier_on_gnn_features(
        self,
        features_list:  List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
    ):
        """Train classifier on GNN output features for consistency."""
        X = []
        y = []

        for i, feat in enumerate(features_list):
            try:
                x = self.extract_method_features(feat)
                if x is not None and not np.any(np.isnan(x)):
                    X.append(x)
                    label = labels[i] if labels else feat.label
                    if label is not None: 
                        y.append(label)
            except Exception as e:
                logger.warning(f"Failed to extract features for {feat.sample_id}: {e}")

        if len(X) > 0 and len(y) > 0:
            X = np.array(X)
            y = np.array(y)
            X_scaled = self.scaler.fit_transform(X)
            self.classifier = self._create_classifier()
            self.classifier.fit(X_scaled, y)

    def _evaluate_graphs(
        self,
        model: HyperCHARMModel,
        graphs: List[HypergraphData],
    ) -> Dict[str, float]: 
        """Evaluate model on hypergraphs."""
        model.eval()
        all_y = []
        all_p = []

        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(DEVICE)
                logits = model(graph)
                prob = torch.sigmoid(logits)

                mask = graph.node_pos >= graph.response_idx[graph.batch]
                all_y.append(graph.y[mask].float().cpu().numpy())
                all_p.append(prob[mask].float().cpu().numpy())

        if len(all_y) == 0:
            return {"auroc": 0.5, "aupr":  0.0}

        y = np.concatenate(all_y)
        p = np.concatenate(all_p)

        try:
            auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        except Exception: 
            auroc = 0.5

        try:
            aupr = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0.0
        except Exception:
            aupr = 0.0

        return {"auroc": auroc, "aupr": aupr}

    def predict(self, features: ExtractedFeatures) -> Prediction:
        """Predict hallucination probability."""
        if not self.is_fitted:
            raise ValueError("Method not fitted.  Call fit() first.")

        if self._use_gnn and self.gnn_model is not None: 
            hypergraph = self.builder.build_from_features(features)

            if hypergraph is not None: 
                self.gnn_model.eval()
                hypergraph = hypergraph.to(DEVICE)

                with torch.no_grad():
                    logits = self.gnn_model(hypergraph)
                    probs = torch.sigmoid(logits)

                    mask = hypergraph.node_pos >= hypergraph.response_idx[hypergraph.batch]
                    response_probs = probs[mask]

                    if self.aggregation == "max": 
                        score = response_probs.max().item() if len(response_probs) > 0 else 0.5
                    elif self.aggregation == "mean":
                        score = response_probs.mean().item() if len(response_probs) > 0 else 0.5
                    else:
                        score = response_probs.max().item() if len(response_probs) > 0 else 0.5

                return Prediction(
                    sample_id=features.sample_id,
                    score=score,
                    label=1 if score > self.score_threshold else 0,
                    confidence=abs(score - 0.5) * 2,
                )

        # Fallback to classifier
        x = self.extract_method_features(features)
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        proba = self.classifier.predict_proba(x_scaled)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])

        return Prediction(
            sample_id=features.sample_id,
            score=score,
            label=1 if score > self.score_threshold else 0,
            confidence=abs(score - 0.5) * 2,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save method including GNN model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted method")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            "use_gnn": self._use_gnn,
            "node_dim": self._node_dim,
            "hedge_dim": self._hedge_dim,
            "gnn_state":  self.gnn_model.state_dict() if self.gnn_model else None,
            "gnn_config": {
                "hidden_dim": self.hidden_dim,
                "gnn_layers": self.gnn_layers,
                "dropout": self.dropout,
                "residual_mp":  self.residual_mp,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved hypergraph method to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load method including GNN model."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Method file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.config = state["config"]
        self.classifier = state["classifier"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        self._feature_dim = state.get("feature_dim")
        self._use_gnn = state.get("use_gnn", False)
        self._node_dim = state.get("node_dim")
        self._hedge_dim = state.get("hedge_dim")

        if state.get("gnn_state") is not None and self._node_dim and self._hedge_dim:
            self.gnn_model = HyperCHARMModel(
                self._node_dim,
                self._hedge_dim,
                state["gnn_config"]
            ).to(DEVICE)
            self.gnn_model.load_state_dict(state["gnn_state"])

        logger.info(f"Loaded hypergraph method from {path}")


@METHODS.register("hypergraph_token", aliases=["hypercharm_token"])
class HypergraphTokenMethod(HypergraphMethod):
    """Token-level hypergraph method for detailed analysis."""

    def predict_tokens(self, features:  ExtractedFeatures) -> Dict[str, Any]:
        """Predict token-level hallucination probabilities."""
        if not self.is_fitted or self.gnn_model is None:
            raise ValueError("Model not fitted or GNN not available.")

        hypergraph = self.builder.build_from_features(features)
        if hypergraph is None:
            return {"sample_id": features.sample_id, "token_probs": [], "response_idx": features.prompt_len}

        self.gnn_model.eval()
        hypergraph = hypergraph.to(DEVICE)

        with torch.no_grad():
            logits = self.gnn_model(hypergraph)
            probs = torch.sigmoid(logits)

        return {
            "sample_id": features.sample_id,
            "token_probs": probs.float().cpu().numpy().tolist(),
            "response_idx": features.prompt_len,
        }