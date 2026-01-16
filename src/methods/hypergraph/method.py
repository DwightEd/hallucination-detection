"""Hypergraph Method - Framework Integration.

将超图检测方法集成到框架中，支持GNN和统计特征两种模式。
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction
from src.methods.base import BaseMethod

from .model import HyperCHARMModel
from .data import HypergraphData, HypergraphBuilder
from .utils import DEVICE

logger = logging.getLogger(__name__)


@METHODS.register("hypergraph", aliases=["hypercharm", "hypergraph_nn"])
class HypergraphMethod(BaseMethod):
    """Hypergraph-based hallucination detection method.

    支持两种模式：
    1. GNN模式：当有full_attention时，使用HyperCHARM进行训练和预测
    2. 统计模式：当无full_attention时，提取统计特征使用传统分类器

    支持两种训练级别 (config.level):
    - "sample": 样本级别，幻觉样本的所有 response tokens 标记为 1
    - "token": Token级别，使用 hallucination_labels 精确标记
    - "both": 优先使用 token 级别，无标签时回退到 sample 级别
    """

    # 支持 token 级别训练
    supports_token_level = True

    def __init__(self, config: Optional[MethodConfig] = None):
        """初始化超图方法。
        
        Args:
            config: 方法配置
        """
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
        self.gnn_model: Optional[HyperCHARMModel] = None
        self.builder = HypergraphBuilder({
            "attention_threshold": self.attention_threshold,
            "topk_per_row": self.topk_per_row,
        })

        # State
        self._node_dim = None
        self._hedge_dim = None
        self._use_gnn = False

    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract features for classification.

        当有full_attention时，使用GNN输出作为特征；
        否则使用注意力统计特征。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            特征向量
        """
        hypergraph = self.builder.build_from_features(features)

        if hypergraph is not None and self._use_gnn and self.gnn_model is not None:
            return self._extract_gnn_features(hypergraph)

        return self._extract_statistical_features(features)

    def _extract_gnn_features(self, hypergraph: HypergraphData) -> np.ndarray:
        """使用GNN提取特征。
        
        Args:
            hypergraph: HypergraphData对象
            
        Returns:
            特征向量
        """
        self.gnn_model.eval()
        hypergraph = hypergraph.to(DEVICE)

        with torch.no_grad():
            logits = self.gnn_model(hypergraph)
            probs = torch.sigmoid(logits)

            mask = hypergraph.node_pos >= hypergraph.response_idx[hypergraph.batch]
            response_probs = probs[mask]

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

    def _extract_statistical_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract statistical features from attention diagonals.
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            特征向量
        """
        from src.core import FeatureAccessor

        feat_list = []

        with FeatureAccessor(features, prefer_fast=True) as accessor:
            # Attention diagonal statistics
            diag = accessor.get_attention_diags()
            if diag is not None:
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
            lap = accessor.get_laplacian_diags()
            if lap is not None:
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
            ent = accessor.get_attention_entropy()
            if ent is not None:
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
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """Train the method.

        自动检测是否有full_attention，决定使用GNN还是统计特征模式。
        
        Args:
            features_list: 特征列表
            labels: 标签列表
            cv: 是否使用交叉验证
            
        Returns:
            训练指标
        """
        def has_full_attention_available(f):
            if hasattr(f, 'full_attention') and f.full_attention is not None:
                return True
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
        """Train GNN model.
        
        Args:
            features_list: 特征列表
            labels: 标签列表
            
        Returns:
            训练指标
        """
        logger.info("Building hypergraphs...")

        level = self.config.level
        logger.info(f"Level: {level}")

        hypergraphs = []
        valid_labels = []
        skipped_count = 0
        no_label_count = 0
        n_with_token_labels = 0
        n_fallback = 0

        for i, feat in enumerate(features_list):
            hg = self.builder.build_from_features(feat)
            if hg is not None:
                label = labels[i] if labels else feat.label
                if label is not None:
                    has_token_labels = hg.y.sum() > 0

                    if level == "token":
                        if label == 1 and not has_token_labels:
                            skipped_count += 1
                            continue
                        if has_token_labels:
                            n_with_token_labels += 1

                    elif level == "sample":
                        if label == 1:
                            response_mask = hg.node_pos >= hg.response_idx[0]
                            hg.y[response_mask] = 1.0
                            n_fallback += 1

                    else:  # "both" 或默认
                        if label == 1 and not has_token_labels:
                            response_mask = hg.node_pos >= hg.response_idx[0]
                            hg.y[response_mask] = 1.0
                            n_fallback += 1
                        elif has_token_labels:
                            n_with_token_labels += 1

                    hypergraphs.append(hg)
                    valid_labels.append(label)
                else:
                    no_label_count += 1
            else:
                skipped_count += 1

        n_hallucinated = sum(1 for l in valid_labels if l == 1)

        logger.info(f"Hypergraph building complete:")
        logger.info(f"  - Total samples: {len(features_list)}")
        logger.info(f"  - Valid hypergraphs: {len(hypergraphs)}")
        logger.info(f"  - Skipped: {skipped_count}")
        logger.info(f"  - No label: {no_label_count}")

        if level in ("token", "both"):
            logger.info(f"  - With precise token labels: {n_with_token_labels}/{n_hallucinated} hallucinated")
        if level in ("sample", "both") and n_fallback > 0:
            logger.info(f"  - Sample-level fallback: {n_fallback} samples")

        if skipped_count > len(features_list) * 0.1:
            logger.warning(
                f"More than 10% of samples were skipped ({skipped_count}/{len(features_list)}). "
                f"Consider increasing max_length in feature extraction."
            )

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

        val_indices = pos_indices[:n_pos_val] + neg_indices[:n_neg_val]
        train_indices = list(set(range(len(hypergraphs))) - set(val_indices))

        train_graphs = [hypergraphs[i] for i in train_indices]
        val_graphs = [hypergraphs[i] for i in val_indices]

        self.gnn_model = self._train_gnn(train_graphs, val_graphs)
        self._train_classifier_on_gnn_features(features_list, labels)
        self.is_fitted = True

        metrics = self._evaluate_graphs(self.gnn_model, val_graphs)
        metrics["n_samples"] = len(hypergraphs)
        metrics["n_positive"] = sum(valid_labels)
        metrics["n_negative"] = len(valid_labels) - sum(valid_labels)
        metrics["mode"] = "gnn"

        return metrics

    def _train_gnn(
        self,
        train_graphs: List[HypergraphData],
        val_graphs: List[HypergraphData],
    ) -> HyperCHARMModel:
        """Train HyperCHARM model.
        
        Args:
            train_graphs: 训练图列表
            val_graphs: 验证图列表
            
        Returns:
            训练后的模型
        """
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
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
    ):
        """Train classifier on GNN output features for consistency.
        
        Args:
            features_list: 特征列表
            labels: 标签列表
        """
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
        """Evaluate model on hypergraphs.
        
        Args:
            model: 模型
            graphs: 图列表
            
        Returns:
            评估指标
        """
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
            return {"auroc": 0.5, "aupr": 0.0}

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
        """Predict hallucination probability.
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("Method not fitted. Call fit() first.")

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
        """Save method including GNN model.
        
        Args:
            path: 保存路径
        """
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
            "gnn_state": self.gnn_model.state_dict() if self.gnn_model else None,
            "gnn_config": {
                "hidden_dim": self.hidden_dim,
                "gnn_layers": self.gnn_layers,
                "dropout": self.dropout,
                "residual_mp": self.residual_mp,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved hypergraph method to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load method including GNN model.
        
        Args:
            path: 加载路径
        """
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

    def predict_tokens(self, features: ExtractedFeatures) -> Dict[str, Any]:
        """Predict token-level hallucination probabilities.
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            包含token级别预测的字典
        """
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
