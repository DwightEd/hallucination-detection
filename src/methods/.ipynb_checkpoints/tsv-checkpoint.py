"""
TSV (Truthfulness Separator Vector) - Steer LLM Latents for Hallucination Detection

基于 ICML 2025 论文: "Steer LLM Latents for Hallucination Detection"
GitHub: https://github.com/deeplearning-wisc/tsv

=============================================================================
修复版本 - 统一模型保存格式为 model.pkl
=============================================================================

核心思想:
1. 学习一个轻量级的 steering vector (TSV)，在推理时加入 LLM 的隐藏状态
2. TSV 重塑表示空间，增强 truthful 和 hallucinated 数据的分离度
3. 使用 vMF (von Mises-Fisher) 分布进行分类
4. 两阶段训练: 初始训练 + 基于最优传输的伪标签增强训练

关键设计:
- 使用最后一个 token 的 embedding (last-token)
- 应用于早中期层 (25%-50% 的层深度)
- Steering 强度 λ 默认为 5
- vMF 浓度参数 κ 默认为 10
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction, MethodError
from .base import BaseMethod

logger = logging.getLogger(__name__)


# =============================================================================
# Sinkhorn-Knopp 算法 (用于最优传输伪标签)
# =============================================================================

def sinkhorn_knopp(
    scores: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 3,
) -> torch.Tensor:
    """Sinkhorn-Knopp 算法用于最优传输。"""
    Q = torch.exp(scores / epsilon)
    Q /= Q.sum()
    
    N, K = Q.shape
    
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= Q.sum(dim=0, keepdim=True)
    
    Q /= Q.sum(dim=1, keepdim=True)
    
    return Q


# =============================================================================
# TSV 核心检测器
# =============================================================================

class TSVDetector:
    """Truthfulness Separator Vector 检测器。"""
    
    DEFAULT_SEED = 42
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # TSV 参数
        self.steering_strength = config.get('steering_strength', 5.0)
        self.kappa = config.get('kappa', 10.0)
        
        # 层选择参数
        self.layer_selection = config.get('layer_selection', 'middle')
        self.specific_layers = config.get('specific_layers', None)
        
        # 训练参数
        self.lr = config.get('learning_rate', 5e-3)
        self.epochs_stage1 = config.get('epochs_stage1', 20)
        self.epochs_stage2 = config.get('epochs_stage2', 20)
        self.batch_size = config.get('batch_size', 128)
        
        # Sinkhorn 参数
        self.sinkhorn_iters = config.get('sinkhorn_iters', 3)
        self.sinkhorn_epsilon = config.get('sinkhorn_epsilon', 0.05)
        
        # EMA 参数
        self.ema_decay = config.get('ema_decay', 0.99)
        
        # 伪标签置信度阈值
        self.confidence_threshold = config.get('confidence_threshold', 0.9)
        
        # 可学习参数 (训练时初始化)
        self.tsv_vector = None
        self.prototype_truthful = None
        self.prototype_hallucinated = None
        
        # 训练状态
        self.hidden_dim = None
        self.selected_layers = None
        self.is_fitted = False
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _select_layers(self, num_layers: int) -> List[int]:
        """选择要使用的层 (论文发现早中期层 25%-50% 效果最好)"""
        if self.layer_selection == 'specific' and self.specific_layers:
            return [l for l in self.specific_layers if 0 <= l < num_layers]
        elif self.layer_selection == 'middle':
            start = max(0, int(num_layers * 0.25))
            end = min(num_layers, int(num_layers * 0.5))
            return list(range(start, end))
        elif self.layer_selection == 'all':
            return list(range(num_layers))
        else:
            start = max(0, int(num_layers * 0.25))
            end = min(num_layers, int(num_layers * 0.5))
            return list(range(start, end))
    
    def _extract_last_token_embedding(
        self,
        hidden_states: np.ndarray,
        num_layers: int
    ) -> np.ndarray:
        """提取 last-token embedding (论文核心)。
        
        Args:
            hidden_states: [n_layers, seq_len, hidden_dim]
            num_layers: 总层数
            
        Returns:
            [selected_layers * hidden_dim] 拼接的特征向量
        """
        if self.selected_layers is None:
            self.selected_layers = self._select_layers(num_layers)
        
        # 获取最后一个 token
        embeddings = []
        for layer_idx in self.selected_layers:
            if layer_idx < hidden_states.shape[0]:
                emb = hidden_states[layer_idx, -1, :]  # last token
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            raise ValueError("No valid layers selected")
        
        return np.concatenate(embeddings, axis=0)
    
    def _compute_steered_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """计算 steered embedding: f' = f + λ * v"""
        if self.tsv_vector is None:
            return features
        return features + self.steering_strength * self.tsv_vector
    
    def _compute_vmf_logits(self, features: torch.Tensor) -> torch.Tensor:
        """计算 vMF 分布的 logits。
        
        vMF: p(x|μ) ∝ exp(κ * μᵀx / ||x||)
        """
        features_norm = F.normalize(features, dim=-1)
        proto_truthful_norm = F.normalize(self.prototype_truthful, dim=-1)
        proto_hallu_norm = F.normalize(self.prototype_hallucinated, dim=-1)
        
        logit_truthful = self.kappa * (features_norm @ proto_truthful_norm)
        logit_hallu = self.kappa * (features_norm @ proto_hallu_norm)
        
        return torch.stack([logit_truthful, logit_hallu], dim=-1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练 TSV 检测器。
        
        Args:
            X: [N, feature_dim] 特征矩阵
            y: [N] 标签 (0=truthful, 1=hallucinated)
            
        Returns:
            训练指标字典
        """
        torch.manual_seed(self.DEFAULT_SEED)
        np.random.seed(self.DEFAULT_SEED)
        
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).long().to(self.device)
        
        self.hidden_dim = X.shape[1]
        
        # 初始化 TSV 向量 (零初始化，论文推荐)
        self.tsv_vector = nn.Parameter(torch.zeros(self.hidden_dim, device=self.device))
        
        # 初始化原型 (使用类均值)
        mask_truthful = (y == 0)
        mask_hallu = (y == 1)
        
        self.prototype_truthful = X[mask_truthful].mean(dim=0)
        self.prototype_hallucinated = X[mask_hallu].mean(dim=0)
        
        # 创建数据加载器
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 优化器
        optimizer = torch.optim.Adam([self.tsv_vector], lr=self.lr)
        
        # Stage 1: 在标注数据上训练
        total_loss = 0.0
        for epoch in range(self.epochs_stage1):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                steered = self._compute_steered_embedding(batch_X)
                logits = self._compute_vmf_logits(steered)
                loss = F.cross_entropy(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                # EMA 更新原型
                with torch.no_grad():
                    for i, (x, label) in enumerate(zip(batch_X, batch_y)):
                        if label == 0:
                            self.prototype_truthful = (
                                self.ema_decay * self.prototype_truthful + 
                                (1 - self.ema_decay) * x
                            )
                        else:
                            self.prototype_hallucinated = (
                                self.ema_decay * self.prototype_hallucinated + 
                                (1 - self.ema_decay) * x
                            )
                
                epoch_loss += loss.item()
            
            total_loss = epoch_loss / len(dataloader)
        
        # Stage 2: 伪标签增强 (可选)
        if self.epochs_stage2 > 0:
            for epoch in range(self.epochs_stage2):
                # 计算当前预测
                with torch.no_grad():
                    steered = self._compute_steered_embedding(X)
                    logits = self._compute_vmf_logits(steered)
                    probs = F.softmax(logits, dim=-1)
                    
                    # Sinkhorn 分配
                    soft_labels = sinkhorn_knopp(
                        logits, 
                        epsilon=self.sinkhorn_epsilon, 
                        n_iters=self.sinkhorn_iters
                    )
                    
                    # 高置信度样本
                    confidence = probs.max(dim=-1).values
                    high_conf_mask = confidence > self.confidence_threshold
                    
                    if high_conf_mask.sum() > 0:
                        aug_features = X[high_conf_mask]
                        aug_labels = soft_labels[high_conf_mask].argmax(dim=-1)
                        
                        combined_X = torch.cat([X, aug_features], dim=0)
                        combined_y = torch.cat([y, aug_labels], dim=0)
                        
                        aug_dataset = TensorDataset(combined_X, combined_y)
                        aug_loader = DataLoader(aug_dataset, batch_size=self.batch_size, shuffle=True)
                        
                        for batch_X, batch_y in aug_loader:
                            optimizer.zero_grad()
                            steered = self._compute_steered_embedding(batch_X)
                            logits = self._compute_vmf_logits(steered)
                            loss = F.cross_entropy(logits, batch_y)
                            loss.backward()
                            optimizer.step()
        
        self.is_fitted = True
        return {'train_loss': total_loss}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测 hallucination 概率。"""
        if not self.is_fitted:
            raise ValueError("Detector not fitted")
        
        X = torch.from_numpy(X).float().to(self.device)
        
        with torch.no_grad():
            steered = self._compute_steered_embedding(X)
            logits = self._compute_vmf_logits(steered)
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1].cpu().numpy()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取检测器状态字典（用于保存）。"""
        return {
            'config': self.config,
            'tsv_vector': self.tsv_vector.detach().cpu() if self.tsv_vector is not None else None,
            'prototype_truthful': self.prototype_truthful.detach().cpu() if self.prototype_truthful is not None else None,
            'prototype_hallucinated': self.prototype_hallucinated.detach().cpu() if self.prototype_hallucinated is not None else None,
            'hidden_dim': self.hidden_dim,
            'selected_layers': self.selected_layers,
            'is_fitted': self.is_fitted,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从状态字典加载检测器。"""
        self.config = state['config']
        self.hidden_dim = state['hidden_dim']
        self.selected_layers = state['selected_layers']
        self.is_fitted = state['is_fitted']
        
        if state['tsv_vector'] is not None:
            self.tsv_vector = nn.Parameter(state['tsv_vector'].to(self.device))
        if state['prototype_truthful'] is not None:
            self.prototype_truthful = state['prototype_truthful'].to(self.device)
        if state['prototype_hallucinated'] is not None:
            self.prototype_hallucinated = state['prototype_hallucinated'].to(self.device)


# =============================================================================
# TSV 方法 (继承 BaseMethod) - 修复版本
# =============================================================================

@METHODS.register("tsv", aliases=["truthfulness_separator_vector", "steering_vector"])
class TSVMethod(BaseMethod):
    """TSV 幻觉检测方法 - 统一保存格式修复版。"""
    
    PAPER_SEED = 42
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        
        detector_config = {
            'steering_strength': params.get('steering_strength', 5.0),
            'kappa': params.get('kappa', 10.0),
            'layer_selection': params.get('layer_selection', 'middle'),
            'specific_layers': params.get('specific_layers', None),
            'learning_rate': params.get('learning_rate', 5e-3),
            'epochs_stage1': params.get('epochs_stage1', 20),
            'epochs_stage2': params.get('epochs_stage2', 20),
            'batch_size': params.get('batch_size', 128),
            'sinkhorn_iters': params.get('sinkhorn_iters', 3),
            'sinkhorn_epsilon': params.get('sinkhorn_epsilon', 0.05),
            'ema_decay': params.get('ema_decay', 0.99),
            'confidence_threshold': params.get('confidence_threshold', 0.9),
        }
        
        self.detector = TSVDetector(detector_config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 TSV 特征 (last-token embedding)。"""
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError(f"Sample {features.sample_id} has no hidden_states")
        
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        feat = self.detector._extract_last_token_embedding(
            hidden_states, hidden_states.shape[0]
        )
        
        if np.any(~np.isfinite(feat)):
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """训练 TSV。"""
        np.random.seed(self.PAPER_SEED)
        
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
                    feat.release_large_features()
            except Exception as e:
                logger.warning(f"Feature extraction failed for {feat.sample_id}: {e}")
        
        if len(X) == 0:
            raise ValueError("No valid features extracted")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"TSV: Training on {len(X)} samples, feature_dim={X.shape[1]}")
        self._feature_dim = X.shape[1]
        
        metrics = self.detector.fit(X, y)
        self.is_fitted = True
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """预测单个样本。"""
        if not self.is_fitted:
            raise MethodError("Method not fitted. Call fit() first.")
        
        x = self.extract_method_features(features)
        prob = self.detector.predict_proba(x.reshape(1, -1))[0]
        
        return Prediction(
            sample_id=features.sample_id,
            score=float(prob),
            label=1 if prob > 0.5 else 0,
            confidence=abs(prob - 0.5) * 2,
        )
    
    def predict_batch(self, features_list: List[ExtractedFeatures]) -> List[Prediction]:
        """批量预测。"""
        if not self.is_fitted:
            raise MethodError("Method not fitted. Call fit() first.")
        
        X = []
        valid_indices = []
        
        for i, feat in enumerate(features_list):
            try:
                x = self.extract_method_features(feat)
                if x is not None and not np.any(np.isnan(x)):
                    X.append(x)
                    valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Failed to predict for {feat.sample_id}: {e}")
        
        if len(X) == 0:
            return []
        
        X = np.array(X)
        probas = self.detector.predict_proba(X)
        
        predictions = []
        for idx, proba in zip(valid_indices, probas):
            feat = features_list[idx]
            predictions.append(Prediction(
                sample_id=feat.sample_id,
                score=float(proba),
                label=1 if proba > 0.5 else 0,
                confidence=abs(proba - 0.5) * 2,
            ))
        
        return predictions
    
    def save(self, path: Union[str, Path]) -> None:
        """保存模型到单个 .pkl 文件（统一格式）。
        
        ⚠️ 修复: 直接保存到 path（应为 model.pkl），而不是创建目录结构
        """
        if not self.is_fitted:
            raise MethodError("Cannot save unfitted method")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 统一保存格式：所有状态保存到单个文件
        state = {
            "config": self.config,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            # 检测器状态
            "detector_state": self.detector.get_state_dict(),
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved TSV method to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """从单个 .pkl 文件加载模型（统一格式）。
        
        ⚠️ 修复: 直接从 path 加载，支持新旧格式兼容
        """
        path = Path(path)
        
        # 兼容旧格式：如果 path 是目录或不存在，尝试查找 model.pkl
        if path.is_dir():
            # 旧格式：目录结构
            if (path / "method_state.pkl").exists():
                self._load_legacy_format(path)
                return
            elif (path / "model.pkl").exists():
                path = path / "model.pkl"
        
        if not path.exists():
            raise MethodError(f"Method file not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        # 检查是否是新格式
        if "detector_state" in state:
            # 新格式
            self.config = state["config"]
            self.is_fitted = state["is_fitted"]
            self._feature_dim = state["feature_dim"]
            self.detector.load_state_dict(state["detector_state"])
        else:
            # 可能是旧格式，尝试兼容
            self._load_legacy_state(state)
        
        logger.info(f"Loaded TSV method from {path}")
    
    def _load_legacy_format(self, directory: Path) -> None:
        """加载旧版目录格式。"""
        # 加载检测器
        detector_path = directory / "tsv_detector.pkl"
        if detector_path.exists():
            with open(detector_path, "rb") as f:
                detector_state = pickle.load(f)
            self.detector.load_state_dict(detector_state)
        
        # 加载方法状态
        method_state_path = directory / "method_state.pkl"
        if method_state_path.exists():
            with open(method_state_path, "rb") as f:
                state = pickle.load(f)
            self.config = state.get('config', self.config)
            self.is_fitted = state.get('is_fitted', False)
            self._feature_dim = state.get('feature_dim', None)
        
        logger.info(f"Loaded TSV method from legacy format: {directory}")
    
    def _load_legacy_state(self, state: Dict[str, Any]) -> None:
        """尝试从旧格式状态加载。"""
        self.config = state.get("config", self.config)
        self.is_fitted = state.get("is_fitted", False)
        self._feature_dim = state.get("feature_dim", None)
        
        # 如果包含检测器相关字段，尝试恢复
        if "tsv_vector" in state:
            self.detector.load_state_dict(state)