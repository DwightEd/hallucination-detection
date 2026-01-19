"""
TSV (Truthfulness Separator Vector) - Steer LLM Latents for Hallucination Detection

基于 ICML 2025 论文: "Steer LLM Latents for Hallucination Detection"
论文链接: https://arxiv.org/abs/2503.01917
GitHub: https://github.com/deeplearning-wisc/tsv

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
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction
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
    """
    Sinkhorn-Knopp 算法用于最优传输
    
    将 softmax scores 转换为满足行列约束的软分配矩阵
    
    Args:
        scores: [N, K] 原始分数矩阵
        epsilon: 正则化参数
        n_iters: 迭代次数
        
    Returns:
        [N, K] 软分配矩阵
    """
    Q = torch.exp(scores / epsilon)
    Q /= Q.sum()
    
    N, K = Q.shape
    
    for _ in range(n_iters):
        # 行归一化
        Q /= Q.sum(dim=1, keepdim=True)
        # 列归一化 (均衡分配)
        Q /= Q.sum(dim=0, keepdim=True)
    
    # 最终行归一化
    Q /= Q.sum(dim=1, keepdim=True)
    
    return Q


# =============================================================================
# TSV 核心检测器
# =============================================================================

class TSVDetector:
    """
    Truthfulness Separator Vector 检测器
    
    核心组件:
    1. TSV 向量 v ∈ R^d: 可学习的 steering 向量
    2. 类别原型 μ_0, μ_1: truthful 和 hallucinated 的原型向量
    3. vMF 分布: 用于计算类别概率
    
    训练流程:
    1. Stage 1: 在标注数据上训练 TSV 和原型
    2. Stage 2: 使用最优传输分配伪标签，增强训练
    """
    
    # 论文默认随机种子
    DEFAULT_SEED = 42
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 TSV 检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # TSV 参数
        self.steering_strength = config.get('steering_strength', 5.0)  # λ
        self.kappa = config.get('kappa', 10.0)  # vMF 浓度参数
        
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
        self.tsv_vector = None  # v ∈ R^d
        self.prototype_truthful = None  # μ_0
        self.prototype_hallucinated = None  # μ_1
        
        # 训练状态
        self.hidden_dim = None
        self.selected_layers = None
        self.is_fitted = False
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _select_layers(self, num_layers: int) -> List[int]:
        """
        选择要使用的层
        
        论文发现早中期层 (25%-50%) 效果最好
        """
        if self.layer_selection == 'specific' and self.specific_layers:
            return [l for l in self.specific_layers if 0 <= l < num_layers]
        elif self.layer_selection == 'middle':
            start = max(0, int(num_layers * 0.25))
            end = min(num_layers, int(num_layers * 0.5))
            return list(range(start, end))
        elif self.layer_selection == 'all':
            return list(range(num_layers))
        else:
            # 默认使用中间层
            start = max(0, int(num_layers * 0.25))
            end = min(num_layers, int(num_layers * 0.5))
            return list(range(start, end))
    
    def _extract_last_token_embedding(
        self,
        hidden_states: np.ndarray,
        num_layers: int,
    ) -> np.ndarray:
        """
        提取最后一个 token 的 embedding
        
        Args:
            hidden_states: [num_layers, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
            num_layers: 模型总层数
            
        Returns:
            [len(selected_layers) * hidden_dim] 展平的特征向量
        """
        if len(hidden_states.shape) == 2:
            # [seq_len, hidden_dim] -> 取最后一个 token
            return hidden_states[-1, :].flatten()
        
        elif len(hidden_states.shape) == 3:
            # [num_layers, seq_len, hidden_dim]
            if self.selected_layers is None:
                self.selected_layers = self._select_layers(hidden_states.shape[0])
            
            # 取每层的最后一个 token
            selected = hidden_states[self.selected_layers, -1, :]
            return selected.flatten()
        
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
    
    def _compute_steered_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        计算 steered embedding
        
        h_steered = h + λ * v
        
        Args:
            embedding: [batch, d] 原始 embedding
            
        Returns:
            [batch, d] steered embedding
        """
        if self.tsv_vector is None:
            return embedding
        
        return embedding + self.steering_strength * self.tsv_vector
    
    def _compute_vmf_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算 vMF 分布的 logits
        
        logit_c = κ * cos_sim(h, μ_c)
        
        Args:
            embeddings: [batch, d] steered embeddings
            
        Returns:
            [batch, 2] logits for [truthful, hallucinated]
        """
        # L2 归一化
        embeddings_norm = F.normalize(embeddings, dim=-1)
        proto_truthful_norm = F.normalize(self.prototype_truthful.unsqueeze(0), dim=-1)
        proto_hallucinated_norm = F.normalize(self.prototype_hallucinated.unsqueeze(0), dim=-1)
        
        # 计算余弦相似度
        sim_truthful = torch.sum(embeddings_norm * proto_truthful_norm, dim=-1)
        sim_hallucinated = torch.sum(embeddings_norm * proto_hallucinated_norm, dim=-1)
        
        # 乘以浓度参数
        logits = torch.stack([
            self.kappa * sim_truthful,
            self.kappa * sim_hallucinated
        ], dim=-1)
        
        return logits
    
    def _update_prototypes_ema(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        使用 EMA 更新类别原型
        
        Args:
            embeddings: [batch, d] steered embeddings
            labels: [batch] 标签 (0=truthful, 1=hallucinated)
        """
        with torch.no_grad():
            # 计算当前批次的类别均值
            mask_truthful = (labels == 0)
            mask_hallucinated = (labels == 1)
            
            if mask_truthful.sum() > 0:
                mean_truthful = embeddings[mask_truthful].mean(dim=0)
                self.prototype_truthful = (
                    self.ema_decay * self.prototype_truthful + 
                    (1 - self.ema_decay) * mean_truthful
                )
            
            if mask_hallucinated.sum() > 0:
                mean_hallucinated = embeddings[mask_hallucinated].mean(dim=0)
                self.prototype_hallucinated = (
                    self.ema_decay * self.prototype_hallucinated + 
                    (1 - self.ema_decay) * mean_hallucinated
                )
    
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        unlabeled_features: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        两阶段训练
        
        Args:
            features: [N, d] 标注数据的特征
            labels: [N] 标签 (0=truthful, 1=hallucinated)
            unlabeled_features: [M, d] 无标签数据的特征 (可选)
            
        Returns:
            训练指标
        """
        features = torch.from_numpy(features).float().to(self.device)
        labels = torch.from_numpy(labels).long().to(self.device)
        
        self.hidden_dim = features.shape[1]
        
        # 初始化可学习参数
        self.tsv_vector = nn.Parameter(
            torch.randn(self.hidden_dim, device=self.device) * 0.01
        )
        
        # 初始化原型 (使用类别均值)
        mask_truthful = (labels == 0)
        mask_hallucinated = (labels == 1)
        
        if mask_truthful.sum() > 0:
            self.prototype_truthful = features[mask_truthful].mean(dim=0).detach().clone()
        else:
            self.prototype_truthful = torch.zeros(self.hidden_dim, device=self.device)
        
        if mask_hallucinated.sum() > 0:
            self.prototype_hallucinated = features[mask_hallucinated].mean(dim=0).detach().clone()
        else:
            self.prototype_hallucinated = torch.zeros(self.hidden_dim, device=self.device)
        
        # 优化器
        optimizer = torch.optim.AdamW([self.tsv_vector], lr=self.lr)
        
        # Stage 1: 初始训练
        logger.info(f"TSV Stage 1: Training on {len(features)} labeled samples")
        
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs_stage1):
            total_loss = 0
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                
                # 计算 steered embedding
                steered = self._compute_steered_embedding(batch_features)
                
                # 计算 vMF logits
                logits = self._compute_vmf_logits(steered)
                
                # 交叉熵损失
                loss = F.cross_entropy(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                # 更新原型 (EMA)
                with torch.no_grad():
                    self._update_prototypes_ema(steered.detach(), batch_labels)
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.debug(f"Stage 1 Epoch {epoch+1}/{self.epochs_stage1}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Stage 2: 增强训练 (如果有无标签数据)
        if unlabeled_features is not None and len(unlabeled_features) > 0:
            logger.info(f"TSV Stage 2: Augmented training with {len(unlabeled_features)} unlabeled samples")
            
            unlabeled = torch.from_numpy(unlabeled_features).float().to(self.device)
            
            for epoch in range(self.epochs_stage2):
                # 为无标签数据分配伪标签
                with torch.no_grad():
                    steered_unlabeled = self._compute_steered_embedding(unlabeled)
                    logits_unlabeled = self._compute_vmf_logits(steered_unlabeled)
                    probs_unlabeled = F.softmax(logits_unlabeled, dim=-1)
                    
                    # Sinkhorn 归一化
                    pseudo_probs = sinkhorn_knopp(
                        logits_unlabeled,
                        epsilon=self.sinkhorn_epsilon,
                        n_iters=self.sinkhorn_iters,
                    )
                    
                    # 置信度过滤
                    max_probs, pseudo_labels = pseudo_probs.max(dim=-1)
                    confident_mask = max_probs > self.confidence_threshold
                    
                    if confident_mask.sum() > 0:
                        confident_features = unlabeled[confident_mask]
                        confident_labels = pseudo_labels[confident_mask]
                        
                        # 合并标注数据和伪标签数据
                        combined_features = torch.cat([features, confident_features], dim=0)
                        combined_labels = torch.cat([labels, confident_labels], dim=0)
                    else:
                        combined_features = features
                        combined_labels = labels
                
                # 在合并数据上训练
                combined_dataset = TensorDataset(combined_features, combined_labels)
                combined_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True)
                
                total_loss = 0
                for batch_features, batch_labels in combined_loader:
                    optimizer.zero_grad()
                    
                    steered = self._compute_steered_embedding(batch_features)
                    logits = self._compute_vmf_logits(steered)
                    loss = F.cross_entropy(logits, batch_labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        self._update_prototypes_ema(steered.detach(), batch_labels)
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 5 == 0:
                    logger.debug(f"Stage 2 Epoch {epoch+1}/{self.epochs_stage2}, Loss: {total_loss/len(combined_loader):.4f}")
        
        self.is_fitted = True
        
        # 计算训练集上的准确率
        with torch.no_grad():
            steered = self._compute_steered_embedding(features)
            logits = self._compute_vmf_logits(steered)
            preds = logits.argmax(dim=-1)
            train_acc = (preds == labels).float().mean().item()
        
        return {
            'train_accuracy': train_acc,
            'n_samples': len(features),
        }
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        预测 hallucination 概率
        
        Args:
            features: [N, d] 特征
            
        Returns:
            [N] hallucination 概率
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted")
        
        features = torch.from_numpy(features).float().to(self.device)
        
        with torch.no_grad():
            steered = self._compute_steered_embedding(features)
            logits = self._compute_vmf_logits(steered)
            probs = F.softmax(logits, dim=-1)
            
            # 返回 hallucination (class 1) 的概率
            return probs[:, 1].cpu().numpy()
    
    def save(self, path: Path):
        """保存检测器状态"""
        state = {
            'config': self.config,
            'tsv_vector': self.tsv_vector.detach().cpu() if self.tsv_vector is not None else None,
            'prototype_truthful': self.prototype_truthful.detach().cpu() if self.prototype_truthful is not None else None,
            'prototype_hallucinated': self.prototype_hallucinated.detach().cpu() if self.prototype_hallucinated is not None else None,
            'hidden_dim': self.hidden_dim,
            'selected_layers': self.selected_layers,
            'is_fitted': self.is_fitted,
        }
        
        with open(path / 'tsv_detector.pkl', 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path):
        """加载检测器状态"""
        with open(path / 'tsv_detector.pkl', 'rb') as f:
            state = pickle.load(f)
        
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
# TSV 方法 (继承 BaseMethod)
# =============================================================================

@METHODS.register("tsv", aliases=["truthfulness_separator_vector", "steering_vector"])
class TSVMethod(BaseMethod):
    """
    TSV (Truthfulness Separator Vector) 幻觉检测方法
    
    特点:
    - 使用 hidden states 的 last-token embedding
    - 学习 steering vector 重塑表示空间
    - 支持无标签数据的半监督学习
    - Sample-level 检测
    
    配置示例:
        method:
          name: tsv
          params:
            steering_strength: 5.0      # λ, steering 强度
            kappa: 10.0                 # vMF 浓度参数
            layer_selection: middle     # 层选择策略
            learning_rate: 0.005
            epochs_stage1: 20
            epochs_stage2: 20
    """
    
    # 论文随机种子
    PAPER_SEED = 42
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        # 构建检测器配置
        params = self.config.params or {}
        
        detector_config = {
            # TSV 核心参数
            'steering_strength': params.get('steering_strength', 5.0),
            'kappa': params.get('kappa', 10.0),
            
            # 层选择
            'layer_selection': params.get('layer_selection', 'middle'),
            'specific_layers': params.get('specific_layers', None),
            
            # 训练参数
            'learning_rate': params.get('learning_rate', 5e-3),
            'epochs_stage1': params.get('epochs_stage1', 20),
            'epochs_stage2': params.get('epochs_stage2', 20),
            'batch_size': params.get('batch_size', 128),
            
            # Sinkhorn 参数
            'sinkhorn_iters': params.get('sinkhorn_iters', 3),
            'sinkhorn_epsilon': params.get('sinkhorn_epsilon', 0.05),
            
            # EMA
            'ema_decay': params.get('ema_decay', 0.99),
            
            # 置信度阈值
            'confidence_threshold': params.get('confidence_threshold', 0.9),
        }
        
        self.detector = TSVDetector(detector_config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """
        从 ExtractedFeatures 提取 TSV 所需的特征
        
        关键: 使用 last-token embedding
        
        Args:
            features: 提取的特征
            
        Returns:
            [feature_dim] 特征向量
        """
        # 获取 hidden states (支持懒加载)
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError(
                f"Sample {features.sample_id} has no hidden_states. "
                "Make sure hidden_states are extracted and available."
            )
        
        # 转换为 numpy
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        # 处理 NaN/Inf
        if np.any(~np.isfinite(hidden_states)):
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 提取 last-token embedding
        num_layers = hidden_states.shape[0] if len(hidden_states.shape) == 3 else 1
        embedding = self.detector._extract_last_token_embedding(hidden_states, num_layers)
        
        # 处理结果中的 NaN/Inf
        if np.any(~np.isfinite(embedding)):
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return embedding.astype(np.float32)
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
        unlabeled_features_list: Optional[List[ExtractedFeatures]] = None,
    ) -> Dict[str, float]:
        """
        训练 TSV 方法
        
        Args:
            features_list: 标注数据的特征列表
            labels: 标签列表
            cv: 是否进行交叉验证 (TSV 不支持标准 CV)
            unlabeled_features_list: 无标签数据的特征列表 (可选)
            
        Returns:
            训练指标
        """
        # 设置随机种子
        np.random.seed(self.PAPER_SEED)
        torch.manual_seed(self.PAPER_SEED)
        
        # 提取特征
        X = []
        y = []
        
        logger.info(f"Extracting features from {len(features_list)} samples...")
        
        for i, feat in enumerate(features_list):
            try:
                x = self.extract_method_features(feat)
                if x is not None and not np.any(np.isnan(x)):
                    X.append(x)
                    label = labels[i] if labels else feat.label
                    if label is not None:
                        y.append(label)
                    
                # 释放大特征
                feat.release_large_features()
                
            except Exception as e:
                logger.warning(f"Failed to extract features for {feat.sample_id}: {e}")
        
        if len(X) == 0:
            raise ValueError("No valid features extracted")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training TSV on {len(X)} samples, feature dim={X.shape[1]}")
        self._feature_dim = X.shape[1]
        
        # 处理无标签数据
        X_unlabeled = None
        if unlabeled_features_list:
            logger.info(f"Processing {len(unlabeled_features_list)} unlabeled samples...")
            X_unlabeled_list = []
            
            for feat in unlabeled_features_list:
                try:
                    x = self.extract_method_features(feat)
                    if x is not None and not np.any(np.isnan(x)):
                        X_unlabeled_list.append(x)
                    feat.release_large_features()
                except Exception as e:
                    pass
            
            if X_unlabeled_list:
                X_unlabeled = np.array(X_unlabeled_list)
        
        # 训练检测器
        metrics = self.detector.fit(X, y, X_unlabeled)
        
        self.is_fitted = True
        
        metrics['n_samples'] = len(X)
        metrics['n_positive'] = int(y.sum())
        metrics['n_negative'] = int(len(y) - y.sum())
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """
        预测单个样本的 hallucination 概率
        
        Args:
            features: 提取的特征
            
        Returns:
            Prediction 对象
        """
        if not self.is_fitted:
            raise ValueError("Method not fitted. Call fit() first.")
        
        x = self.extract_method_features(features)
        proba = self.detector.predict_proba(x.reshape(1, -1))[0]
        
        return Prediction(
            sample_id=features.sample_id,
            score=float(proba),
            label=1 if proba > 0.5 else 0,
            confidence=abs(proba - 0.5) * 2,
        )
    
    def predict_batch(self, features_list: List[ExtractedFeatures]) -> List[Prediction]:
        """
        批量预测
        
        Args:
            features_list: 特征列表
            
        Returns:
            Prediction 列表
        """
        if not self.is_fitted:
            raise ValueError("Method not fitted. Call fit() first.")
        
        # 批量提取特征
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
    
    def save(self, path: Path) -> None:
        """保存方法"""
        path = Path(path)
        
        # 使用目录存储
        if path.suffix:
            save_dir = path.parent / path.stem
        else:
            save_dir = path
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存检测器
        self.detector.save(save_dir)
        
        # 保存方法状态
        state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'feature_dim': self._feature_dim,
        }
        
        with open(save_dir / 'method_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved TSV method to {save_dir}")
    
    def load(self, path: Path) -> None:
        """加载方法"""
        path = Path(path)
        
        if path.is_file():
            load_dir = path.parent / path.stem
        else:
            load_dir = path
        
        # 加载检测器
        self.detector.load(load_dir)
        
        # 加载方法状态
        method_state_path = load_dir / 'method_state.pkl'
        if method_state_path.exists():
            with open(method_state_path, 'rb') as f:
                state = pickle.load(f)
            self.config = state['config']
            self.is_fitted = state['is_fitted']
            self._feature_dim = state['feature_dim']
        
        logger.info(f"Loaded TSV method from {load_dir}")