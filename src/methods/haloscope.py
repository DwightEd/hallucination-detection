"""
HaloScope - Harnessing Unlabeled LLM Generations for Hallucination Detection

基于 NeurIPS'24 论文: https://arxiv.org/abs/2409.17504
GitHub: https://github.com/deeplearning-wisc/haloscope

=============================================================================
修复版本 - 统一模型保存格式为 model.pkl
=============================================================================

核心设计:
1. 使用最后一个 token 的 embedding (last-token)
2. 使用中间层 (8-14层 for 32层模型)
3. SVD 分数使用投影平方: ζ_i = (1/k) Σ σ_j · ⟨f̄_i, v_j⟩²
4. 两层 MLP 分类器 (d -> 1024 -> 1)
5. 优化器: SGD + cosine schedule, lr=0.05, epochs=50
6. 随机种子: 固定为 41 (论文明确指出)
"""

from __future__ import annotations
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction, MethodError
from .base import BaseMethod

logger = logging.getLogger(__name__)

# =============================================================================
# 论文关键参数
# =============================================================================
HALOSCOPE_SEED = 41  # 论文: "It is set to 41 for all the experiments"
DEFAULT_LAYERS_32 = list(range(8, 15))  # 8-14 层 for 32层模型
DEFAULT_K = 5
DEFAULT_MLP_HIDDEN = 1024
DEFAULT_LR = 0.05
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 512
DEFAULT_WEIGHT_DECAY = 3e-4


# =============================================================================
# 两层 MLP 分类器 - 原论文架构
# =============================================================================

class HaloScopeMLP(nn.Module):
    """原始 HaloScope 两层 MLP: input_dim -> 1024 -> 1"""
    
    def __init__(self, input_dim: int, hidden_dim: int = DEFAULT_MLP_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =============================================================================
# HaloScope 检测器
# =============================================================================

class HaloScopeDetector:
    """HaloScope 检测器 - 严格复现原始论文。"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # SVD 参数
        self.n_components = config.get('n_components', 10)
        self.k = config.get('k', DEFAULT_K)
        self.weighted_svd = config.get('weighted_svd', True)
        self.center = config.get('center', True)
        
        # 层选择
        self.layer_selection = config.get('layer_selection', 'middle')
        self.specific_layers = config.get('specific_layers', None)
        
        # MLP 参数
        self.mlp_hidden_dim = config.get('mlp_hidden_dim', DEFAULT_MLP_HIDDEN)
        self.lr = config.get('learning_rate', DEFAULT_LR)
        self.epochs = config.get('epochs', DEFAULT_EPOCHS)
        self.batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
        self.weight_decay = config.get('weight_decay', DEFAULT_WEIGHT_DECAY)
        
        # 状态
        self.mlp = None
        self.mean = None
        self.singular_values = None
        self.right_singular_vectors = None
        self.selected_layers = None
        self.is_fitted = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _select_layers(self, num_layers: int) -> List[int]:
        """选择要使用的层。"""
        if self.layer_selection == 'specific' and self.specific_layers:
            return [l for l in self.specific_layers if 0 <= l < num_layers]
        elif self.layer_selection == 'middle':
            # 论文: 使用中间层 (8-14 for 32层模型)
            start = max(0, int(num_layers * 0.25))
            end = min(num_layers, int(num_layers * 0.45))
            n_layers = min(7, end - start)  # 最多7层
            return list(range(start, start + n_layers))
        elif self.layer_selection == 'all':
            return list(range(num_layers))
        else:
            return DEFAULT_LAYERS_32
    
    def _extract_last_token_embedding(self, hidden_states: np.ndarray) -> np.ndarray:
        """提取 last-token embedding (论文核心)。
        
        Args:
            hidden_states: [n_layers, seq_len, hidden_dim]
            
        Returns:
            [selected_layers * hidden_dim] 特征向量
        """
        num_layers = hidden_states.shape[0]
        
        if self.selected_layers is None:
            self.selected_layers = self._select_layers(num_layers)
        
        embeddings = []
        for layer_idx in self.selected_layers:
            if layer_idx < num_layers:
                emb = hidden_states[layer_idx, -1, :]  # last token
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            raise ValueError("No valid layers selected")
        
        return np.concatenate(embeddings, axis=0)
    
    def _compute_svd_scores(self, features: np.ndarray) -> np.ndarray:
        """计算 SVD 分数 (论文公式)。
        
        ζ_i = (1/k) Σ_{j=1}^{k} σ_j · ⟨f̄_i, v_j⟩²
        
        Args:
            features: [N, d] 特征矩阵 (已中心化)
            
        Returns:
            [N, k] SVD 分数矩阵
        """
        if self.right_singular_vectors is None:
            return features[:, :self.k]
        
        # 投影到右奇异向量
        projections = features @ self.right_singular_vectors.T  # [N, k]
        
        # 投影平方（论文关键）
        proj_squared = projections ** 2  # [N, k]
        
        # 加权
        if self.weighted_svd and self.singular_values is not None:
            weights = self.singular_values[:self.k]
            weights = weights / weights.sum()  # 归一化
            proj_squared = proj_squared * weights
        
        return proj_squared
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练 HaloScope。
        
        Args:
            X: [N, d] 特征矩阵
            y: [N] 标签
            
        Returns:
            训练指标
        """
        torch.manual_seed(HALOSCOPE_SEED)
        np.random.seed(HALOSCOPE_SEED)
        
        # 中心化
        if self.center:
            self.mean = X.mean(axis=0)
            X_centered = X - self.mean
        else:
            self.mean = np.zeros(X.shape[1])
            X_centered = X
        
        # SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.singular_values = s[:self.n_components]
        self.right_singular_vectors = Vt[:self.k]  # [k, d]
        
        # 计算 SVD 分数
        svd_scores = self._compute_svd_scores(X_centered)  # [N, k]
        
        # 训练 MLP
        input_dim = svd_scores.shape[1]
        self.mlp = HaloScopeMLP(input_dim, self.mlp_hidden_dim).to(self.device)
        
        X_tensor = torch.from_numpy(svd_scores).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # SGD + Cosine schedule (论文设置)
        optimizer = optim.SGD(
            self.mlp.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        self.mlp.train()
        total_loss = 0.0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.mlp(batch_X).squeeze()
                loss = F.binary_cross_entropy_with_logits(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            total_loss = epoch_loss / len(dataloader)
        
        self.is_fitted = True
        return {'train_loss': total_loss}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测 hallucination 概率。"""
        if not self.is_fitted:
            raise ValueError("Detector not fitted")
        
        # 中心化
        if self.mean is not None:
            X_centered = X - self.mean
        else:
            X_centered = X
        
        # SVD 分数
        svd_scores = self._compute_svd_scores(X_centered)
        
        X_tensor = torch.from_numpy(svd_scores).float().to(self.device)
        
        self.mlp.eval()
        with torch.no_grad():
            outputs = self.mlp(X_tensor).squeeze()
            probs = torch.sigmoid(outputs)
        
        return probs.cpu().numpy()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取检测器状态字典。"""
        return {
            'config': self.config,
            'mlp_state_dict': self.mlp.state_dict() if self.mlp else None,
            'mean': self.mean,
            'singular_values': self.singular_values,
            'right_singular_vectors': self.right_singular_vectors,
            'selected_layers': self.selected_layers,
            'is_fitted': self.is_fitted,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从状态字典加载检测器。"""
        self.config = state['config']
        self.mean = state['mean']
        self.singular_values = state['singular_values']
        self.right_singular_vectors = state['right_singular_vectors']
        self.selected_layers = state['selected_layers']
        self.is_fitted = state['is_fitted']
        
        if state['mlp_state_dict'] is not None:
            # 重建 MLP
            input_dim = state['right_singular_vectors'].shape[0] if state['right_singular_vectors'] is not None else self.k
            self.mlp = HaloScopeMLP(input_dim, self.mlp_hidden_dim).to(self.device)
            self.mlp.load_state_dict(state['mlp_state_dict'])


# =============================================================================
# HaloScope Method 类 - 修复版本
# =============================================================================

@METHODS.register("haloscope", aliases=["halo_scope", "svd_hallucination"])
class HaloScopeMethod(BaseMethod):
    """HaloScope 幻觉检测方法 - 统一保存格式修复版。"""
    
    PAPER_SEED = HALOSCOPE_SEED
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        
        detector_config = {
            'n_components': params.get('svd_config', {}).get('n_components', 10),
            'k': params.get('svd_config', {}).get('k', DEFAULT_K),
            'weighted_svd': params.get('detection', {}).get('weighted_svd', True),
            'center': params.get('svd_config', {}).get('center', True),
            'layer_selection': params.get('detection', {}).get('layer_selection', 'middle'),
            'specific_layers': params.get('detection', {}).get('specific_layers', None),
            'mlp_hidden_dim': params.get('mlp_config', {}).get('hidden_dim', DEFAULT_MLP_HIDDEN),
            'learning_rate': params.get('mlp_config', {}).get('learning_rate', DEFAULT_LR),
            'epochs': params.get('mlp_config', {}).get('epochs', DEFAULT_EPOCHS),
            'batch_size': params.get('mlp_config', {}).get('batch_size', DEFAULT_BATCH_SIZE),
            'weight_decay': params.get('mlp_config', {}).get('weight_decay', DEFAULT_WEIGHT_DECAY),
        }
        
        self.detector = HaloScopeDetector(detector_config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 HaloScope 特征。"""
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError(f"Sample {features.sample_id} has no hidden_states")
        
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        feat = self.detector._extract_last_token_embedding(hidden_states)
        
        if np.any(~np.isfinite(feat)):
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """训练 HaloScope。"""
        np.random.seed(self.PAPER_SEED)
        torch.manual_seed(self.PAPER_SEED)
        
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
        
        logger.info(f"HaloScope: Training on {len(X)} samples, feature_dim={X.shape[1]}")
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
        
        ⚠️ 修复: 直接保存到 path（应为 model.pkl）
        """
        if not self.is_fitted:
            raise MethodError("Cannot save unfitted method")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            "detector_state": self.detector.get_state_dict(),
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved HaloScope method to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """从单个 .pkl 文件加载模型（统一格式）。"""
        path = Path(path)
        
        # 兼容旧格式
        if path.is_dir():
            if (path / "model.pkl").exists():
                path = path / "model.pkl"
            elif (path / "haloscope_detector.pkl").exists():
                self._load_legacy_format(path)
                return
        
        if not path.exists():
            raise MethodError(f"Method file not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        if "detector_state" in state:
            # 新格式
            self.config = state["config"]
            self.is_fitted = state["is_fitted"]
            self._feature_dim = state["feature_dim"]
            self.detector.load_state_dict(state["detector_state"])
        else:
            # 尝试旧格式
            self._load_legacy_state(state)
        
        logger.info(f"Loaded HaloScope method from {path}")
    
    def _load_legacy_format(self, directory: Path) -> None:
        """加载旧版目录格式。"""
        detector_path = directory / "haloscope_detector.pkl"
        if detector_path.exists():
            with open(detector_path, "rb") as f:
                detector_state = pickle.load(f)
            self.detector.load_state_dict(detector_state)
        
        method_state_path = directory / "method_state.pkl"
        if method_state_path.exists():
            with open(method_state_path, "rb") as f:
                state = pickle.load(f)
            self.config = state.get('config', self.config)
            self.is_fitted = state.get('is_fitted', False)
            self._feature_dim = state.get('feature_dim', None)
        
        logger.info(f"Loaded HaloScope from legacy format: {directory}")
    
    def _load_legacy_state(self, state: Dict[str, Any]) -> None:
        """尝试从旧格式状态加载。"""
        self.config = state.get("config", self.config)
        self.is_fitted = state.get("is_fitted", False)
        self._feature_dim = state.get("feature_dim", None)
        
        # 如果包含检测器相关字段
        if "mlp_state_dict" in state or "mean" in state:
            self.detector.load_state_dict(state)