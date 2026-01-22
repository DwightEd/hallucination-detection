"""LapEigvals: Laplacian Eigenvalue-based Hallucination Detection.

基于 EMNLP 2025 论文: "Hallucination Detection in LLMs Using Spectral Features of Attention Maps"
GitHub: https://github.com/graphml-lab-pwr/lapeigvals

=============================================================================
修复版本: 解决 BFloat16 转换问题
=============================================================================

核心算法 (严格按照论文实现):
1. 将注意力矩阵 A 视为有向图的邻接矩阵
2. 定义出度矩阵 D: d_ii = Σ_{u>i} a_ui / (T - i)  [式(2)]
3. Laplacian: L = D - A  [式(1)]
4. 由于 L 是下三角矩阵，特征值 = 对角线元素: λ_i = d_ii - a_ii  [式(3)]
5. 对特征值排序取 top-k，拼接所有层和头  [式(4)]
6. PCA 降维到 512 维 + LogisticRegression
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import pickle
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction, MethodError
from .base import BaseMethod

logger = logging.getLogger(__name__)


# =============================================================================
# 辅助函数: 安全的张量转 NumPy
# =============================================================================

def safe_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """安全地将张量转换为 NumPy 数组，处理 BFloat16 等不支持的类型。
    
    Args:
        tensor: PyTorch 张量或 NumPy 数组
        
    Returns:
        NumPy float32 数组
    """
    if tensor is None:
        return None
    
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor
    
    if isinstance(tensor, torch.Tensor):
        # 关键修复: 先转为 float32，再转 numpy
        # BFloat16 不被 numpy 直接支持
        return tensor.detach().cpu().float().numpy()
    
    # 其他类型尝试直接转换
    return np.asarray(tensor, dtype=np.float32)


# =============================================================================
# 核心算法 - 严格按照原论文公式实现
# =============================================================================

def compute_laplacian_eigenvalues_from_attention(
    attention: np.ndarray,
    top_k: int = 100,
) -> np.ndarray:
    """从单个注意力头计算 Laplacian 特征值 (即对角线元素)。
    
    论文公式:
    - Laplacian: L = D - A, 其中 D 是出度矩阵
    - 出度: d_ii = Σ_{u>i} a_ui / (T - i)  [式(2)]
    - 由于 L 是下三角, 特征值 = 对角线: λ_i = d_ii - a_ii  [式(3)]
    
    Args:
        attention: 注意力矩阵 [seq_len, seq_len], 下三角, 行和为1
        top_k: 返回的 top-k 特征值数量
        
    Returns:
        Top-k 特征值 (降序排列), shape [k]
    """
    # 安全转换
    attention = safe_to_numpy(attention)
    attention = np.asarray(attention, dtype=np.float64)
    T = attention.shape[0]
    
    if T == 0:
        return np.zeros(top_k, dtype=np.float32)
    
    # 计算出度 d_ii = Σ_{u>i} a_ui / (T - i)
    # 论文公式(2): 后续token u (u > i) 对token i 的注意力之和，除以 (T - i)
    out_degrees = np.zeros(T, dtype=np.float64)
    for i in range(T):
        # 后续 tokens: 索引从 i+1 到 T-1
        subsequent_attn_sum = attention[i+1:, i].sum() if i < T - 1 else 0.0
        # 分母: (T - i)，按论文公式
        denominator = T - i
        out_degrees[i] = subsequent_attn_sum / denominator if denominator > 0 else 0.0
    
    # 注意力对角线 a_ii
    attn_diag = np.diag(attention)
    
    # Laplacian 特征值 λ_i = d_ii - a_ii
    eigenvalues = out_degrees - attn_diag
    
    # 按降序排序
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    
    # 取 top-k
    k = min(top_k, len(sorted_eigenvalues))
    top_eigenvalues = sorted_eigenvalues[:k]
    
    # 如果不足 k 个，用 0 填充
    if len(top_eigenvalues) < top_k:
        top_eigenvalues = np.concatenate([
            top_eigenvalues,
            np.zeros(top_k - len(top_eigenvalues), dtype=np.float64)
        ])
    
    return top_eigenvalues.astype(np.float32)


def compute_laplacian_eigenvalues_batch(
    attention: np.ndarray,  # [n_heads, seq_len, seq_len]
    top_k: int = 100,
) -> np.ndarray:
    """批量计算多个头的 Laplacian 特征值。
    
    Args:
        attention: [n_heads, seq_len, seq_len]
        top_k: 每个头的 top-k 特征值数量
        
    Returns:
        [n_heads, k] 的特征值矩阵
    """
    # 安全转换
    attention = safe_to_numpy(attention)
    n_heads = attention.shape[0]
    eigenvalues = np.zeros((n_heads, top_k), dtype=np.float32)
    
    for h in range(n_heads):
        eigenvalues[h] = compute_laplacian_eigenvalues_from_attention(
            attention[h], top_k
        )
    
    return eigenvalues


def extract_lapeigvals_features(
    full_attention: Union[np.ndarray, torch.Tensor],  # [n_layers, n_heads, seq_len, seq_len]
    prompt_len: int,
    response_len: int,
    top_k: int = 100,
    response_only: bool = True,
) -> np.ndarray:
    """从完整注意力矩阵提取 LapEigvals 特征。
    
    按论文式(4): 拼接所有层和头的 top-k 特征值。
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        top_k: 每个 (layer, head) 的 top-k 特征值
        response_only: 是否只使用 response 部分的注意力
        
    Returns:
        特征向量 [n_layers * n_heads * top_k]
    """
    # 安全转换 - 修复 BFloat16 问题
    full_attention = safe_to_numpy(full_attention)
    
    n_layers, n_heads, seq_len, _ = full_attention.shape
    
    # 确定 response 范围
    if response_only and response_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len)
    else:
        start_idx = 0
        end_idx = seq_len
    
    all_eigenvalues = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # 提取 response 部分的注意力
            attn = full_attention[layer, head, start_idx:end_idx, start_idx:end_idx]
            
            if attn.shape[0] == 0:
                # 空序列，返回零
                all_eigenvalues.append(np.zeros(top_k, dtype=np.float32))
                continue
            
            # 计算特征值
            eigvals = compute_laplacian_eigenvalues_from_attention(attn, top_k)
            all_eigenvalues.append(eigvals)
    
    return np.concatenate(all_eigenvalues)


def extract_features_from_precomputed_laplacian_diag(
    laplacian_diag: Union[np.ndarray, torch.Tensor],  # [n_layers, n_heads, seq_len]
    prompt_len: int,
    response_len: int,
    top_k: int = 100,
    response_only: bool = True,
) -> np.ndarray:
    """从预计算的 Laplacian 对角线提取特征。
    
    当已经预计算了 laplacian_diags 时使用此函数，
    直接从对角线值提取 top-k 特征。
    
    Args:
        laplacian_diag: [n_layers, n_heads, seq_len] 预计算的 Laplacian 对角线
        prompt_len: Prompt 长度
        response_len: Response 长度
        top_k: 每个 (layer, head) 的 top-k 特征值
        response_only: 是否只使用 response 部分
        
    Returns:
        特征向量
    """
    # 安全转换 - 修复 BFloat16 问题
    laplacian_diag = safe_to_numpy(laplacian_diag)
    
    n_layers, n_heads, seq_len = laplacian_diag.shape
    
    # 确定范围
    if response_only and response_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len)
    else:
        start_idx = 0
        end_idx = seq_len
    
    actual_k = min(top_k, end_idx - start_idx)
    
    all_eigenvalues = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # 获取 response 部分的对角线值
            eigvals = laplacian_diag[layer, head, start_idx:end_idx]
            
            # 排序取 top-k
            sorted_eigvals = np.sort(eigvals)[::-1]
            top_eigvals = sorted_eigvals[:actual_k]
            
            # 填充
            if len(top_eigvals) < top_k:
                top_eigvals = np.concatenate([
                    top_eigvals,
                    np.zeros(top_k - len(top_eigvals), dtype=np.float32)
                ])
            
            all_eigenvalues.append(top_eigvals.astype(np.float32))
    
    return np.concatenate(all_eigenvalues)


# =============================================================================
# LapEigvals Method 类
# =============================================================================

@METHODS.register("lapeigvals", aliases=["lap_eigvals", "laplacian_eigenvalues"])
class LapEigvalsMethod(BaseMethod):
    """Laplacian Eigenvalue-based 幻觉检测方法。
    
    严格按照 EMNLP 2025 论文实现:
    - 从注意力矩阵计算 Laplacian 特征值
    - 利用下三角结构高效计算: O(T) vs O(T³)
    - PCA 降维 (默认 512 维) + LogisticRegression
    """
    
    # 论文默认参数
    DEFAULT_TOP_K = 100
    DEFAULT_PCA_DIM = 512
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        
        # 核心参数 (按论文默认值)
        self.top_k_eigenvalues = params.get("top_k_eigenvalues", self.DEFAULT_TOP_K)
        self.pca_dim = params.get("pca_dim", self.DEFAULT_PCA_DIM)
        self.response_only = params.get("response_only", True)
        
        # PCA 和标准化器
        self._pca: Optional[PCA] = None
        self._scaler = StandardScaler()
        
        # 内部状态
        self._raw_feature_dim = None
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 LapEigvals 特征。
        
        优先级:
        1. 完整注意力矩阵 (full_attention) - 最准确
        2. 预计算的 Laplacian 对角线 (laplacian_diags) - 高效
        3. 报错 - 需要至少上述之一
        """
        # 尝试获取完整注意力矩阵
        full_attention = features.full_attention
        if full_attention is None:
            full_attention = features.get_full_attention()
        
        if full_attention is not None:
            # 方案1: 从完整注意力矩阵计算
            # 使用 safe_to_numpy 处理 BFloat16
            full_attention = safe_to_numpy(full_attention)
            
            feat_vec = extract_lapeigvals_features(
                full_attention,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
                top_k=self.top_k_eigenvalues,
                response_only=self.response_only,
            )
            
            # 释放大内存
            features.release_large_features()
            
        elif features.laplacian_diags is not None:
            # 方案2: 从预计算的 Laplacian 对角线提取
            laplacian_diag = safe_to_numpy(features.laplacian_diags)
            
            feat_vec = extract_features_from_precomputed_laplacian_diag(
                laplacian_diag,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
                top_k=self.top_k_eigenvalues,
                response_only=self.response_only,
            )
        else:
            raise MethodError(
                "LapEigvals requires either 'full_attention' or 'laplacian_diags'. "
                "Please enable 'store_full_attention: true' in features config, "
                "or pre-compute laplacian_diags during feature extraction."
            )
        
        # 处理无效值
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """训练 LapEigvals 方法。"""
        X = []
        y = []
        
        n_total = len(features_list)
        log_interval = max(1, n_total // 10)
        
        for i, feat in enumerate(features_list):
            if i % log_interval == 0:
                logger.info(f"Extracting features: {i}/{n_total} ({100*i/n_total:.0f}%)")
            
            try:
                x = self.extract_method_features(feat)
                if x is not None and len(x) > 0:
                    X.append(x)
                    label = labels[i] if labels else feat.label
                    if label is not None:
                        y.append(label)
                    else:
                        logger.warning(f"No label for sample {feat.sample_id}")
            except Exception as e:
                logger.warning(f"Feature extraction failed for {feat.sample_id}: {e}")
        
        if len(X) == 0:
            raise MethodError("No valid features extracted")
        
        X = np.array(X)
        y = np.array(y)
        
        self._raw_feature_dim = X.shape[1]
        logger.info(f"Extracted features: {X.shape}, Labels: {y.shape}")
        logger.info(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
        
        # 标准化
        X_scaled = self._scaler.fit_transform(X)
        
        # PCA 降维
        if self.pca_dim and self.pca_dim < X_scaled.shape[1]:
            self._pca = PCA(n_components=self.pca_dim, random_state=42)
            X_reduced = self._pca.fit_transform(X_scaled)
            logger.info(f"PCA: {X_scaled.shape[1]} -> {X_reduced.shape[1]}")
        else:
            X_reduced = X_scaled
            self._pca = None
        
        self._feature_dim = X_reduced.shape[1]
        
        # 训练分类器
        params = self.config.params or {}
        self.classifier = LogisticRegression(
            max_iter=params.get("max_iter", 2000),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42,
        )
        
        # 交叉验证
        metrics = {}
        if cv and len(y) >= 5:
            try:
                cv_scores = cross_val_score(
                    self.classifier, X_reduced, y, 
                    cv=min(5, len(y)), 
                    scoring='roc_auc'
                )
                metrics['cv_auroc_mean'] = float(np.mean(cv_scores))
                metrics['cv_auroc_std'] = float(np.std(cv_scores))
                logger.info(f"CV AUROC: {metrics['cv_auroc_mean']:.4f} ± {metrics['cv_auroc_std']:.4f}")
            except Exception as e:
                logger.warning(f"CV failed: {e}")
        
        # 最终训练
        self.classifier.fit(X_reduced, y)
        self.is_fitted = True
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """预测单个样本。"""
        if not self.is_fitted:
            raise MethodError("Method not fitted. Call fit() first.")
        
        # 提取特征
        x = self.extract_method_features(features)
        
        # 标准化
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        
        # PCA 变换
        if self._pca is not None:
            x_reduced = self._pca.transform(x_scaled)
        else:
            x_reduced = x_scaled
        
        # 预测
        proba = self.classifier.predict_proba(x_reduced)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return Prediction(
            sample_id=features.sample_id,
            score=score,
            label=1 if score > 0.5 else 0,
            confidence=abs(score - 0.5) * 2,
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """保存模型到文件 (统一 model.pkl 格式)。"""
        if not self.is_fitted:
            raise MethodError("Cannot save unfitted method")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "classifier": self.classifier,
            "scaler": self._scaler,
            "pca": self._pca,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            "raw_feature_dim": self._raw_feature_dim,
            # 保存关键参数以便恢复
            "top_k_eigenvalues": self.top_k_eigenvalues,
            "pca_dim": self.pca_dim,
            "response_only": self.response_only,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved method to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """从文件加载模型。"""
        path = Path(path)
        if not path.exists():
            raise MethodError(f"Method file not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.classifier = state["classifier"]
        self._scaler = state["scaler"]
        self._pca = state.get("pca")
        self.is_fitted = state["is_fitted"]
        self._feature_dim = state.get("feature_dim")
        self._raw_feature_dim = state.get("raw_feature_dim")
        
        # 恢复参数
        self.top_k_eigenvalues = state.get("top_k_eigenvalues", self.DEFAULT_TOP_K)
        self.pca_dim = state.get("pca_dim", self.DEFAULT_PCA_DIM)
        self.response_only = state.get("response_only", True)
        
        logger.info(f"Loaded method from {path}")


# =============================================================================
# LapEigvals Full Method (使用完整特征值计算)
# =============================================================================

@METHODS.register("lapeigvals_full", aliases=["lap_eigvals_full"])
class LapEigvalsFullMethod(LapEigvalsMethod):
    """LapEigvals with full eigenvalue computation.
    
    需要完整注意力矩阵，更准确但内存消耗更大。
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """强制使用完整注意力矩阵计算特征值。"""
        full_attention = features.full_attention
        if full_attention is None:
            full_attention = features.get_full_attention()
        
        if full_attention is None:
            raise MethodError(
                "LapEigvalsFullMethod requires 'full_attention'. "
                "Please enable 'store_full_attention: true' in features config."
            )
        
        # 安全转换 - 处理 BFloat16
        full_attention = safe_to_numpy(full_attention)
        
        feat_vec = extract_lapeigvals_features(
            full_attention,
            prompt_len=features.prompt_len,
            response_len=features.response_len,
            top_k=self.top_k_eigenvalues,
            response_only=self.response_only,
        )
        
        # 释放大内存
        features.release_large_features()
        
        # 处理无效值
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec