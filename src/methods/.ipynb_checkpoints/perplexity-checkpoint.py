"""Perplexity-based Hallucination Detection.

基于论文: "Out-of-Distribution Detection and Selective Generation for Conditional Language Models"
(Ren et al., ICLR 2023 Spotlight, Google Research)

=============================================================================
核心原理
=============================================================================
Perplexity (困惑度) 是语言模型不确定性的经典度量:
    PPL(y|x) = exp(-1/T × Σₜ log P(yₜ | y<t, x))

关键发现 (论文):
- 单独使用 perplexity 对 OOD 检测效果有限 (AUROC ~0.424)
- 但与其他特征结合时能提升性能 (+12% 相关性)
- Token-level perplexity 对检测局部幻觉更有效

=============================================================================
实现的特征
=============================================================================
1. Sequence-level Perplexity: 标准困惑度 (长度归一化)
2. Token-level Perplexity: 每个token的困惑度 (-log P)
3. Entropy Features: 预测分布的熵
4. Aggregation Methods: mean, max, percentile 等
5. Combined Score: perplexity + entropy 的百分位组合

=============================================================================
BFloat16 安全处理
=============================================================================
所有张量操作都通过 safe_to_numpy() 处理，避免 BFloat16 转换错误。
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import logging
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction, MethodError
from src.methods.base import BaseMethod

logger = logging.getLogger(__name__)


# =============================================================================
# 辅助函数: 安全的张量转换 (处理BFloat16)
# =============================================================================

def safe_to_numpy(tensor: Union[torch.Tensor, np.ndarray, None]) -> Optional[np.ndarray]:
    """安全地将张量转换为NumPy数组，处理BFloat16等不支持的类型。
    
    Args:
        tensor: PyTorch张量或NumPy数组
        
    Returns:
        NumPy float32数组
    """
    if tensor is None:
        return None
    
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32) if tensor.dtype != np.float32 else tensor
    
    if isinstance(tensor, torch.Tensor):
        # 关键修复: 先转float32再转numpy，解决BFloat16问题
        return tensor.detach().cpu().float().numpy()
    
    return np.asarray(tensor, dtype=np.float32)


def safe_to_tensor(data: Union[torch.Tensor, np.ndarray, None],
                   dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    """安全地将数据转换为PyTorch张量。"""
    if data is None:
        return None
    
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data.astype(np.float32)).to(dtype)
    
    return torch.tensor(data, dtype=dtype)


# =============================================================================
# Perplexity 计算函数
# =============================================================================

def compute_token_perplexity(
    token_probs: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-10,
) -> np.ndarray:
    """计算每个token的perplexity (-log probability)。
    
    Token-level perplexity = exp(-log P(token))
    
    Args:
        token_probs: token概率 [seq_len] 或 [batch, seq_len]
        eps: 数值稳定性常数
        
    Returns:
        Token-level perplexity [seq_len]
    """
    probs = safe_to_numpy(token_probs)
    
    if probs is None or len(probs) == 0:
        return np.array([1.0], dtype=np.float32)
    
    # 确保是1D
    if probs.ndim > 1:
        probs = probs.flatten()
    
    # 裁剪避免log(0)
    probs = np.clip(probs, eps, 1.0)
    
    # Token perplexity = exp(-log(p)) = 1/p
    neg_log_probs = -np.log(probs)
    token_ppl = np.exp(neg_log_probs)
    
    return token_ppl.astype(np.float32)


def compute_sequence_perplexity(
    token_probs: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-10,
) -> float:
    """计算序列级别的perplexity (长度归一化)。
    
    PPL = exp(-1/T × Σ log P(token))
    
    Args:
        token_probs: token概率 [seq_len]
        eps: 数值稳定性常数
        
    Returns:
        Sequence perplexity (scalar)
    """
    probs = safe_to_numpy(token_probs)
    
    if probs is None or len(probs) == 0:
        return 1.0
    
    # 确保是1D
    if probs.ndim > 1:
        probs = probs.flatten()
    
    probs = np.clip(probs, eps, 1.0)
    
    # Mean negative log probability
    mean_neg_log_prob = -np.mean(np.log(probs))
    
    # Perplexity
    ppl = np.exp(mean_neg_log_prob)
    
    return float(ppl)


def compute_token_entropy_from_logits(
    logits: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-10,
) -> np.ndarray:
    """从logits计算每个位置的预测熵。
    
    H = -Σ P(v) × log P(v)
    
    Args:
        logits: [seq_len, vocab_size]
        eps: 数值稳定性常数
        
    Returns:
        Entropy [seq_len]
    """
    logits = safe_to_tensor(logits)
    
    if logits is None:
        return np.array([0.0], dtype=np.float32)
    
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return safe_to_numpy(entropy)


def compute_normalized_entropy(
    entropy: np.ndarray,
    vocab_size: int = 32000,
) -> np.ndarray:
    """归一化熵到[0,1]范围。
    
    Normalized Entropy = H / log(vocab_size)
    
    Args:
        entropy: 原始熵值
        vocab_size: 词表大小
        
    Returns:
        归一化熵
    """
    max_entropy = np.log(vocab_size)
    return (entropy / max_entropy).astype(np.float32)


# =============================================================================
# 特征提取函数
# =============================================================================

def extract_perplexity_features(
    token_probs: Union[torch.Tensor, np.ndarray, None],
    token_entropy: Union[torch.Tensor, np.ndarray, None] = None,
    sequence_perplexity: Optional[float] = None,
    prompt_len: int = 0,
    response_len: int = 0,
    include_percentiles: bool = True,
) -> np.ndarray:
    """提取综合的Perplexity特征向量。
    
    特征包括:
    - Sequence-level perplexity 统计
    - Token-level perplexity 统计 (mean, max, std, percentiles)
    - Token entropy 统计
    - 组合特征
    
    Args:
        token_probs: token概率 [seq_len]
        token_entropy: token熵 [seq_len] (可选)
        sequence_perplexity: 预计算的序列perplexity (可选)
        prompt_len: prompt长度
        response_len: response长度
        include_percentiles: 是否包含百分位特征
        
    Returns:
        特征向量
    """
    features = []
    eps = 1e-10
    
    # === 1. Sequence Perplexity ===
    if sequence_perplexity is not None:
        ppl = sequence_perplexity
    elif token_probs is not None:
        probs = safe_to_numpy(token_probs)
        if probs is not None and len(probs) > 0:
            ppl = compute_sequence_perplexity(probs)
        else:
            ppl = 1.0
    else:
        ppl = 1.0
    
    features.extend([
        ppl,                          # 原始perplexity
        np.log(ppl + 1),              # log perplexity
        min(ppl, 1000) / 1000,        # 归一化 (裁剪到1000)
    ])
    
    # === 2. Token Probability Statistics ===
    if token_probs is not None:
        probs = safe_to_numpy(token_probs)
        
        if probs is not None and len(probs) > 0:
            # 聚焦response部分
            if response_len > 0 and prompt_len < len(probs):
                end_idx = min(prompt_len + response_len, len(probs))
                resp_probs = probs[prompt_len:end_idx] if prompt_len < end_idx else probs
            else:
                resp_probs = probs
            
            if len(resp_probs) > 0:
                # 基础统计
                features.extend([
                    np.mean(resp_probs),
                    np.std(resp_probs),
                    np.min(resp_probs),
                    np.max(resp_probs),
                ])
                
                # Negative log probability statistics
                neg_log_probs = -np.log(np.clip(resp_probs, eps, 1.0))
                features.extend([
                    np.mean(neg_log_probs),      # Mean NLL
                    np.std(neg_log_probs),       # NLL variance
                    np.max(neg_log_probs),       # Worst-case uncertainty
                ])
                
                # Percentiles (捕捉分布形状)
                if include_percentiles:
                    features.extend([
                        np.percentile(resp_probs, 10),
                        np.percentile(resp_probs, 25),
                        np.percentile(resp_probs, 50),
                        np.percentile(resp_probs, 75),
                        np.percentile(resp_probs, 90),
                    ])
                
                # Token perplexity statistics
                token_ppl = compute_token_perplexity(resp_probs)
                features.extend([
                    np.mean(token_ppl),
                    np.max(token_ppl),           # Max token PPL (局部幻觉检测)
                    np.std(token_ppl),
                ])
            else:
                # 填充默认值
                n_prob_features = 7 + (5 if include_percentiles else 0) + 3
                features.extend([0.0] * n_prob_features)
        else:
            n_prob_features = 7 + (5 if include_percentiles else 0) + 3
            features.extend([0.0] * n_prob_features)
    else:
        n_prob_features = 7 + (5 if include_percentiles else 0) + 3
        features.extend([0.0] * n_prob_features)
    
    # === 3. Token Entropy Statistics ===
    if token_entropy is not None:
        entropy = safe_to_numpy(token_entropy)
        
        if entropy is not None and len(entropy) > 0:
            # 聚焦response部分
            if response_len > 0 and prompt_len < len(entropy):
                end_idx = min(prompt_len + response_len, len(entropy))
                resp_entropy = entropy[prompt_len:end_idx] if prompt_len < end_idx else entropy
            else:
                resp_entropy = entropy
            
            if len(resp_entropy) > 0:
                features.extend([
                    np.mean(resp_entropy),
                    np.std(resp_entropy),
                    np.max(resp_entropy),        # Max entropy (最不确定的token)
                    np.min(resp_entropy),
                ])
                
                if include_percentiles:
                    features.extend([
                        np.percentile(resp_entropy, 75),
                        np.percentile(resp_entropy, 90),
                        np.percentile(resp_entropy, 95),
                    ])
            else:
                n_entropy_features = 4 + (3 if include_percentiles else 0)
                features.extend([0.0] * n_entropy_features)
        else:
            n_entropy_features = 4 + (3 if include_percentiles else 0)
            features.extend([0.0] * n_entropy_features)
    else:
        n_entropy_features = 4 + (3 if include_percentiles else 0)
        features.extend([0.0] * n_entropy_features)
    
    # === 4. Combined Features ===
    # Perplexity-Entropy correlation (如果都有)
    if token_probs is not None and token_entropy is not None:
        probs = safe_to_numpy(token_probs)
        entropy = safe_to_numpy(token_entropy)
        
        if probs is not None and entropy is not None and len(probs) > 0 and len(entropy) > 0:
            min_len = min(len(probs), len(entropy))
            if min_len > 1:
                correlation = np.corrcoef(probs[:min_len], entropy[:min_len])[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                features.append(correlation)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # 处理无效值
    features = np.array(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return features


def extract_token_level_perplexity_features(
    token_probs: Union[torch.Tensor, np.ndarray],
    token_entropy: Union[torch.Tensor, np.ndarray, None] = None,
    prompt_len: int = 0,
    response_len: int = 0,
) -> np.ndarray:
    """提取token级别的perplexity特征 (用于token-level检测)。
    
    Args:
        token_probs: [seq_len]
        token_entropy: [seq_len] (可选)
        prompt_len: prompt长度
        response_len: response长度
        
    Returns:
        Token-level features [response_len, feature_dim]
    """
    probs = safe_to_numpy(token_probs)
    
    if probs is None or len(probs) == 0:
        return np.zeros((1, 4), dtype=np.float32)
    
    # 确定response范围
    seq_len = len(probs)
    if response_len > 0 and prompt_len < seq_len:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len)
    else:
        start_idx = 0
        end_idx = seq_len
    
    resp_probs = probs[start_idx:end_idx]
    actual_len = len(resp_probs)
    
    if actual_len == 0:
        return np.zeros((1, 4), dtype=np.float32)
    
    # 计算token perplexity
    token_ppl = compute_token_perplexity(resp_probs)
    neg_log_probs = -np.log(np.clip(resp_probs, 1e-10, 1.0))
    
    # 基础特征: [prob, neg_log_prob, ppl, entropy]
    features = np.zeros((actual_len, 4), dtype=np.float32)
    features[:, 0] = resp_probs
    features[:, 1] = neg_log_probs
    features[:, 2] = token_ppl
    
    # 添加entropy (如果有)
    if token_entropy is not None:
        entropy = safe_to_numpy(token_entropy)
        if entropy is not None and len(entropy) > start_idx:
            resp_entropy = entropy[start_idx:end_idx]
            if len(resp_entropy) == actual_len:
                features[:, 3] = resp_entropy
    
    return features


# =============================================================================
# Perplexity Method 类
# =============================================================================

@METHODS.register("perplexity", aliases=["ppl", "perplexity_ood"])
class PerplexityMethod(BaseMethod):
    """Perplexity-based 幻觉检测方法。
    
    基于论文: "Out-of-Distribution Detection and Selective Generation 
    for Conditional Language Models" (ICLR 2023)
    
    特点:
    - Training-free: 无需额外训练，直接使用模型输出
    - 支持 sample-level 和 token-level 检测
    - 多种聚合策略: mean, max, percentile
    - 可与 entropy 特征组合
    
    关键参数:
    - aggregation: 聚合方式 ("mean", "max", "percentile")
    - use_entropy: 是否使用熵特征
    - use_percentiles: 是否使用百分位特征
    - response_only: 是否只使用response部分
    """
    
    # 是否支持token-level训练
    supports_token_level: bool = True
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        
        # 特征配置
        self.aggregation = params.get("aggregation", "mean")
        self.use_entropy = params.get("use_entropy", True)
        self.use_percentiles = params.get("use_percentiles", True)
        self.response_only = params.get("response_only", True)
        
        # 分类器参数
        classifier_params = params.get("classifier_params", {})
        self._classifier_max_iter = classifier_params.get("max_iter", 1000)
        self._classifier_C = classifier_params.get("C", 1.0)
        self._classifier_class_weight = classifier_params.get("class_weight", "balanced")
        
        # Token-level 分类器
        self._token_classifier = None
        self._token_scaler = StandardScaler()
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取Perplexity特征向量。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            特征向量
        """
        # 获取token概率
        token_probs = None
        if features.token_probs is not None:
            token_probs = safe_to_numpy(features.token_probs)
        
        # 获取token熵
        token_entropy = None
        if self.use_entropy and features.token_entropy is not None:
            token_entropy = safe_to_numpy(features.token_entropy)
        
        # 获取序列perplexity
        sequence_ppl = features.perplexity
        
        # 提取特征
        feat_vec = extract_perplexity_features(
            token_probs=token_probs,
            token_entropy=token_entropy,
            sequence_perplexity=sequence_ppl,
            prompt_len=features.prompt_len if self.response_only else 0,
            response_len=features.response_len,
            include_percentiles=self.use_percentiles,
        )
        
        return feat_vec
    
    def extract_token_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取token级别特征 (用于token-level检测)。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            Token-level features [response_len, feature_dim]
        """
        token_probs = safe_to_numpy(features.token_probs)
        token_entropy = safe_to_numpy(features.token_entropy) if self.use_entropy else None
        
        if token_probs is None:
            raise MethodError("Perplexity method requires token_probs for token-level detection")
        
        return extract_token_level_perplexity_features(
            token_probs=token_probs,
            token_entropy=token_entropy,
            prompt_len=features.prompt_len if self.response_only else 0,
            response_len=features.response_len,
        )
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """训练Perplexity方法。
        
        Args:
            features_list: ExtractedFeatures列表
            labels: 标签 (如果不在features中)
            cv: 是否进行交叉验证
            
        Returns:
            训练指标
        """
        X = []
        y = []
        
        n_total = len(features_list)
        log_interval = max(1, n_total // 10)
        
        for i, feat in enumerate(features_list):
            if i % log_interval == 0:
                logger.info(f"提取特征: {i}/{n_total} ({100*i/n_total:.0f}%)")
            
            try:
                x = self.extract_method_features(feat)
                if x is not None and len(x) > 0 and not np.any(np.isnan(x)):
                    X.append(x)
                    label = labels[i] if labels else feat.label
                    if label is not None:
                        y.append(label)
                    else:
                        logger.warning(f"样本 {feat.sample_id} 没有标签")
            except Exception as e:
                logger.warning(f"特征提取失败 {feat.sample_id}: {e}")
        
        if len(X) == 0:
            raise MethodError("没有有效的特征")
        
        X = np.array(X)
        y = np.array(y)
        
        self._feature_dim = X.shape[1]
        logger.info(f"特征维度: {X.shape}, 标签: {y.shape}")
        logger.info(f"类别分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练分类器
        self.classifier = LogisticRegression(
            max_iter=self._classifier_max_iter,
            C=self._classifier_C,
            class_weight=self._classifier_class_weight,
            random_state=42,
        )
        
        # 交叉验证
        metrics = {}
        if cv and len(y) >= 5:
            try:
                cv_scores = cross_val_score(
                    self.classifier, X_scaled, y,
                    cv=min(5, len(y)),
                    scoring='roc_auc'
                )
                metrics['cv_auroc_mean'] = float(np.mean(cv_scores))
                metrics['cv_auroc_std'] = float(np.std(cv_scores))
                logger.info(f"CV AUROC: {metrics['cv_auroc_mean']:.4f} ± {metrics['cv_auroc_std']:.4f}")
            except Exception as e:
                logger.warning(f"交叉验证失败: {e}")
        
        # 最终训练
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        
        return metrics
    
    def fit_token_level(
        self,
        features_list: List[ExtractedFeatures],
        token_labels_list: Optional[List[List[int]]] = None,
    ) -> Dict[str, float]:
        """训练token级别分类器。
        
        Args:
            features_list: ExtractedFeatures列表
            token_labels_list: token级别标签列表
            
        Returns:
            训练指标
        """
        X_all = []
        y_all = []
        
        for i, feat in enumerate(features_list):
            try:
                token_feats = self.extract_token_features(feat)
                
                # 获取标签
                if token_labels_list is not None and i < len(token_labels_list):
                    token_labels = token_labels_list[i]
                elif feat.hallucination_labels is not None:
                    token_labels = feat.hallucination_labels
                else:
                    continue
                
                # 确保长度匹配
                min_len = min(len(token_feats), len(token_labels))
                X_all.extend(token_feats[:min_len].tolist())
                y_all.extend(token_labels[:min_len])
                
            except Exception as e:
                logger.warning(f"Token特征提取失败 {feat.sample_id}: {e}")
        
        if len(X_all) == 0:
            raise MethodError("没有有效的token级别特征")
        
        X = np.array(X_all)
        y = np.array(y_all)
        
        logger.info(f"Token特征: {X.shape}, 标签分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
        
        # 训练
        X_scaled = self._token_scaler.fit_transform(X)
        
        self._token_classifier = LogisticRegression(
            max_iter=self._classifier_max_iter,
            C=self._classifier_C,
            class_weight=self._classifier_class_weight,
            random_state=42,
        )
        self._token_classifier.fit(X_scaled, y)
        self.is_token_fitted = True
        
        return {'token_samples': len(X)}
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """预测单个样本。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            Prediction对象
        """
        if not self.is_fitted:
            raise MethodError("方法未训练，请先调用fit()")
        
        # 提取特征
        x = self.extract_method_features(features)
        
        # 标准化
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        
        # 预测
        proba = self.classifier.predict_proba(x_scaled)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return Prediction(
            sample_id=features.sample_id,
            score=score,
            label=1 if score > 0.5 else 0,
            confidence=abs(score - 0.5) * 2,
        )
    
    def predict_token_level(
        self,
        features: ExtractedFeatures,
    ) -> Dict[str, Any]:
        """Token级别预测。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            包含token预测的字典
        """
        if not self.is_token_fitted:
            raise MethodError("Token分类器未训练")
        
        token_feats = self.extract_token_features(features)
        X_scaled = self._token_scaler.transform(token_feats)
        
        probs = self._token_classifier.predict_proba(X_scaled)[:, 1]
        predictions = (probs > 0.5).astype(int).tolist()
        
        # 聚合为sample-level
        if self.aggregation == "max":
            sample_score = float(probs.max())
        elif self.aggregation == "mean":
            sample_score = float(probs.mean())
        else:  # percentile
            sample_score = float(np.percentile(probs, 90))
        
        return {
            'token_predictions': predictions,
            'token_probabilities': probs.tolist(),
            'sample_score': sample_score,
            'sample_label': 1 if sample_score > 0.5 else 0,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """保存模型到单个文件 (统一model.pkl格式)。
        
        Args:
            path: 保存路径
        """
        if not self.is_fitted:
            raise MethodError("无法保存未训练的方法")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            # 保存参数
            "aggregation": self.aggregation,
            "use_entropy": self.use_entropy,
            "use_percentiles": self.use_percentiles,
            "response_only": self.response_only,
            # Token-level
            "token_classifier": self._token_classifier,
            "token_scaler": self._token_scaler,
            "is_token_fitted": getattr(self, 'is_token_fitted', False),
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"保存方法到 {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """从文件加载模型。
        
        Args:
            path: 模型路径
        """
        path = Path(path)
        
        if not path.exists():
            raise MethodError(f"模型文件不存在: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.classifier = state["classifier"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        self._feature_dim = state.get("feature_dim")
        
        # 恢复参数
        self.aggregation = state.get("aggregation", "mean")
        self.use_entropy = state.get("use_entropy", True)
        self.use_percentiles = state.get("use_percentiles", True)
        self.response_only = state.get("response_only", True)
        
        # Token-level
        self._token_classifier = state.get("token_classifier")
        self._token_scaler = state.get("token_scaler", StandardScaler())
        self.is_token_fitted = state.get("is_token_fitted", False)
        
        logger.info(f"从 {path} 加载方法")


# =============================================================================
# 便捷函数: Training-Free 检测
# =============================================================================

def compute_perplexity_score(
    token_probs: Union[torch.Tensor, np.ndarray],
    method: str = "mean",
) -> float:
    """计算perplexity-based幻觉分数 (training-free)。
    
    分数越高越可能是幻觉。
    
    Args:
        token_probs: token概率
        method: 聚合方法 ("mean", "max", "percentile90")
        
    Returns:
        幻觉分数 (0-1)
    """
    probs = safe_to_numpy(token_probs)
    
    if probs is None or len(probs) == 0:
        return 0.5
    
    # 计算token perplexity
    token_ppl = compute_token_perplexity(probs)
    
    # 聚合
    if method == "max":
        ppl = np.max(token_ppl)
    elif method == "percentile90":
        ppl = np.percentile(token_ppl, 90)
    else:  # mean
        ppl = np.mean(token_ppl)
    
    # 转换为0-1分数 (使用sigmoid-like变换)
    # 经验性地，perplexity > 10 通常表示高不确定性
    score = 1.0 / (1.0 + np.exp(-0.1 * (ppl - 10)))
    
    return float(np.clip(score, 0.0, 1.0))


def detect_high_perplexity_tokens(
    token_probs: Union[torch.Tensor, np.ndarray],
    threshold_percentile: float = 90,
) -> List[int]:
    """检测高perplexity的token位置。
    
    Args:
        token_probs: token概率
        threshold_percentile: 阈值百分位
        
    Returns:
        高perplexity的token索引列表
    """
    probs = safe_to_numpy(token_probs)
    
    if probs is None or len(probs) == 0:
        return []
    
    token_ppl = compute_token_perplexity(probs)
    threshold = np.percentile(token_ppl, threshold_percentile)
    
    high_ppl_indices = np.where(token_ppl > threshold)[0].tolist()
    
    return high_ppl_indices