"""Lookback Lens: Attention Ratio-based Hallucination Detection.

基于论文: "Lookback Lens: Detecting and Mitigating Contextual Hallucinations 
in Large Language Models Using Only Attention Maps" (EMNLP 2024)
代码参考: https://github.com/voidism/Lookback-Lens

核心思想:
- 幻觉 token 通常对 context 的注意力较低
- Lookback ratio = attention_to_context / (attention_to_context + attention_to_new_tokens)
- 支持 token 级别和 sample 级别检测

级别 (level):
- sample: 聚合特征 + LogisticRegression (默认)
- token: 逐token特征 + 分类器
- both: 优先token级别，无标签时回退到sample
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod

logger = logging.getLogger(__name__)


class InsufficientResponseError(ValueError):
    """Raised when response length is insufficient for feature extraction."""
    pass


def compute_token_lookback_ratios(
    attn_diags: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """计算每个 response token 的 lookback ratio。
    
    原论文公式: LR_t = A_context / (A_context + A_new)
    其中 A_context 是对 prompt 的平均注意力，A_new 是对新生成 token 的注意力
    
    Args:
        attn_diags: [n_layers, n_heads, seq_len]
        prompt_len: prompt 长度
        response_len: response 长度
        
    Returns:
        Token-level features [resp_len, n_layers * n_heads * 2]
        
    Raises:
        InsufficientResponseError: When actual response length <= 0
    """
    n_layers, n_heads, seq_len = attn_diags.shape
    attn_diags = attn_diags.float()
    
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    actual_resp_len = resp_end - resp_start
    
    # 严格检查：如果 response 长度不足，抛出异常让调用方跳过该样本
    if actual_resp_len <= 0:
        raise InsufficientResponseError(
            f"Invalid response length: actual_resp_len={actual_resp_len} "
            f"(prompt_len={prompt_len}, response_len={response_len}, seq_len={seq_len})"
        )
    
    # 对每个 response token 计算特征
    token_features = []
    
    for t in range(resp_start, resp_end):
        # 获取位置 t 的对角线值（所有层和头）
        diag_at_t = attn_diags[:, :, t]  # [n_layers, n_heads]
        
        # 计算 context 部分的统计（0 到 prompt_len）
        if t > 0:
            context_attn = attn_diags[:, :, :min(t, prompt_len)].mean(dim=-1)
        else:
            context_attn = torch.zeros(n_layers, n_heads)
        
        # 计算新生成部分的统计（prompt_len 到 t）
        if t > prompt_len:
            new_attn = attn_diags[:, :, prompt_len:t].mean(dim=-1)
        else:
            new_attn = torch.zeros(n_layers, n_heads)
        
        # Lookback ratio
        total = context_attn + new_attn + 1e-8
        lookback_ratio = context_attn / total
        
        # 拼接特征: [diag_at_t, lookback_ratio]
        feat = torch.cat([diag_at_t.flatten(), lookback_ratio.flatten()])
        token_features.append(feat)
    
    return torch.stack(token_features)  # [resp_len, n_layers * n_heads * 2]


def compute_lookback_ratio(
    attn_diags: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """Compute sample-level lookback ratio features.
    
    Args:
        attn_diags: [n_layers, n_heads, seq_len]
        prompt_len: Length of prompt
        response_len: Length of response
        
    Returns:
        Feature vector [n_layers * n_heads * 4]
    """
    n_layers, n_heads, seq_len = attn_diags.shape
    attn_diags = attn_diags.float()
    
    # Split into prompt and response portions
    prompt_diag = attn_diags[:, :, :prompt_len]
    
    if prompt_len + response_len <= seq_len:
        response_diag = attn_diags[:, :, prompt_len:prompt_len + response_len]
    else:
        response_diag = attn_diags[:, :, prompt_len:]
    
    # Compute statistics
    p_mean = prompt_diag.mean(dim=-1) if prompt_diag.shape[-1] > 0 else torch.zeros(n_layers, n_heads)
    r_mean = response_diag.mean(dim=-1) if response_diag.shape[-1] > 0 else torch.zeros(n_layers, n_heads)
    
    # Lookback ratio
    ratio = p_mean / (r_mean + 1e-8)
    diff = p_mean - r_mean
    
    features = torch.stack([p_mean, r_mean, ratio, diff], dim=-1)
    return features.cpu().numpy().flatten().astype(np.float32)


def compute_attention_patterns(
    attn_diags: torch.Tensor,
    attn_entropy: Optional[torch.Tensor],
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """Compute comprehensive attention pattern features."""
    n_layers, n_heads, seq_len = attn_diags.shape
    
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    resp_diag = attn_diags[:, :, resp_start:resp_end].float()
    resp_len_actual = resp_diag.shape[-1]
    
    # 需要至少 2 个元素才能计算 std
    if resp_len_actual <= 1:
        n_features = 5 * n_layers * n_heads
        if attn_entropy is not None:
            n_features += 2 * n_layers * n_heads
        if resp_len_actual == 1:
            # 只有一个元素时，mean/max/min 有意义，std/trend 设为 0
            diag_val = resp_diag.squeeze(-1)  # [n_layers, n_heads]
            diag_features = torch.stack([
                diag_val,  # mean = single value
                torch.zeros_like(diag_val),  # std = 0
                diag_val,  # max = single value
                diag_val,  # min = single value
                torch.zeros_like(diag_val),  # trend = 0
            ], dim=-1)
            
            if attn_entropy is not None:
                resp_entropy = attn_entropy[:, :, resp_start:resp_end].float()
                if resp_entropy.shape[-1] == 1:
                    ent_val = resp_entropy.squeeze(-1)
                    ent_features = torch.stack([ent_val, torch.zeros_like(ent_val)], dim=-1)
                else:
                    ent_features = torch.zeros(n_layers, n_heads, 2)
                all_features = torch.cat([diag_features, ent_features], dim=-1)
            else:
                all_features = diag_features
            return all_features.cpu().numpy().flatten().astype(np.float32)
        return np.zeros(n_features, dtype=np.float32)
    
    # Statistics
    diag_mean = resp_diag.mean(dim=-1)
    diag_std = resp_diag.std(dim=-1)
    diag_max = resp_diag.max(dim=-1).values
    diag_min = resp_diag.min(dim=-1).values
    
    # Trend
    if resp_len_actual > 1:
        x = torch.arange(resp_len_actual, dtype=resp_diag.dtype, device=resp_diag.device)
        x_mean = x.mean()
        y_mean = resp_diag.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        y_centered = resp_diag - y_mean
        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = (x_centered ** 2).sum() + 1e-8
        trend = numerator / denominator
    else:
        trend = torch.zeros(n_layers, n_heads)
    
    diag_features = torch.stack([diag_mean, diag_std, diag_max, diag_min, trend], dim=-1)
    
    if attn_entropy is not None:
        resp_entropy = attn_entropy[:, :, resp_start:resp_end].float()
        if resp_entropy.shape[-1] > 1:
            ent_mean = resp_entropy.mean(dim=-1)
            ent_std = resp_entropy.std(dim=-1)
        elif resp_entropy.shape[-1] == 1:
            # 只有一个元素时，std 设为 0
            ent_mean = resp_entropy.squeeze(-1)
            ent_std = torch.zeros_like(ent_mean)
        else:
            ent_mean = torch.zeros(n_layers, n_heads)
            ent_std = torch.zeros(n_layers, n_heads)
        
        ent_features = torch.stack([ent_mean, ent_std], dim=-1)
        all_features = torch.cat([diag_features, ent_features], dim=-1)
    else:
        all_features = diag_features
    
    return all_features.cpu().numpy().flatten().astype(np.float32)


@METHODS.register("lookback_lens", aliases=["lookback", "attention_ratio"])
class LookbackLensMethod(BaseMethod):
    """Lookback Lens hallucination detection.
    
    支持两种级别:
    - sample: 样本级别特征聚合 + LogisticRegression
    - token: 逐token特征 + 分类器（需要 hallucination_labels）
    - both: 优先token，无标签时回退sample
    
    Attributes:
        supports_token_level: True - 原论文支持 token 级别检测
    """
    
    # 原论文支持 token 级别
    supports_token_level = True
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.use_entropy = params.get("use_entropy", True)
        self.use_trends = params.get("use_trends", True)
        
        # Token-level classifier
        self._token_classifier = None
        self._token_scaler = StandardScaler()
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract sample-level lookback features."""
        if features.attn_diags is None:
            raise ValueError("LookbackLens requires attn_diags")
        
        attn_diags = features.attn_diags
        attn_entropy = features.attn_entropy if self.use_entropy else None
        
        if isinstance(attn_diags, np.ndarray):
            attn_diags = torch.from_numpy(attn_diags)
        if attn_entropy is not None and isinstance(attn_entropy, np.ndarray):
            attn_entropy = torch.from_numpy(attn_entropy)
        
        lookback_features = compute_lookback_ratio(
            attn_diags, features.prompt_len, features.response_len
        )
        pattern_features = compute_attention_patterns(
            attn_diags, attn_entropy, features.prompt_len, features.response_len
        )
        
        feat_vec = np.concatenate([lookback_features, pattern_features])
        
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec
    
    def extract_token_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract token-level lookback features.
        
        Returns:
            [resp_len, feature_dim] token-level features
            
        Raises:
            ValueError: When attn_diags is not available
            InsufficientResponseError: When response length is insufficient
        """
        if features.attn_diags is None:
            raise ValueError("LookbackLens requires attn_diags")
        
        attn_diags = features.attn_diags
        if isinstance(attn_diags, np.ndarray):
            attn_diags = torch.from_numpy(attn_diags)
        
        # This may raise InsufficientResponseError
        token_features = compute_token_lookback_ratios(
            attn_diags, features.prompt_len, features.response_len
        )
        
        return token_features.cpu().numpy()
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """Train the method based on level.
        
        Args:
            features_list: List of extracted features
            labels: Sample-level labels
            cv: Whether to run cross-validation
            
        Returns:
            Training metrics
        """
        level = self.config.level
        
        if level == "token":
            return self._fit_token_level(features_list, labels)
        elif level == "both":
            # Check if we have token labels
            has_token_labels = any(f.hallucination_labels is not None for f in features_list)
            if has_token_labels:
                return self._fit_token_level(features_list, labels)
            else:
                logger.info("No token labels available, falling back to sample-level")
                return super().fit(features_list, labels, cv)
        else:
            # sample level
            return super().fit(features_list, labels, cv)
    
    def _fit_token_level(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Train token-level classifier."""
        logger.info("Training Lookback Lens at token level...")
        
        all_X = []
        all_y = []
        n_samples_with_labels = 0
        n_skipped = 0
        
        for i, feat in enumerate(features_list):
            try:
                token_features = self.extract_token_features(feat)
                
                # Get token labels
                if feat.hallucination_labels is not None:
                    token_labels = np.array(feat.hallucination_labels)
                    # 只取 response 部分
                    resp_start = feat.prompt_len
                    resp_end = min(feat.prompt_len + feat.response_len, len(token_labels))
                    token_labels = token_labels[resp_start:resp_end]
                    
                    # 确保长度匹配
                    min_len = min(len(token_features), len(token_labels))
                    if min_len > 0:
                        all_X.append(token_features[:min_len])
                        all_y.append(token_labels[:min_len])
                        n_samples_with_labels += 1
                else:
                    # 使用 sample label 作为所有 token 的标签
                    sample_label = labels[i] if labels else feat.label
                    if sample_label is not None and sample_label == 1:
                        # 幻觉样本：标记所有 response token
                        token_labels = np.ones(len(token_features))
                        all_X.append(token_features)
                        all_y.append(token_labels)
                    elif sample_label == 0:
                        # 正确样本：所有 token 标记为 0
                        token_labels = np.zeros(len(token_features))
                        all_X.append(token_features)
                        all_y.append(token_labels)
            except InsufficientResponseError as e:
                logger.debug(f"Skipping sample {feat.sample_id}: {e}")
                n_skipped += 1
            except Exception as e:
                logger.warning(f"Failed to extract features for {feat.sample_id}: {e}")
                n_skipped += 1
        
        if not all_X:
            raise ValueError("No valid token features extracted")
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        logger.info(f"Token-level training data: {len(X)} tokens, {y.sum():.0f} positive")
        logger.info(f"  Samples with precise labels: {n_samples_with_labels}")
        if n_skipped > 0:
            logger.info(f"  Skipped samples (insufficient response): {n_skipped}")
        
        # Scale and train
        X_scaled = self._token_scaler.fit_transform(X)
        
        self._token_classifier = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=self.config.random_seed
        )
        self._token_classifier.fit(X_scaled, y)
        
        # Also train sample-level classifier for fallback
        super().fit(features_list, labels, cv=False)
        
        return {
            "n_tokens": len(X),
            "n_positive_tokens": int(y.sum()),
            "n_samples_with_labels": n_samples_with_labels,
            "n_skipped": n_skipped,
        }
    
    def predict(self, features: ExtractedFeatures):
        """Predict based on level."""
        from src.core import Prediction
        
        level = self.config.level
        
        if level in ("token", "both") and self._token_classifier is not None:
            # Token-level prediction, aggregate to sample
            try:
                token_features = self.extract_token_features(features)
                X_scaled = self._token_scaler.transform(token_features)
                token_probs = self._token_classifier.predict_proba(X_scaled)[:, 1]
                
                # Aggregate to sample level
                aggregation = self.config.params.get("aggregation", "max")
                if aggregation == "max":
                    score = float(token_probs.max())
                elif aggregation == "mean":
                    score = float(token_probs.mean())
                else:  # any
                    score = float((token_probs > 0.5).any())
                
                return Prediction(
                    sample_id=features.sample_id,
                    score=score,
                    label=1 if score > 0.5 else 0,
                    confidence=abs(score - 0.5) * 2,
                )
            except InsufficientResponseError as e:
                logger.debug(f"Token prediction skipped due to insufficient response: {e}, falling back to sample")
            except Exception as e:
                logger.warning(f"Token prediction failed: {e}, falling back to sample")
        
        # Fallback to sample-level
        return super().predict(features)


@METHODS.register("attention_stats", aliases=["attn_stats"])
class AttentionStatsMethod(BaseMethod):
    """Simple attention statistics-based detection."""
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract simple attention statistics."""
        if features.attn_diags is None:
            raise ValueError("AttentionStats requires attn_diags")
        
        attn_diags = features.attn_diags
        if isinstance(attn_diags, np.ndarray):
            attn_diags = torch.from_numpy(attn_diags)
        
        resp_start = features.prompt_len
        resp_end = features.prompt_len + features.response_len
        
        if resp_end <= attn_diags.shape[-1]:
            resp_attn = attn_diags[:, :, resp_start:resp_end]
        else:
            resp_attn = attn_diags[:, :, resp_start:]
        
        flat = resp_attn.float().cpu().numpy().flatten()
        
        features_list = [
            np.mean(flat),
            np.std(flat),
            np.max(flat),
            np.min(flat),
            np.median(flat),
            np.percentile(flat, 25),
            np.percentile(flat, 75),
            np.percentile(flat, 90),
            np.percentile(flat, 10),
        ]
        
        n_layers = attn_diags.shape[0]
        for layer in range(n_layers):
            layer_flat = resp_attn[layer].float().cpu().numpy().flatten()
            features_list.extend([np.mean(layer_flat), np.std(layer_flat)])
        
        return np.array(features_list, dtype=np.float32)
