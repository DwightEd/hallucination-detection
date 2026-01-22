"""Lookback Lens: Attention Ratio-based Hallucination Detection.

基于论文: "Lookback Lens: Detecting and Mitigating Contextual Hallucinations 
in Large Language Models Using Only Attention Maps" (EMNLP 2024)
代码参考: https://github.com/voidism/Lookback-Lens

核心思想:
- 幻觉 token 通常对 context 的注意力较低
- Lookback ratio = A_context / (A_context + A_new)
- 支持 token 级别和 sample 级别检测
- 支持 Guided Decoding 缓解幻觉

⚠️ 关键修正:
1. Lookback ratio 计算: 对每个 response token t，计算其对 context (prompt) 的注意力比率
2. 特征维度: 每个 (layer, head) 产生一个 lookback ratio
3. Span-level 聚合: 使用滑动窗口 (默认 size=8)
4. 分类器: LogisticRegression
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction
from .base import BaseMethod

logger = logging.getLogger(__name__)


class InsufficientResponseError(ValueError):
    """当 response 长度不足以提取特征时抛出。"""
    pass


# =============================================================================
# 核心算法 - Lookback Ratio 计算
# =============================================================================

def compute_lookback_ratio_per_token(
    full_attention: np.ndarray,
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """计算每个 response token 的 lookback ratio。
    
    原论文公式: LR_t^{l,h} = A_context / (A_context + A_new)
    其中:
    - A_context = sum_{i < prompt_len} A[t, i]  (对 context 的注意力)
    - A_new = sum_{prompt_len <= i < t} A[t, i]  (对新生成 token 的注意力)
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        Lookback ratios [n_layers, n_heads, response_len]
    """
    n_layers, n_heads, seq_len, _ = full_attention.shape
    
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    actual_resp_len = resp_end - resp_start
    
    if actual_resp_len <= 0:
        raise InsufficientResponseError(
            f"Invalid response length: {actual_resp_len}"
        )
    
    lookback_ratios = np.zeros((n_layers, n_heads, actual_resp_len), dtype=np.float32)
    
    for t_idx, t in enumerate(range(resp_start, resp_end)):
        for layer in range(n_layers):
            for head in range(n_heads):
                # 对 context (prompt) 的注意力
                if prompt_len > 0:
                    attn_context = full_attention[layer, head, t, :prompt_len].sum()
                else:
                    attn_context = 0.0
                
                # 对新生成 token (prompt_len 到 t-1) 的注意力
                if t > prompt_len:
                    attn_new = full_attention[layer, head, t, prompt_len:t].sum()
                else:
                    attn_new = 0.0
                
                # Lookback ratio
                total = attn_context + attn_new + 1e-10
                lookback_ratios[layer, head, t_idx] = attn_context / total
    
    return lookback_ratios


def compute_token_lookback_ratios(
    attn_diags: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """从注意力对角线近似计算 lookback ratio (降级模式)。
    
    当没有完整注意力矩阵时，使用对角线值作为近似。
    
    Args:
        attn_diags: [n_layers, n_heads, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        Token-level features [resp_len, n_layers * n_heads * 2]
    """
    n_layers, n_heads, seq_len = attn_diags.shape
    attn_diags = attn_diags.float()
    
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    actual_resp_len = resp_end - resp_start
    
    if actual_resp_len <= 0:
        raise InsufficientResponseError(
            f"Invalid response length: actual_resp_len={actual_resp_len}"
        )
    
    token_features = []
    
    for t in range(resp_start, resp_end):
        diag_at_t = attn_diags[:, :, t]  # [n_layers, n_heads]
        
        # Context 部分的统计
        if t > 0:
            context_attn = attn_diags[:, :, :min(t, prompt_len)].mean(dim=-1)
        else:
            context_attn = torch.zeros(n_layers, n_heads)
        
        # 新生成部分的统计
        if t > prompt_len:
            new_attn = attn_diags[:, :, prompt_len:t].mean(dim=-1)
        else:
            new_attn = torch.zeros(n_layers, n_heads)
        
        # Lookback ratio
        total = context_attn + new_attn + 1e-8
        lookback_ratio = context_attn / total
        
        feat = torch.cat([diag_at_t.flatten(), lookback_ratio.flatten()])
        token_features.append(feat)
    
    return torch.stack(token_features)


def compute_lookback_ratio(
    attn_diags: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """计算 sample-level lookback ratio 特征。
    
    Args:
        attn_diags: [n_layers, n_heads, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        Feature vector [n_layers * n_heads * 4]
    """
    n_layers, n_heads, seq_len = attn_diags.shape
    attn_diags = attn_diags.float()
    
    prompt_diag = attn_diags[:, :, :prompt_len]
    
    if prompt_len + response_len <= seq_len:
        response_diag = attn_diags[:, :, prompt_len:prompt_len + response_len]
    else:
        response_diag = attn_diags[:, :, prompt_len:]
    
    p_mean = prompt_diag.mean(dim=-1) if prompt_diag.shape[-1] > 0 else torch.zeros(n_layers, n_heads)
    r_mean = response_diag.mean(dim=-1) if response_diag.shape[-1] > 0 else torch.zeros(n_layers, n_heads)
    
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
    """计算综合注意力模式特征。"""
    n_layers, n_heads, seq_len = attn_diags.shape
    
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    resp_diag = attn_diags[:, :, resp_start:resp_end].float()
    resp_len_actual = resp_diag.shape[-1]
    
    if resp_len_actual <= 1:
        n_features = 5 * n_layers * n_heads
        if attn_entropy is not None:
            n_features += 2 * n_layers * n_heads
        
        if resp_len_actual == 1:
            diag_val = resp_diag.squeeze(-1)
            diag_features = torch.stack([
                diag_val, torch.zeros_like(diag_val),
                diag_val, diag_val, torch.zeros_like(diag_val),
            ], dim=-1)
            
            if attn_entropy is not None:
                resp_entropy = attn_entropy[:, :, resp_start:resp_end].float()
                ent_val = resp_entropy.squeeze(-1)
                ent_features = torch.stack([ent_val, torch.zeros_like(ent_val)], dim=-1)
                features = torch.cat([diag_features, ent_features], dim=-1)
            else:
                features = diag_features
            
            return features.cpu().numpy().flatten().astype(np.float32)
        
        return np.zeros(n_features, dtype=np.float32)
    
    resp_np = resp_diag.cpu().numpy()
    
    diag_mean = np.mean(resp_np, axis=-1)
    diag_std = np.std(resp_np, axis=-1)
    diag_max = np.max(resp_np, axis=-1)
    diag_min = np.min(resp_np, axis=-1)
    
    # Trend
    x = np.arange(resp_len_actual)
    diag_trend = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            if resp_len_actual > 1:
                diag_trend[l, h] = np.polyfit(x, resp_np[l, h], 1)[0]
    
    diag_features = np.stack([diag_mean, diag_std, diag_max, diag_min, diag_trend], axis=-1)
    
    if attn_entropy is not None:
        resp_entropy = attn_entropy[:, :, resp_start:resp_end].float().cpu().numpy()
        ent_mean = np.mean(resp_entropy, axis=-1)
        ent_std = np.std(resp_entropy, axis=-1)
        ent_features = np.stack([ent_mean, ent_std], axis=-1)
        features = np.concatenate([diag_features, ent_features], axis=-1)
    else:
        features = diag_features
    
    return features.flatten().astype(np.float32)


def aggregate_span_features(
    token_features: np.ndarray,
    window_size: int = 8,
    aggregation: str = "mean",
) -> np.ndarray:
    """将 token-level 特征聚合到 span-level。
    
    Args:
        token_features: [n_tokens, feature_dim]
        window_size: 滑动窗口大小 (原论文默认 8)
        aggregation: 聚合方式 (mean, max)
        
    Returns:
        Span-level features [n_spans, feature_dim]
    """
    n_tokens, feature_dim = token_features.shape
    
    if n_tokens <= window_size:
        if aggregation == "mean":
            return token_features.mean(axis=0, keepdims=True)
        else:
            return token_features.max(axis=0, keepdims=True)
    
    spans = []
    for i in range(0, n_tokens - window_size + 1):
        window = token_features[i:i + window_size]
        if aggregation == "mean":
            span_feat = window.mean(axis=0)
        else:
            span_feat = window.max(axis=0)
        spans.append(span_feat)
    
    return np.array(spans)


# =============================================================================
# Lookback Lens Method 类
# =============================================================================

@METHODS.register("lookback_lens", aliases=["lookback", "attention_ratio"])
class LookbackLensMethod(BaseMethod):
    """Lookback Lens 幻觉检测方法。
    
    支持:
    - sample: 样本级别特征聚合 + LogisticRegression
    - token: 逐 token 特征 + 分类器
    - both: 优先 token，无标签时回退 sample
    """
    
    supports_token_level = True
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.use_entropy = params.get("use_entropy", True)
        self.use_trends = params.get("use_trends", True)
        self.window_size = params.get("window_size", 8)
        
        self._token_classifier = None
        self._token_scaler = StandardScaler()
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 sample-level lookback 特征。"""
        # 优先使用完整注意力
        full_attention = features.full_attention
        if full_attention is None:
            full_attention = features.get_full_attention()
        
        if full_attention is not None:
            if isinstance(full_attention, torch.Tensor):
                full_attention = full_attention.cpu().numpy()
            
            try:
                # 计算精确的 lookback ratio
                lookback_ratios = compute_lookback_ratio_per_token(
                    full_attention,
                    features.prompt_len,
                    features.response_len,
                )
                
                # 聚合到 sample level
                feat_vec = np.concatenate([
                    lookback_ratios.mean(axis=-1).flatten(),  # 平均
                    lookback_ratios.std(axis=-1).flatten(),   # 标准差
                    lookback_ratios.min(axis=-1).flatten(),   # 最小值
                    lookback_ratios.max(axis=-1).flatten(),   # 最大值
                ])
                
                features.release_large_features()
                
                if np.any(~np.isfinite(feat_vec)):
                    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
                
                return feat_vec
                
            except InsufficientResponseError:
                pass
        
        # 降级到对角线模式
        if features.attn_diags is None:
            raise ValueError("LookbackLens requires full_attention or attn_diags")
        
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
        """提取 token-level lookback 特征。"""
        # 优先使用完整注意力
        full_attention = features.full_attention
        if full_attention is None:
            full_attention = features.get_full_attention()
        
        if full_attention is not None:
            if isinstance(full_attention, torch.Tensor):
                full_attention = full_attention.cpu().numpy()
            
            lookback_ratios = compute_lookback_ratio_per_token(
                full_attention,
                features.prompt_len,
                features.response_len,
            )
            
            # [n_layers, n_heads, resp_len] -> [resp_len, n_layers * n_heads]
            token_features = lookback_ratios.transpose(2, 0, 1).reshape(
                lookback_ratios.shape[2], -1
            )
            
            features.release_large_features()
            return token_features
        
        # 降级模式
        if features.attn_diags is None:
            raise ValueError("LookbackLens requires full_attention or attn_diags")
        
        attn_diags = features.attn_diags
        if isinstance(attn_diags, np.ndarray):
            attn_diags = torch.from_numpy(attn_diags)
        
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
        """训练方法。"""
        level = self.config.level
        
        if level in ("token", "both"):
            # 尝试 token-level 训练
            token_X = []
            token_y = []
            
            for i, feat in enumerate(features_list):
                try:
                    token_feat = self.extract_token_features(feat)
                    
                    # 获取 token-level 标签
                    token_labels = getattr(feat, 'hallucination_labels', None)
                    
                    if token_labels is not None and len(token_labels) == len(token_feat):
                        token_X.append(token_feat)
                        token_y.append(token_labels)
                    elif labels is not None:
                        # 使用 sample-level 标签扩展
                        sample_label = labels[i] if i < len(labels) else feat.label
                        if sample_label is not None:
                            token_X.append(token_feat)
                            token_y.append([sample_label] * len(token_feat))
                except (InsufficientResponseError, Exception) as e:
                    logger.debug(f"Token feature extraction failed: {e}")
            
            if len(token_X) > 0:
                token_X = np.vstack(token_X)
                token_y = np.concatenate(token_y)
                
                token_X_scaled = self._token_scaler.fit_transform(token_X)
                
                self._token_classifier = LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=self.config.random_seed or 42,
                )
                self._token_classifier.fit(token_X_scaled, token_y)
                
                logger.info(f"Token-level classifier trained on {len(token_X)} tokens")
        
        # Sample-level 训练
        return super().fit(features_list, labels, cv)
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """预测单个样本。"""
        level = self.config.level
        
        if level in ("token", "both") and self._token_classifier is not None:
            try:
                token_features = self.extract_token_features(features)
                X_scaled = self._token_scaler.transform(token_features)
                token_probs = self._token_classifier.predict_proba(X_scaled)[:, 1]
                
                aggregation = self.config.params.get("aggregation", "max")
                if aggregation == "max":
                    score = float(token_probs.max())
                elif aggregation == "mean":
                    score = float(token_probs.mean())
                else:
                    score = float((token_probs > 0.5).any())
                
                return Prediction(
                    sample_id=features.sample_id,
                    score=score,
                    label=1 if score > 0.5 else 0,
                    confidence=abs(score - 0.5) * 2,
                )
            except (InsufficientResponseError, Exception) as e:
                logger.debug(f"Token prediction failed: {e}, falling back to sample")
        
        return super().predict(features)


@METHODS.register("attention_stats", aliases=["attn_stats"])
class AttentionStatsMethod(BaseMethod):
    """简单的注意力统计特征方法。"""
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取简单注意力统计特征。"""
        if features.attn_diags is None:
            raise ValueError("AttentionStats requires attn_diags")
        
        attn_diags = features.attn_diags
        if isinstance(attn_diags, np.ndarray):
            attn_diags = torch.from_numpy(attn_diags)
        
        n_layers, n_heads, seq_len = attn_diags.shape
        attn_diags = attn_diags.float()
        
        # 全局统计
        global_mean = attn_diags.mean().item()
        global_std = attn_diags.std().item()
        global_max = attn_diags.max().item()
        global_min = attn_diags.min().item()
        
        # 每层统计
        layer_means = attn_diags.mean(dim=(1, 2)).cpu().numpy()
        layer_stds = attn_diags.std(dim=(1, 2)).cpu().numpy()
        
        features = np.concatenate([
            [global_mean, global_std, global_max, global_min],
            layer_means,
            layer_stds,
        ])
        
        return features.astype(np.float32)