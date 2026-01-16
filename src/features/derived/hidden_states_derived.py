"""Hidden States-based Derived Features - 基于隐藏状态的派生特征。

所有从 hidden_states 计算的派生特征：
- pooled_states: 池化后的隐藏状态
- layer_similarity: 层间相似度
- representation_stats: 表示统计量

Usage:
    from src.features.derived.hidden_states_derived import (
        compute_pooled_states,
        compute_layer_similarity,
        compute_representation_stats,
    )
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum
import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class PoolingStrategy(Enum):
    """池化策略。"""
    NONE = "none"
    MEAN = "mean"
    MAX = "max"
    LAST = "last"
    FIRST = "first"
    RESPONSE_MEAN = "response_mean"
    CLS = "cls"


# =============================================================================
# 1. Pooled States - 池化隐藏状态
# =============================================================================

def compute_pooled_states(
    hidden_states: torch.Tensor,
    pooling: str = "mean",
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> torch.Tensor:
    """对隐藏状态进行池化。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
        pooling: 池化策略 (mean, max, last, first, response_mean)
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只对 response 部分池化
        
    Returns:
        Tensor [n_layers, hidden_dim] 或 [hidden_dim]
    """
    # 处理维度
    has_layer_dim = hidden_states.dim() == 3
    if not has_layer_dim:
        hidden_states = hidden_states.unsqueeze(0)
    
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    # 确定范围
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    relevant_states = hidden_states[:, start_idx:end_idx, :]
    
    # 池化
    if pooling == "mean" or pooling == "response_mean":
        pooled = relevant_states.mean(dim=1)
    elif pooling == "max":
        pooled = relevant_states.max(dim=1)[0]
    elif pooling == "last":
        pooled = relevant_states[:, -1, :]
    elif pooling == "first":
        pooled = relevant_states[:, 0, :]
    elif pooling == "cls":
        pooled = hidden_states[:, 0, :]  # 始终取第一个 token
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")
    
    if not has_layer_dim:
        pooled = pooled.squeeze(0)
    
    return pooled.cpu()


def compute_layerwise_pooled_states(
    hidden_states: torch.Tensor,
    layers: Optional[List[int]] = None,
    pooling: str = "mean",
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> torch.Tensor:
    """对指定层的隐藏状态进行池化。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        layers: 要使用的层（None = 所有层）
        pooling: 池化策略
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只对 response 部分池化
        
    Returns:
        Tensor [n_selected_layers, hidden_dim]
    """
    n_layers = hidden_states.shape[0]
    
    if layers is not None:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
        hidden_states = hidden_states[layer_indices]
    
    return compute_pooled_states(
        hidden_states, pooling, prompt_len, response_len, response_only
    )


# =============================================================================
# 2. Layer Similarity - 层间相似度
# =============================================================================

def compute_layer_similarity(
    hidden_states: torch.Tensor,
    method: str = "cosine",
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> torch.Tensor:
    """计算相邻层之间的相似度。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        method: 相似度方法 (cosine, l2, dot)
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        Tensor [n_layers-1, seq_len] - 相邻层的相似度
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    relevant_states = hidden_states[:, start_idx:end_idx, :]
    range_len = end_idx - start_idx
    
    similarity = torch.zeros(n_layers - 1, range_len)
    
    for i in range(n_layers - 1):
        layer_i = relevant_states[i]      # [range_len, hidden_dim]
        layer_j = relevant_states[i + 1]  # [range_len, hidden_dim]
        
        if method == "cosine":
            sim = F.cosine_similarity(layer_i, layer_j, dim=-1)
        elif method == "l2":
            sim = -torch.norm(layer_i - layer_j, p=2, dim=-1)
        elif method == "dot":
            sim = (layer_i * layer_j).sum(dim=-1)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        similarity[i] = sim
    
    return similarity.cpu()


def compute_layer_similarity_matrix(
    hidden_states: torch.Tensor,
    method: str = "cosine",
    pooling: str = "mean",
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> torch.Tensor:
    """计算所有层之间的相似度矩阵。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        method: 相似度方法
        pooling: 对序列维度的池化方式
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        Tensor [n_layers, n_layers] - 层间相似度矩阵
    """
    # 先池化
    pooled = compute_pooled_states(
        hidden_states, pooling, prompt_len, response_len, response_only
    )  # [n_layers, hidden_dim]
    
    n_layers = pooled.shape[0]
    sim_matrix = torch.zeros(n_layers, n_layers)
    
    for i in range(n_layers):
        for j in range(n_layers):
            if method == "cosine":
                sim = F.cosine_similarity(
                    pooled[i].unsqueeze(0), pooled[j].unsqueeze(0)
                ).item()
            elif method == "l2":
                sim = -torch.norm(pooled[i] - pooled[j], p=2).item()
            elif method == "dot":
                sim = (pooled[i] * pooled[j]).sum().item()
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            sim_matrix[i, j] = sim
    
    return sim_matrix.cpu()


# =============================================================================
# 3. Representation Statistics - 表示统计量
# =============================================================================

def compute_representation_stats(
    hidden_states: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """计算隐藏状态的统计量。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        {
            "mean": 均值,
            "std": 标准差,
            "norm": L2 范数,
            "max": 最大值,
            "min": 最小值,
        }
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    relevant_states = hidden_states[:, start_idx:end_idx, :]
    
    return {
        "mean": relevant_states.mean(dim=-1).cpu(),        # [n_layers, seq_len]
        "std": relevant_states.std(dim=-1).cpu(),          # [n_layers, seq_len]
        "norm": relevant_states.norm(dim=-1).cpu(),        # [n_layers, seq_len]
        "max": relevant_states.max(dim=-1)[0].cpu(),       # [n_layers, seq_len]
        "min": relevant_states.min(dim=-1)[0].cpu(),       # [n_layers, seq_len]
    }


def compute_representation_drift(
    hidden_states: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
) -> torch.Tensor:
    """计算表示漂移（response 相对于 prompt 的变化）。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        Tensor [n_layers, response_len] - 每个位置相对于 prompt 均值的距离
    """
    if prompt_len <= 0:
        raise ValueError("Representation drift requires prompt_len > 0")
    
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    # Prompt 的均值表示
    prompt_states = hidden_states[:, :prompt_len, :]  # [n_layers, prompt_len, hidden_dim]
    prompt_mean = prompt_states.mean(dim=1)  # [n_layers, hidden_dim]
    
    # Response 部分
    if response_len <= 0:
        response_len = seq_len - prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    response_states = hidden_states[:, prompt_len:resp_end, :]  # [n_layers, resp_len, hidden_dim]
    
    # 计算每个 response token 与 prompt 均值的距离
    actual_resp_len = resp_end - prompt_len
    drift = torch.zeros(n_layers, actual_resp_len)
    
    for i in range(actual_resp_len):
        resp_state = response_states[:, i, :]  # [n_layers, hidden_dim]
        dist = torch.norm(resp_state - prompt_mean, dim=-1)  # [n_layers]
        drift[:, i] = dist
    
    return drift.cpu()


# =============================================================================
# 4. SVD-based Features (for HaloScope) - SVD 特征
# =============================================================================

def compute_svd_features(
    hidden_states: torch.Tensor,
    n_components: int = 10,
    layers: Optional[List[int]] = None,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """计算 SVD 分解特征（用于 HaloScope 等方法）。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        n_components: 保留的主成分数量
        layers: 要使用的层（None = 后半层）
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        {
            "singular_values": 奇异值,
            "explained_variance_ratio": 解释方差比,
            "projected_states": 投影后的状态,
        }
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    # 默认使用后半层
    if layers is None:
        layer_indices = list(range(n_layers // 2, n_layers))
    else:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    # 选择并展平
    selected_states = hidden_states[layer_indices][:, start_idx:end_idx, :]
    # [n_selected_layers, range_len, hidden_dim]
    
    range_len = end_idx - start_idx
    flat_states = selected_states.reshape(-1, hidden_dim)  # [n_layers * range_len, hidden_dim]
    
    # 中心化
    mean_state = flat_states.mean(dim=0)
    centered_states = flat_states - mean_state
    
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(centered_states, full_matrices=False)
        
        # 保留前 n_components
        n_components = min(n_components, S.shape[0])
        singular_values = S[:n_components]
        
        # 解释方差比
        total_variance = (S ** 2).sum()
        explained_variance_ratio = (singular_values ** 2) / total_variance
        
        # 投影
        projected = centered_states @ Vh[:n_components].T  # [n_samples, n_components]
        projected_states = projected.reshape(len(layer_indices), range_len, n_components)
        
        return {
            "singular_values": singular_values.cpu(),
            "explained_variance_ratio": explained_variance_ratio.cpu(),
            "projected_states": projected_states.cpu(),
            "mean_state": mean_state.cpu(),
            "components": Vh[:n_components].cpu(),
        }
    except Exception as e:
        logger.warning(f"SVD computation failed: {e}")
        return {
            "singular_values": torch.zeros(n_components),
            "explained_variance_ratio": torch.zeros(n_components),
            "projected_states": torch.zeros(len(layer_indices), range_len, n_components),
        }
