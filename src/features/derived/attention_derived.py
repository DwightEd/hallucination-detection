"""Attention-based Derived Features - 基于注意力的派生特征。

所有从 full_attention 计算的派生特征：
- attention_diags: 注意力对角线
- laplacian_diags: Laplacian 对角线  
- attention_entropy: 注意力熵
- lookback_ratio: Lookback 比率
- mva_features: Multi-View Attention 特征

Usage:
    from src.features.derived.attention_derived import (
        compute_attention_diags,
        compute_laplacian_diags,
        compute_attention_entropy,
        compute_lookback_ratio,
        compute_mva_features,
    )
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Attention Diagonals - 注意力对角线
# =============================================================================

@dataclass
class AttentionDiagsConfig:
    """注意力对角线提取配置。"""
    layers: Optional[List[int]] = None
    heads: Optional[List[int]] = None
    head_aggregation: str = "none"  # none, mean, max
    layer_aggregation: str = "none"  # none, mean, max, last
    scope: str = "response"  # full, prompt, response
    store_on_cpu: bool = True


def compute_attention_diags(
    full_attention: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    response_only: bool = True,
    head_aggregation: str = "none",
) -> torch.Tensor:
    """计算注意力对角线。
    
    对角线元素 attn[i,i] 表示 token i 对自身的注意力权重。
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        layers: 要使用的层（None = 所有层）
        heads: 要使用的头（None = 所有头）
        response_only: 是否只计算 response 部分
        head_aggregation: 头聚合方式
        
    Returns:
        Tensor [n_layers, n_heads, seq_len]
    """
    n_layers, n_heads, seq_len, _ = full_attention.shape
    
    # 选择层
    if layers is not None:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    else:
        layer_indices = list(range(n_layers))
    
    # 选择头
    if heads is not None:
        head_indices = [h if h >= 0 else n_heads + h for h in heads]
        head_indices = [h for h in head_indices if 0 <= h < n_heads]
    else:
        head_indices = list(range(n_heads))
    
    # 确定范围
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    # 提取对角线
    selected_attn = full_attention[layer_indices][:, head_indices]
    range_len = end_idx - start_idx
    diags = torch.zeros(len(layer_indices), len(head_indices), range_len)
    
    for i, pos in enumerate(range(start_idx, end_idx)):
        diags[:, :, i] = selected_attn[:, :, pos, pos]
    
    # 头聚合
    if head_aggregation == "mean":
        diags = diags.mean(dim=1, keepdim=True)
    elif head_aggregation == "max":
        diags = diags.max(dim=1, keepdim=True)[0]
    
    return diags.cpu()


def compute_attention_diags_direct(
    attentions: tuple,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    response_only: bool = True,
) -> torch.Tensor:
    """直接从模型 attentions 计算对角线（节省内存）。"""
    n_total_layers = len(attentions)
    
    if layers is None:
        layer_indices = list(range(n_total_layers))
    else:
        layer_indices = [l if l >= 0 else n_total_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_total_layers]
    
    diag_list = []
    
    for layer_idx in layer_indices:
        attn = attentions[layer_idx]
        if attn.dim() == 4:
            attn = attn.squeeze(0)
        
        seq_len = attn.shape[-1]
        
        if response_only and prompt_len > 0:
            start_idx = prompt_len
            end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
        else:
            start_idx = 0
            end_idx = seq_len
        
        range_len = end_idx - start_idx
        layer_diags = torch.zeros(attn.shape[0], range_len)
        
        for i, pos in enumerate(range(start_idx, end_idx)):
            layer_diags[:, i] = attn[:, pos, pos]
        
        diag_list.append(layer_diags.cpu())
        del attn
    
    return torch.stack(diag_list, dim=0)


# =============================================================================
# 2. Laplacian Diagonals - Laplacian 对角线
# =============================================================================

def compute_laplacian_diags(
    full_attention: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    response_only: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """计算 Laplacian 对角线。
    
    L = D - A，对角线 L[i,i] = degree[i] - self_attention[i,i]
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        layers: 要使用的层
        heads: 要使用的头
        response_only: 是否只计算 response 部分
        normalize: 是否归一化
        
    Returns:
        Tensor [n_layers, n_heads, seq_len]
    """
    n_layers, n_heads, seq_len, _ = full_attention.shape
    
    if layers is not None:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    else:
        layer_indices = list(range(n_layers))
    
    if heads is not None:
        head_indices = [h if h >= 0 else n_heads + h for h in heads]
        head_indices = [h for h in head_indices if 0 <= h < n_heads]
    else:
        head_indices = list(range(n_heads))
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    selected_attn = full_attention[layer_indices][:, head_indices]
    range_len = end_idx - start_idx
    lap_diags = torch.zeros(len(layer_indices), len(head_indices), range_len)
    
    for i, pos in enumerate(range(start_idx, end_idx)):
        out_degree = selected_attn[:, :, pos, :].sum(dim=-1)
        self_attn = selected_attn[:, :, pos, pos]
        lap_diags[:, :, i] = out_degree - self_attn
    
    if normalize:
        for l in range(lap_diags.shape[0]):
            for h in range(lap_diags.shape[1]):
                vals = lap_diags[l, h]
                min_val, max_val = vals.min(), vals.max()
                if max_val > min_val:
                    lap_diags[l, h] = (vals - min_val) / (max_val - min_val)
    
    return lap_diags.cpu()


def compute_laplacian_diags_direct(
    attentions: tuple,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    response_only: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """直接从模型 attentions 计算 Laplacian 对角线。"""
    n_total_layers = len(attentions)
    
    if layers is None:
        layer_indices = list(range(n_total_layers))
    else:
        layer_indices = [l if l >= 0 else n_total_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_total_layers]
    
    lap_list = []
    
    for layer_idx in layer_indices:
        attn = attentions[layer_idx]
        if attn.dim() == 4:
            attn = attn.squeeze(0)
        
        seq_len = attn.shape[-1]
        n_heads = attn.shape[0]
        
        if response_only and prompt_len > 0:
            start_idx = prompt_len
            end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
        else:
            start_idx = 0
            end_idx = seq_len
        
        range_len = end_idx - start_idx
        layer_lap = torch.zeros(n_heads, range_len)
        
        for i, pos in enumerate(range(start_idx, end_idx)):
            out_degree = attn[:, pos, :].sum(dim=-1)
            self_attn = attn[:, pos, pos]
            layer_lap[:, i] = out_degree - self_attn
        
        if normalize:
            for h in range(n_heads):
                vals = layer_lap[h]
                min_val, max_val = vals.min(), vals.max()
                if max_val > min_val:
                    layer_lap[h] = (vals - min_val) / (max_val - min_val)
        
        lap_list.append(layer_lap.cpu())
        del attn
    
    return torch.stack(lap_list, dim=0)


# =============================================================================
# 3. Attention Entropy - 注意力熵
# =============================================================================

def compute_attention_entropy(
    full_attention: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    response_only: bool = True,
    normalize: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """计算注意力熵。
    
    H(attn[i]) = -sum_j(attn[i,j] * log(attn[i,j]))
    熵越高表示注意力分布越均匀。
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        layers: 要使用的层
        heads: 要使用的头
        response_only: 是否只计算 response 部分
        normalize: 是否归一化（除以 log(seq_len)）
        eps: 数值稳定性
        
    Returns:
        Tensor [n_layers, n_heads, seq_len]
    """
    n_layers, n_heads, seq_len, _ = full_attention.shape
    
    if layers is not None:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    else:
        layer_indices = list(range(n_layers))
    
    if heads is not None:
        head_indices = [h if h >= 0 else n_heads + h for h in heads]
        head_indices = [h for h in head_indices if 0 <= h < n_heads]
    else:
        head_indices = list(range(n_heads))
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    selected_attn = full_attention[layer_indices][:, head_indices]
    range_len = end_idx - start_idx
    entropy = torch.zeros(len(layer_indices), len(head_indices), range_len)
    
    for i, pos in enumerate(range(start_idx, end_idx)):
        attn_row = selected_attn[:, :, pos, :] + eps
        row_entropy = -torch.sum(attn_row * torch.log(attn_row), dim=-1)
        entropy[:, :, i] = row_entropy
    
    if normalize:
        max_entropy = np.log(seq_len)
        entropy = entropy / max_entropy
    
    return entropy.cpu()


def compute_attention_entropy_direct(
    attentions: tuple,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    response_only: bool = True,
    normalize: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """直接从模型 attentions 计算注意力熵。"""
    n_total_layers = len(attentions)
    
    if layers is None:
        layer_indices = list(range(n_total_layers))
    else:
        layer_indices = [l if l >= 0 else n_total_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_total_layers]
    
    entropy_list = []
    
    for layer_idx in layer_indices:
        attn = attentions[layer_idx]
        if attn.dim() == 4:
            attn = attn.squeeze(0)
        
        seq_len = attn.shape[-1]
        
        if response_only and prompt_len > 0:
            start_idx = prompt_len
            end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
        else:
            start_idx = 0
            end_idx = seq_len
        
        attn_rows = attn[:, start_idx:end_idx, :] + eps
        layer_entropy = -torch.sum(attn_rows * torch.log(attn_rows), dim=-1)
        
        if normalize:
            max_entropy = np.log(seq_len)
            layer_entropy = layer_entropy / max_entropy
        
        entropy_list.append(layer_entropy.cpu())
        del attn, attn_rows
    
    return torch.stack(entropy_list, dim=0)


# =============================================================================
# 4. Lookback Ratio - Lookback 比率
# =============================================================================

def compute_lookback_ratio(
    full_attention: torch.Tensor,
    prompt_len: int,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    head_aggregation: str = "none",
    include_self: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """计算 Lookback 比率。
    
    Lookback Ratio = attn_to_prompt / total_attn
    高 lookback ratio 表示更多关注上下文。
    
    ⚠️ 需要 prompt 部分的 attention！
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度（必需）
        response_len: Response 长度
        layers: 要使用的层
        heads: 要使用的头
        head_aggregation: 头聚合方式
        include_self: 是否包含自身注意力
        eps: 数值稳定性
        
    Returns:
        Tensor [n_layers, n_heads, response_len]
    """
    if prompt_len <= 0:
        raise ValueError("Lookback ratio requires prompt_len > 0")
    
    n_layers, n_heads, seq_len, _ = full_attention.shape
    
    if response_len <= 0:
        response_len = seq_len - prompt_len
    
    if layers is not None:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    else:
        layer_indices = list(range(n_layers))
    
    if heads is not None:
        head_indices = [h if h >= 0 else n_heads + h for h in heads]
        head_indices = [h for h in head_indices if 0 <= h < n_heads]
    else:
        head_indices = list(range(n_heads))
    
    selected_attn = full_attention[layer_indices][:, head_indices]
    
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    actual_resp_len = resp_end - resp_start
    
    lookback_ratio = torch.zeros(len(layer_indices), len(head_indices), actual_resp_len)
    
    for i, pos in enumerate(range(resp_start, resp_end)):
        attn_row = selected_attn[:, :, pos, :]
        attn_to_prompt = attn_row[:, :, :prompt_len].sum(dim=-1)
        
        if include_self:
            attn_to_response = attn_row[:, :, prompt_len:pos+1].sum(dim=-1)
        else:
            attn_to_response = attn_row[:, :, prompt_len:pos].sum(dim=-1)
        
        total_attn = attn_to_prompt + attn_to_response + eps
        lookback_ratio[:, :, i] = attn_to_prompt / total_attn
    
    if head_aggregation == "mean":
        lookback_ratio = lookback_ratio.mean(dim=1, keepdim=True)
    elif head_aggregation == "max":
        lookback_ratio = lookback_ratio.max(dim=1, keepdim=True)[0]
    
    return lookback_ratio.cpu()


def compute_lookback_ratio_direct(
    attentions: tuple,
    prompt_len: int,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    include_self: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """直接从模型 attentions 计算 Lookback 比率。"""
    if prompt_len <= 0:
        raise ValueError("Lookback ratio requires prompt_len > 0")
    
    n_total_layers = len(attentions)
    
    if layers is None:
        layer_indices = list(range(n_total_layers))
    else:
        layer_indices = [l if l >= 0 else n_total_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_total_layers]
    
    lookback_list = []
    
    for layer_idx in layer_indices:
        attn = attentions[layer_idx]
        if attn.dim() == 4:
            attn = attn.squeeze(0)
        
        seq_len = attn.shape[-1]
        n_heads = attn.shape[0]
        
        if response_len <= 0:
            response_len = seq_len - prompt_len
        
        resp_start = prompt_len
        resp_end = min(prompt_len + response_len, seq_len)
        actual_resp_len = resp_end - resp_start
        
        layer_lookback = torch.zeros(n_heads, actual_resp_len)
        
        for i, pos in enumerate(range(resp_start, resp_end)):
            attn_row = attn[:, pos, :]
            attn_to_prompt = attn_row[:, :prompt_len].sum(dim=-1)
            
            if include_self:
                attn_to_response = attn_row[:, prompt_len:pos+1].sum(dim=-1)
            else:
                attn_to_response = attn_row[:, prompt_len:pos].sum(dim=-1)
            
            total_attn = attn_to_prompt + attn_to_response + eps
            layer_lookback[:, i] = attn_to_prompt / total_attn
        
        lookback_list.append(layer_lookback.cpu())
        del attn
    
    return torch.stack(lookback_list, dim=0)


# =============================================================================
# 5. MVA Features - Multi-View Attention 特征
# =============================================================================

def compute_mva_features(
    full_attention: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    head_aggregation: str = "mean",
    layer_aggregation: str = "mean",
    threshold: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """计算 MVA 特征。
    
    三种视角分析注意力：
    - avg_in: 每个 token 接收到的平均注意力
    - div_in: 接收注意力的多样性
    - div_out: 发出注意力的多样性
    
    ⚠️ 需要完整的 attention 矩阵！
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        layers: 要使用的层（None = 最后 4 层）
        heads: 要使用的头
        head_aggregation: 头聚合方式
        layer_aggregation: 层聚合方式
        threshold: diversity 阈值
        
    Returns:
        {"avg_in": Tensor, "div_in": Tensor, "div_out": Tensor, "combined": Tensor}
    """
    n_layers, n_heads, seq_len, _ = full_attention.shape
    eps = 1e-10
    
    # 默认使用最后 4 层
    if layers is None:
        layer_indices = list(range(max(0, n_layers - 4), n_layers))
    else:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    
    if heads is not None:
        head_indices = [h if h >= 0 else n_heads + h for h in heads]
        head_indices = [h for h in head_indices if 0 <= h < n_heads]
    else:
        head_indices = list(range(n_heads))
    
    if prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    range_len = end_idx - start_idx
    selected_attn = full_attention[layer_indices][:, head_indices]
    
    avg_in = torch.zeros(len(layer_indices), len(head_indices), range_len)
    div_in = torch.zeros(len(layer_indices), len(head_indices), range_len)
    div_out = torch.zeros(len(layer_indices), len(head_indices), range_len)
    
    for i, pos in enumerate(range(start_idx, end_idx)):
        # avg_in: 位置 pos 接收到的平均注意力
        incoming_attn = selected_attn[:, :, :pos+1, pos]
        avg_in[:, :, i] = incoming_attn.mean(dim=-1)
        
        # div_in: 有多少位置显著关注 pos
        significant_in = (incoming_attn > threshold).float().sum(dim=-1)
        div_in[:, :, i] = significant_in / (pos + 1 + eps)
        
        # div_out: 位置 pos 显著关注多少位置
        outgoing_attn = selected_attn[:, :, pos, :pos+1]
        significant_out = (outgoing_attn > threshold).float().sum(dim=-1)
        div_out[:, :, i] = significant_out / (pos + 1 + eps)
    
    # 聚合
    if head_aggregation == "mean":
        avg_in = avg_in.mean(dim=1, keepdim=True)
        div_in = div_in.mean(dim=1, keepdim=True)
        div_out = div_out.mean(dim=1, keepdim=True)
    elif head_aggregation == "max":
        avg_in = avg_in.max(dim=1, keepdim=True)[0]
        div_in = div_in.max(dim=1, keepdim=True)[0]
        div_out = div_out.max(dim=1, keepdim=True)[0]
    
    if layer_aggregation == "mean":
        avg_in = avg_in.mean(dim=0, keepdim=True)
        div_in = div_in.mean(dim=0, keepdim=True)
        div_out = div_out.mean(dim=0, keepdim=True)
    elif layer_aggregation == "max":
        avg_in = avg_in.max(dim=0, keepdim=True)[0]
        div_in = div_in.max(dim=0, keepdim=True)[0]
        div_out = div_out.max(dim=0, keepdim=True)[0]
    
    combined = torch.stack([avg_in, div_in, div_out], dim=-1)
    
    return {
        "avg_in": avg_in.cpu(),
        "div_in": div_in.cpu(),
        "div_out": div_out.cpu(),
        "combined": combined.cpu(),
    }


def compute_mva_features_direct(
    attentions: tuple,
    prompt_len: int = 0,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    threshold: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """直接从模型 attentions 计算 MVA 特征。"""
    n_total_layers = len(attentions)
    eps = 1e-10
    
    if layers is None:
        layer_indices = list(range(max(0, n_total_layers - 4), n_total_layers))
    else:
        layer_indices = [l if l >= 0 else n_total_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_total_layers]
    
    sample_attn = attentions[layer_indices[0]]
    if sample_attn.dim() == 4:
        seq_len = sample_attn.shape[-1]
        n_heads = sample_attn.shape[1]
    else:
        seq_len = sample_attn.shape[-1]
        n_heads = sample_attn.shape[0]
    
    if response_len <= 0:
        response_len = seq_len - prompt_len if prompt_len > 0 else seq_len
    
    start_idx = prompt_len if prompt_len > 0 else 0
    end_idx = min(start_idx + response_len, seq_len)
    range_len = end_idx - start_idx
    
    avg_in_list, div_in_list, div_out_list = [], [], []
    
    for layer_idx in layer_indices:
        attn = attentions[layer_idx]
        if attn.dim() == 4:
            attn = attn.squeeze(0)
        
        layer_avg_in = torch.zeros(n_heads, range_len)
        layer_div_in = torch.zeros(n_heads, range_len)
        layer_div_out = torch.zeros(n_heads, range_len)
        
        for i, pos in enumerate(range(start_idx, end_idx)):
            incoming = attn[:, :pos+1, pos]
            layer_avg_in[:, i] = incoming.mean(dim=-1)
            layer_div_in[:, i] = (incoming > threshold).float().sum(dim=-1) / (pos + 1 + eps)
            
            outgoing = attn[:, pos, :pos+1]
            layer_div_out[:, i] = (outgoing > threshold).float().sum(dim=-1) / (pos + 1 + eps)
        
        avg_in_list.append(layer_avg_in.cpu())
        div_in_list.append(layer_div_in.cpu())
        div_out_list.append(layer_div_out.cpu())
        del attn
    
    return {
        "avg_in": torch.stack(avg_in_list, dim=0),
        "div_in": torch.stack(div_in_list, dim=0),
        "div_out": torch.stack(div_out_list, dim=0),
    }
