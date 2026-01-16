"""Multi-View Attention Feature Extraction.

基于论文 "Hallucinated Span Detection with Multi-View Attention Features"
从注意力矩阵中提取三种互补特征：

1. Average Incoming Attention (avg_in): 每个token收到的平均注意力权重
2. Diversity of Incoming Attention (div_in): 入向注意力的多样性
3. Diversity of Outgoing Attention (div_out): 出向注意力的多样性
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch


def compute_multi_view_attention_features(
    attention: torch.Tensor,
    prompt_len: int,
    response_len: int,
    normalize: bool = True,
) -> torch.Tensor:
    """计算 Multi-View Attention Features。
    
    根据论文从注意力矩阵中提取三种互补特征。
    
    Args:
        attention: 注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
                  或 [n_heads, seq_len, seq_len] (单层)
        prompt_len: Prompt 长度
        response_len: Response 长度
        normalize: 是否标准化特征
        
    Returns:
        Multi-view features [resp_len, feature_dim]
        feature_dim = n_layers * n_heads * 3 (avg_in, div_in, div_out)
    """
    if isinstance(attention, np.ndarray):
        attention = torch.from_numpy(attention)
    
    attention = attention.float()
    
    # 处理不同形状
    if len(attention.shape) == 3:
        attention = attention.unsqueeze(0)
    
    n_layers, n_heads, seq_len, _ = attention.shape
    
    # 确定 response 范围
    resp_start = min(prompt_len, seq_len)
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    actual_resp_len = resp_end - resp_start
    
    features_list = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # 当前层和头的注意力矩阵
            attn = attention[layer, head]  # [seq_len, seq_len]
            
            # 只考虑 response 部分的 token
            resp_attn = attn[resp_start:resp_end, :]  # [resp_len, seq_len]
            
            # 特征 1: Average Incoming Attention
            # 每个 response token 收到的注意力（来自其他 token）
            avg_in = attn[:, resp_start:resp_end].mean(dim=0)  # [resp_len]
            
            # 特征 2: Diversity of Incoming Attention (使用熵)
            incoming_attn = attn[:, resp_start:resp_end]  # [seq_len, resp_len]
            incoming_attn = incoming_attn / (incoming_attn.sum(dim=0, keepdim=True) + 1e-8)
            div_in = -torch.sum(
                incoming_attn * torch.log(incoming_attn + 1e-8), 
                dim=0
            )  # [resp_len]
            
            # 特征 3: Diversity of Outgoing Attention
            outgoing_attn = resp_attn / (resp_attn.sum(dim=1, keepdim=True) + 1e-8)
            div_out = -torch.sum(
                outgoing_attn * torch.log(outgoing_attn + 1e-8), 
                dim=1
            )  # [resp_len]
            
            # 拼接这三个特征
            layer_head_features = torch.stack([avg_in, div_in, div_out], dim=1)  # [resp_len, 3]
            features_list.append(layer_head_features)
    
    # 拼接所有层和头的特征
    all_features = torch.cat(features_list, dim=1)
    
    # 标准化
    if normalize:
        mean = all_features.mean(dim=0, keepdim=True)
        std = all_features.std(dim=0, keepdim=True) + 1e-8
        all_features = (all_features - mean) / std
    
    return all_features


def compute_mva_features_from_diags(
    attn_diags: torch.Tensor,
    attn_entropy: Optional[torch.Tensor],
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """从对角线特征近似计算 MVA 特征。
    
    当没有完整注意力矩阵时，使用对角线特征进行近似。
    
    Args:
        attn_diags: 注意力对角线 [n_layers, n_heads, seq_len]
        attn_entropy: 注意力熵 [n_layers, n_heads, seq_len] (可选)
        prompt_len: Prompt 长度
        response_len: Response 长度
        
    Returns:
        近似的 MVA features [resp_len, feature_dim]
    """
    if isinstance(attn_diags, np.ndarray):
        attn_diags = torch.from_numpy(attn_diags)
    
    attn_diags = attn_diags.float()
    
    n_layers, n_heads, seq_len = attn_diags.shape
    
    # 确定 response 范围
    resp_start = min(prompt_len, seq_len)
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    # 提取 response 部分的对角线
    resp_diags = attn_diags[:, :, resp_start:resp_end]
    
    features_list = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            diag = resp_diags[layer, head]
            
            # 特征 1: 对角线值（自注意力强度）
            feat1 = diag
            
            # 特征 2: 对角线值的局部变化（近似 diversity）
            if len(diag) > 1:
                diff = torch.zeros_like(diag)
                diff[1:] = torch.abs(diag[1:] - diag[:-1])
                feat2 = diff
            else:
                feat2 = torch.zeros_like(diag)
            
            # 特征 3: 累积统计（近似 incoming attention）
            feat3 = torch.cumsum(diag, dim=0) / (torch.arange(len(diag), dtype=torch.float32, device=diag.device) + 1)
            
            layer_head_features = torch.stack([feat1, feat2, feat3], dim=1)
            features_list.append(layer_head_features)
    
    # 添加 attention entropy（如果有）
    if attn_entropy is not None:
        if isinstance(attn_entropy, np.ndarray):
            attn_entropy = torch.from_numpy(attn_entropy)
        
        resp_entropy = attn_entropy[:, :, resp_start:resp_end].float()
        
        for layer in range(n_layers):
            for head in range(n_heads):
                entropy = resp_entropy[layer, head]
                features_list.append(entropy.unsqueeze(1))
    
    all_features = torch.cat(features_list, dim=1)
    
    # 标准化
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True) + 1e-8
    all_features = (all_features - mean) / std
    
    return all_features
