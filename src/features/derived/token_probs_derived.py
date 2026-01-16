"""Token Probability-based Derived Features - 基于Token概率的派生特征。

所有从 token_probs/logits 计算的派生特征：
- token_entropy: Token 级别的熵
- token_confidence: Token 置信度
- perplexity: 困惑度

Usage:
    from src.features.derived.token_probs_derived import (
        compute_token_entropy,
        compute_token_confidence,
        compute_sequence_perplexity,
    )
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Token Entropy - Token 熵
# =============================================================================

def compute_token_entropy(
    logits: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
    normalize: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """计算每个位置的 token 熵。
    
    H = -sum(p * log(p))
    熵越高表示模型越不确定。
    
    Args:
        logits: [batch, seq_len, vocab_size] 或 [seq_len, vocab_size]
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        normalize: 是否归一化（除以 log(vocab_size)）
        eps: 数值稳定性
        
    Returns:
        Tensor [seq_len] 或 [response_len]
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    
    seq_len, vocab_size = logits.shape
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    with torch.no_grad():
        relevant_logits = logits[start_idx:end_idx]
        probs = F.softmax(relevant_logits, dim=-1)
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        if normalize:
            max_entropy = np.log(vocab_size)
            entropy = entropy / max_entropy
    
    return entropy.cpu()


def compute_token_entropy_from_probs(
    probs: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """从概率分布计算熵。
    
    Args:
        probs: [seq_len, vocab_size] 概率分布
        normalize: 是否归一化
        eps: 数值稳定性
        
    Returns:
        Tensor [seq_len]
    """
    vocab_size = probs.shape[-1]
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    if normalize:
        max_entropy = np.log(vocab_size)
        entropy = entropy / max_entropy
    
    return entropy.cpu()


# =============================================================================
# 2. Token Confidence - Token 置信度
# =============================================================================

def compute_token_confidence(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> torch.Tensor:
    """计算每个 token 的置信度（选择该 token 的概率）。
    
    Args:
        logits: [batch, seq_len, vocab_size] 或 [seq_len, vocab_size]
        input_ids: [batch, seq_len] 或 [seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        Tensor [seq_len] 或 [response_len] - 每个 token 的概率
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    
    seq_len = logits.shape[0]
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    with torch.no_grad():
        # logits[i] 预测 token[i+1]
        relevant_logits = logits[start_idx:end_idx]
        probs = F.softmax(relevant_logits, dim=-1)
        
        # 获取实际 token 的概率
        target_ids = input_ids[start_idx + 1:end_idx + 1]
        
        # 对齐长度
        min_len = min(probs.shape[0], target_ids.shape[0])
        probs = probs[:min_len]
        target_ids = target_ids[:min_len]
        
        token_probs = probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_probs.cpu()


def compute_top_k_confidence(
    logits: torch.Tensor,
    k: int = 5,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
) -> Dict[str, torch.Tensor]:
    """计算 Top-K 置信度。
    
    Args:
        logits: [batch, seq_len, vocab_size]
        k: Top-K 数量
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        
    Returns:
        {
            "top_k_probs": Tensor [seq_len, k],
            "top_k_indices": Tensor [seq_len, k],
            "top_1_prob": Tensor [seq_len],
            "prob_gap": Tensor [seq_len] - top1 与 top2 的差距,
        }
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    
    seq_len, vocab_size = logits.shape
    k = min(k, vocab_size)
    
    if response_only and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    with torch.no_grad():
        relevant_logits = logits[start_idx:end_idx]
        probs = F.softmax(relevant_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
        
        top_1_prob = top_k_probs[:, 0]
        prob_gap = top_k_probs[:, 0] - top_k_probs[:, 1] if k >= 2 else top_k_probs[:, 0]
    
    return {
        "top_k_probs": top_k_probs.cpu(),
        "top_k_indices": top_k_indices.cpu(),
        "top_1_prob": top_1_prob.cpu(),
        "prob_gap": prob_gap.cpu(),
    }


# =============================================================================
# 3. Perplexity - 困惑度
# =============================================================================

def compute_sequence_perplexity(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
    eps: float = 1e-10,
) -> float:
    """计算序列的困惑度。
    
    PPL = exp(-1/N * sum(log(p_i)))
    
    Args:
        logits: [batch, seq_len, vocab_size]
        input_ids: [batch, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        eps: 数值稳定性
        
    Returns:
        困惑度值 (float)
    """
    token_probs = compute_token_confidence(
        logits, input_ids, prompt_len, response_len, response_only
    )
    
    log_probs = torch.log(token_probs + eps)
    avg_log_prob = log_probs.mean()
    perplexity = torch.exp(-avg_log_prob)
    
    return perplexity.item()


def compute_token_perplexity(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
    window_size: int = 1,
    eps: float = 1e-10,
) -> torch.Tensor:
    """计算每个 token 的局部困惑度。
    
    使用滑动窗口计算局部困惑度。
    
    Args:
        logits: [batch, seq_len, vocab_size]
        input_ids: [batch, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        window_size: 滑动窗口大小（1 = 单 token）
        eps: 数值稳定性
        
    Returns:
        Tensor [seq_len] - 每个位置的局部困惑度
    """
    token_probs = compute_token_confidence(
        logits, input_ids, prompt_len, response_len, response_only
    )
    
    if window_size <= 1:
        # 单 token 困惑度
        log_probs = torch.log(token_probs + eps)
        token_ppl = torch.exp(-log_probs)
    else:
        # 滑动窗口困惑度
        seq_len = token_probs.shape[0]
        token_ppl = torch.zeros(seq_len)
        log_probs = torch.log(token_probs + eps)
        
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            window_log_probs = log_probs[start:i+1]
            avg_log_prob = window_log_probs.mean()
            token_ppl[i] = torch.exp(-avg_log_prob)
    
    return token_ppl.cpu()


# =============================================================================
# 4. Uncertainty Metrics - 不确定性指标
# =============================================================================

def compute_uncertainty_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int = 0,
    response_len: int = 0,
    response_only: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """计算综合不确定性指标。
    
    Args:
        logits: [batch, seq_len, vocab_size]
        input_ids: [batch, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只计算 response 部分
        top_k: Top-K 数量
        
    Returns:
        {
            "token_entropy": Tensor,
            "token_confidence": Tensor,
            "perplexity": float,
            "top_k_probs": Tensor,
            "prob_gap": Tensor,
            "mean_entropy": float,
            "max_entropy": float,
            "low_confidence_ratio": float,  # 置信度 < 0.5 的比例
        }
    """
    # Token 熵
    token_entropy = compute_token_entropy(
        logits, prompt_len, response_len, response_only
    )
    
    # Token 置信度
    token_confidence = compute_token_confidence(
        logits, input_ids, prompt_len, response_len, response_only
    )
    
    # 困惑度
    perplexity = compute_sequence_perplexity(
        logits, input_ids, prompt_len, response_len, response_only
    )
    
    # Top-K
    top_k_results = compute_top_k_confidence(
        logits, top_k, prompt_len, response_len, response_only
    )
    
    # 统计量
    mean_entropy = token_entropy.mean().item()
    max_entropy = token_entropy.max().item()
    low_confidence_ratio = (token_confidence < 0.5).float().mean().item()
    
    return {
        "token_entropy": token_entropy,
        "token_confidence": token_confidence,
        "perplexity": perplexity,
        "top_k_probs": top_k_results["top_k_probs"],
        "prob_gap": top_k_results["prob_gap"],
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "low_confidence_ratio": low_confidence_ratio,
    }
