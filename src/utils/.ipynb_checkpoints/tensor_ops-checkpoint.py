"""Tensor operations for feature extraction.

Provides utility functions for:
- Attention matrix operations (diagonal, laplacian, entropy)
- Hidden state pooling and stacking
- Token probability computation
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple


# ==============================================================================
# Attention Operations
# ==============================================================================

def extract_attention_diagonal(attention: Tensor) -> Tensor:
    """Extract diagonal from attention matrix.
    
    Args:
        attention: [batch, heads, seq, seq] or [heads, seq, seq]
        
    Returns:
        Diagonal values with same leading dims, last dim is seq
    """
    return torch.diagonal(attention, dim1=-2, dim2=-1)


def compute_attention_row_sums(attention: Tensor) -> Tensor:
    """Compute row sums (degree) from attention matrix.
    
    This is used by lapeigvals method to compute Laplacian eigenvalues.
    
    Args:
        attention: [batch, heads, seq, seq] or [heads, seq, seq]
        
    Returns:
        Row sums with same leading dims, last dim is seq
    """
    return attention.sum(dim=-1)


def compute_laplacian_diagonal(attention: Tensor) -> Tensor:
    """Compute Laplacian diagonal (eigenvalues) from attention matrix.
    
    严格按照 EMNLP 2025 论文 "Hallucination Detection in LLMs Using Spectral 
    Features of Attention Maps" 实现:
    
    公式:
    - 出度: d_ii = Σ_{u>i} a_ui / (T - i)  [式(2)]
    - Laplacian: L = D - A  [式(1)]
    - 由于L是下三角, 特征值 = 对角线: λ_i = d_ii - a_ii  [式(3)]
    
    注意: 这里的"出度"是指后续token对当前token的平均注意力
    (即列和中只取下三角部分, 除以后续token数量)
    
    Args:
        attention: [batch, heads, seq, seq] or [heads, seq, seq]
                  下三角注意力矩阵, 行和为1
        
    Returns:
        Laplacian diagonal (eigenvalues) with same leading dims except last
        即 [batch, heads, seq] 或 [heads, seq]
    """
    seq_len = attention.shape[-1]
    
    if seq_len == 0:
        return torch.zeros_like(attention[..., 0])
    
    device = attention.device
    dtype = attention.dtype
    original_shape = attention.shape[:-1]
    
    # 提取注意力对角线 a_ii
    attn_diag = extract_attention_diagonal(attention)  # [..., seq]
    
    # 向量化计算出度
    # 创建下三角掩码 (只保留 u > i 的部分)
    # mask[u, i] = 1 if u > i else 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=-1)
    
    # 计算每列的下三角部分和 (u > i 的 a_ui 之和)
    # attention: [..., seq, seq], mask: [seq, seq]
    masked_attention = attention * mask  # 广播乘法
    col_sums = masked_attention.sum(dim=-2)  # [..., seq] - 沿行维度求和得到列和
    
    # 计算分母: T - i for each position i
    denominators = torch.arange(seq_len, 0, -1, device=device, dtype=dtype)  # [T, T-1, ..., 1]
    
    # 出度 = 列和 / (T - i)
    out_degrees = col_sums / denominators
    
    # Laplacian 对角线 = 出度 - 注意力对角线
    laplacian_diag = out_degrees - attn_diag
    
    return laplacian_diag


def compute_laplacian_diagonal_slow(attention: Tensor) -> Tensor:
    """Compute Laplacian diagonal - 非向量化版本 (用于验证)。
    
    与 compute_laplacian_diagonal 功能相同，但使用显式循环实现。
    主要用于验证向量化版本的正确性。
    """
    seq_len = attention.shape[-1]
    
    if seq_len == 0:
        return torch.zeros_like(attention[..., 0])
    
    original_shape = attention.shape[:-1]
    device = attention.device
    dtype = attention.dtype
    
    attn_diag = extract_attention_diagonal(attention)
    out_degrees = torch.zeros(original_shape + (seq_len,), device=device, dtype=dtype)
    
    for i in range(seq_len):
        if i < seq_len - 1:
            col_sum = attention[..., i+1:, i].sum(dim=-1)
            denominator = seq_len - i
            out_degrees[..., i] = col_sum / denominator
    
    return out_degrees - attn_diag


def compute_attention_entropy(attention: Tensor, eps: float = 1e-10) -> Tensor:
    """Compute row-wise entropy of attention distribution.
    
    Args:
        attention: [batch, heads, seq, seq]
        eps: Small value for numerical stability
        
    Returns:
        Entropy [batch, heads, seq]
    """
    attention = attention.clamp(min=eps)
    return -torch.sum(attention * torch.log(attention), dim=-1)


def stack_layer_attentions(
    attentions: Tuple[Tensor, ...],
    layers: Optional[List[int]] = None,
) -> Tensor:
    """Stack attention matrices from specified layers.
    
    Args:
        attentions: Tuple of attention tensors from model output,
                   each tensor is [batch, heads, seq, seq]
        layers: List of layer indices to stack. If None, stack all layers.
        
    Returns:
        Stacked attention tensor [n_layers, batch, heads, seq, seq]
        or [n_layers, heads, seq, seq] if batch=1 and squeezed
    """
    if layers is None:
        layers = list(range(len(attentions)))
    
    selected = []
    for layer_idx in layers:
        if 0 <= layer_idx < len(attentions):
            selected.append(attentions[layer_idx])
    
    if len(selected) == 0:
        raise ValueError(f"No valid layers selected from {len(attentions)} available layers")
    
    return torch.stack(selected, dim=0)


# ==============================================================================
# Hidden State Operations
# ==============================================================================

def pool_hidden_states(hidden_states: Tensor, method: str = "last_token") -> Tensor:
    """Pool hidden states to single vector or keep full sequence.
    
    Args:
        hidden_states: [batch, seq, hidden]
        method: 'last_token', 'first_token', 'mean', 'max', or 'none'
        
    Returns:
        Pooled states [batch, hidden] or full states [batch, seq, hidden] if method='none'
    """
    if method == "none" or method is None:
        # 不池化，保留完整序列
        return hidden_states
    elif method == "last_token":
        return hidden_states[:, -1, :]
    elif method == "first_token":
        return hidden_states[:, 0, :]
    elif method == "mean":
        return hidden_states.mean(dim=1)
    elif method == "max":
        return hidden_states.max(dim=1)[0]
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def stack_layer_hidden_states(
    hidden_states: Tuple[Tensor, ...],
    layers: Optional[List[int]] = None,
) -> Tensor:
    """Stack hidden states from specified layers.
    
    Args:
        hidden_states: Tuple of hidden state tensors from model output,
                      each tensor is [batch, seq, hidden]
        layers: List of layer indices to stack. If None, stack all layers.
        
    Returns:
        Stacked hidden states tensor [n_layers, batch, seq, hidden]
        or [n_layers, seq, hidden] if batch=1 and squeezed
    """
    if layers is None:
        layers = list(range(len(hidden_states)))
    
    selected = []
    for layer_idx in layers:
        if 0 <= layer_idx < len(hidden_states):
            selected.append(hidden_states[layer_idx])
    
    if len(selected) == 0:
        raise ValueError(f"No valid layers selected from {len(hidden_states)} available layers")
    
    return torch.stack(selected, dim=0)


# ==============================================================================
# Token Probability Operations
# ==============================================================================

def compute_token_probs(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Compute probability of target tokens.
    
    Args:
        logits: [batch, seq, vocab]
        target_ids: [batch, seq]
        
    Returns:
        Token probabilities [batch, seq-1]
    """
    shift_logits = logits[:, :-1, :]
    shift_targets = target_ids[:, 1:]
    probs = F.softmax(shift_logits, dim=-1)
    target_probs = torch.gather(probs, dim=-1, index=shift_targets.unsqueeze(-1))
    return target_probs.squeeze(-1)


def compute_token_entropy(logits: Tensor) -> Tensor:
    """Compute entropy of probability distribution at each position.
    
    Args:
        logits: [batch, seq, vocab]
        
    Returns:
        Entropy [batch, seq]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def compute_top_k_probs(
    logits: Tensor,
    k: int = 10,
) -> Tuple[Tensor, Tensor]:
    """Compute top-k token probabilities and their indices.
    
    Args:
        logits: [batch, seq, vocab]
        k: Number of top tokens to return
        
    Returns:
        Tuple of:
        - top_k_probs: [batch, seq, k] - probabilities of top-k tokens
        - top_k_indices: [batch, seq, k] - indices of top-k tokens
    """
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=k, dim=-1)
    return top_k_probs, top_k_indices


def compute_perplexity(token_probs: Tensor, eps: float = 1e-10) -> Tensor:
    """Compute perplexity from token probabilities.
    
    Args:
        token_probs: Token probabilities [batch, seq] or [seq]
        eps: Small value for numerical stability
        
    Returns:
        Perplexity value (scalar tensor)
    """
    log_probs = torch.log(token_probs.clamp(min=eps))
    avg_neg_log_prob = -log_probs.mean()
    return torch.exp(avg_neg_log_prob)


def normalize_attention(attention: Tensor, dim: int = -1) -> Tensor:
    """Normalize attention weights to sum to 1.
    
    Args:
        attention: Attention tensor
        dim: Dimension to normalize over
        
    Returns:
        Normalized attention tensor
    """
    return attention / (attention.sum(dim=dim, keepdim=True) + 1e-10)
