"""Multi-View Attention Feature Extraction.

严格按照论文实现: "Hallucinated Span Detection with Multi-View Attention Features"
GitHub: https://github.com/Ogamon958/mva_hal_det

从注意力矩阵中提取三种互补特征：
1. avg_in (μ): 平均入向注意力 - token收到的平均注意力权重
2. div_in (β): 入向注意力多样性 - 归一化熵
3. div_out (γ): 出向注意力多样性 - 归一化熵

=============================================================================
关键修正 (按原论文公式):
=============================================================================
1. avg_in: 
   - 原公式: α'[i,j] = α[i,j] * i (位置调整)
   - μ[j] = (1 / (T - j + 1)) * Σ_{i=j}^{T} α'[i,j]
   
2. div_in:
   - κ[i,j] = α'[i,j] / Σ_k α'[i,k] (归一化)
   - β[j] = (-Σ_{i=j}^{T} κ[i,j] * log(κ[i,j])) / log(T - j + 1)
   
3. div_out:
   - γ[i] = (-Σ_{j=1}^{i} α[i,j] * log(α[i,j])) / log(i)

4. BFloat16安全转换
=============================================================================
"""
from __future__ import annotations
from typing import Optional, Union, Tuple
import numpy as np
import torch


# =============================================================================
# 辅助函数: 安全的张量转NumPy (修复BFloat16问题)
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
    """安全地将数据转换为PyTorch张量。
    
    Args:
        data: 输入数据
        dtype: 目标数据类型
        
    Returns:
        PyTorch张量
    """
    if data is None:
        return None
    
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data.astype(np.float32)).to(dtype)
    
    return torch.tensor(data, dtype=dtype)


# =============================================================================
# 核心MVA特征计算 - 严格按照原论文公式
# =============================================================================

def compute_average_incoming_attention(
    attention: torch.Tensor,
    output_start_idx: int = 0,
) -> torch.Tensor:
    """计算平均入向注意力 (avg_in / μ)。
    
    论文公式:
        α'[i,j] = α[i,j] * i  (位置调整，补偿频率不平衡)
        μ[j] = (1 / (T - j + 1)) * Σ_{i=j}^{T} α'[i,j]
    
    对于每个key token j，计算它从后续token收到的平均注意力。
    
    Args:
        attention: 注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
        output_start_idx: 输出token的起始索引
        
    Returns:
        avg_in特征 [output_len, n_layers * n_heads]
    """
    n_layers, n_heads, T, _ = attention.shape
    output_len = T - output_start_idx
    
    if output_len <= 0:
        return torch.zeros(1, n_layers * n_heads)
    
    # 创建位置调整矩阵: adjustment[i,j] = i + 1
    position_indices = torch.arange(1, T + 1, device=attention.device, dtype=attention.dtype)
    adjustment = position_indices.view(-1, 1).expand(T, T)
    
    # 调整后的注意力: α'[i,j] = α[i,j] * i
    # [n_layers, n_heads, T, T]
    A_adjusted = attention * adjustment.unsqueeze(0).unsqueeze(0)
    
    avg_in_list = []
    
    for j in range(output_start_idx, T):
        # 对于key位置j，计算从位置j到T-1的query对它的注意力之和
        # tokens attending to j: positions j to T-1
        num_attending = T - j
        
        if num_attending <= 0:
            avg_in_list.append(torch.zeros(n_layers, n_heads, device=attention.device))
            continue
        
        # 取调整后的注意力: 从query j到T-1，对key j的注意力
        # A_adjusted[:, :, j:, j] -> [n_layers, n_heads, num_attending]
        incoming_sum = A_adjusted[:, :, j:, j].sum(dim=-1)  # [n_layers, n_heads]
        
        # 平均
        avg_j = incoming_sum / max(num_attending, 1)
        avg_in_list.append(avg_j)
    
    # Stack: [output_len, n_layers, n_heads] -> [output_len, n_layers * n_heads]
    avg_in = torch.stack(avg_in_list, dim=0)  # [output_len, n_layers, n_heads]
    avg_in = avg_in.reshape(output_len, n_layers * n_heads)
    
    return avg_in


def compute_incoming_attention_entropy(
    attention: torch.Tensor,
    output_start_idx: int = 0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """计算入向注意力熵 (div_in / β)。
    
    论文公式:
        κ[i,j] = α'[i,j] / Σ_k α'[i,k]  (归一化为概率分布)
        β[j] = (-Σ_{i=j}^{T} κ[i,j] * log(κ[i,j])) / log(T - j + 1)
    
    测量token收到的注意力权重的多样性。
    
    Args:
        attention: 注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
        output_start_idx: 输出token的起始索引
        eps: 数值稳定性常数
        
    Returns:
        div_in特征 [output_len, n_layers * n_heads]
    """
    n_layers, n_heads, T, _ = attention.shape
    output_len = T - output_start_idx
    
    if output_len <= 0:
        return torch.zeros(1, n_layers * n_heads)
    
    # 位置调整
    position_indices = torch.arange(1, T + 1, device=attention.device, dtype=attention.dtype)
    adjustment = position_indices.view(-1, 1).expand(T, T)
    A_adjusted = attention * adjustment.unsqueeze(0).unsqueeze(0)
    
    # 归一化: κ[i,j] = α'[i,j] / Σ_k α'[i,k]
    # 每行归一化
    row_sums = A_adjusted.sum(dim=-1, keepdim=True) + eps  # [n_layers, n_heads, T, 1]
    A_normalized = A_adjusted / row_sums  # κ
    
    div_in_list = []
    
    for j in range(output_start_idx, T):
        num_attending = T - j
        
        if num_attending <= 0:
            div_in_list.append(torch.zeros(n_layers, n_heads, device=attention.device))
            continue
        
        # 获取归一化后的注意力: 从query j到T-1，对key j的注意力
        kappa_j = A_normalized[:, :, j:, j]  # [n_layers, n_heads, num_attending]
        
        # 计算熵: -Σ κ * log(κ)
        log_kappa = torch.log(kappa_j + eps)
        entropy_terms = -kappa_j * log_kappa
        entropy_sum = entropy_terms.sum(dim=-1)  # [n_layers, n_heads]
        
        # 归一化熵: 除以log(最大可能熵)
        max_entropy = np.log(max(num_attending, 1)) + eps
        normalized_entropy = entropy_sum / max_entropy
        
        div_in_list.append(normalized_entropy)
    
    div_in = torch.stack(div_in_list, dim=0)
    div_in = div_in.reshape(output_len, n_layers * n_heads)
    
    return div_in


def compute_outgoing_attention_entropy(
    attention: torch.Tensor,
    output_start_idx: int = 0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """计算出向注意力熵 (div_out / γ)。
    
    论文公式:
        γ[i] = (-Σ_{j=1}^{i} α[i,j] * log(α[i,j])) / log(i)
    
    测量token关注的位置的多样性。
    
    Args:
        attention: 注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
        output_start_idx: 输出token的起始索引
        eps: 数值稳定性常数
        
    Returns:
        div_out特征 [output_len, n_layers * n_heads]
    """
    n_layers, n_heads, T, _ = attention.shape
    output_len = T - output_start_idx
    
    if output_len <= 0:
        return torch.zeros(1, n_layers * n_heads)
    
    div_out_list = []
    
    for i in range(output_start_idx, T):
        # Query i关注keys 0到i (因果注意力)
        num_keys = i + 1
        
        # 获取query i的注意力行
        attention_row = attention[:, :, i, :i+1]  # [n_layers, n_heads, i+1]
        
        # 计算熵: -Σ α * log(α)
        log_attention = torch.log(attention_row + eps)
        entropy_terms = -attention_row * log_attention
        entropy_sum = entropy_terms.sum(dim=-1)  # [n_layers, n_heads]
        
        # 归一化
        max_entropy = np.log(max(num_keys, 1)) + eps
        normalized_entropy = entropy_sum / max_entropy
        
        div_out_list.append(normalized_entropy)
    
    div_out = torch.stack(div_out_list, dim=0)
    div_out = div_out.reshape(output_len, n_layers * n_heads)
    
    return div_out


def compute_multi_view_attention_features(
    attention: Union[torch.Tensor, np.ndarray],
    prompt_len: int,
    response_len: int,
    normalize: bool = True,
    max_layers: Optional[int] = None,
    max_heads: Optional[int] = None,
) -> torch.Tensor:
    """计算完整的Multi-View Attention Features。
    
    严格按照原论文实现，从注意力矩阵中提取三种互补特征:
    - avg_in (μ): 平均入向注意力
    - div_in (β): 入向注意力多样性 (归一化熵)
    - div_out (γ): 出向注意力多样性 (归一化熵)
    
    Args:
        attention: 注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt长度
        response_len: Response长度
        normalize: 是否标准化特征
        max_layers: 最大使用层数 (None=全部)
        max_heads: 最大使用头数 (None=全部)
        
    Returns:
        Multi-view features [resp_len, feature_dim]
        feature_dim = n_layers * n_heads * 3
    """
    # 安全转换
    attention = safe_to_tensor(attention)
    
    if attention is None:
        raise ValueError("Attention matrix is None")
    
    # 处理维度
    if len(attention.shape) == 3:
        attention = attention.unsqueeze(0)  # 添加layer维度
    
    n_layers, n_heads, seq_len, _ = attention.shape
    
    # 限制层和头的数量
    if max_layers is not None and max_layers < n_layers:
        # 使用最后max_layers层
        layer_start = n_layers - max_layers
        attention = attention[layer_start:]
        n_layers = max_layers
    
    if max_heads is not None and max_heads < n_heads:
        attention = attention[:, :max_heads]
        n_heads = max_heads
    
    # 确定response范围
    output_start_idx = min(prompt_len, seq_len - 1)
    output_end_idx = min(prompt_len + response_len, seq_len)
    
    if output_end_idx <= output_start_idx:
        output_start_idx = 0
        output_end_idx = seq_len
    
    eps = 1e-10
    
    # 计算三种特征
    avg_in = compute_average_incoming_attention(attention, output_start_idx)
    div_in = compute_incoming_attention_entropy(attention, output_start_idx, eps)
    div_out = compute_outgoing_attention_entropy(attention, output_start_idx, eps)
    
    # 拼接: [output_len, 3 * n_layers * n_heads]
    features = torch.cat([avg_in, div_in, div_out], dim=-1)
    
    # 截取到实际的response长度
    actual_output_len = output_end_idx - output_start_idx
    features = features[:actual_output_len]
    
    # 标准化
    if normalize and features.shape[0] > 0:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + eps
        features = (features - mean) / std
    
    # 处理无效值
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return features


def compute_mva_features_from_diags(
    attn_diags: Union[torch.Tensor, np.ndarray],
    attn_entropy: Optional[Union[torch.Tensor, np.ndarray]],
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """从对角线特征近似计算MVA特征 (降级模式)。
    
    当没有完整注意力矩阵时的备选方案。
    
    Args:
        attn_diags: [n_layers, n_heads, seq_len] 注意力对角线
        attn_entropy: [n_layers, n_heads, seq_len] 注意力熵 (可选)
        prompt_len: Prompt长度
        response_len: Response长度
        
    Returns:
        近似的MVA features [resp_len, feature_dim]
    """
    attn_diags = safe_to_tensor(attn_diags)
    
    if attn_diags is None:
        raise ValueError("attn_diags is None")
    
    n_layers, n_heads, seq_len = attn_diags.shape
    
    # 确定response范围
    resp_start = min(prompt_len, seq_len - 1)
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    actual_resp_len = resp_end - resp_start
    
    # 从对角线近似avg_in (使用对角线值作为自注意力强度的近似)
    resp_diag = attn_diags[:, :, resp_start:resp_end]  # [n_layers, n_heads, resp_len]
    
    # 转换维度: [resp_len, n_layers * n_heads]
    avg_in_approx = resp_diag.permute(2, 0, 1).reshape(actual_resp_len, -1)
    
    # div_in和div_out: 如果有熵信息则使用
    if attn_entropy is not None:
        attn_entropy = safe_to_tensor(attn_entropy)
        resp_entropy = attn_entropy[:, :, resp_start:resp_end]
        div_approx = resp_entropy.permute(2, 0, 1).reshape(actual_resp_len, -1)
        
        # 使用熵作为div_in和div_out的近似
        features = torch.cat([avg_in_approx, div_approx, div_approx], dim=1)
    else:
        # 只有对角线，复制三份
        features = torch.cat([avg_in_approx, avg_in_approx, avg_in_approx], dim=1)
    
    # 处理无效值
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return features


def compute_mva_sample_features(
    attention: Union[torch.Tensor, np.ndarray],
    prompt_len: int,
    response_len: int,
    pooling: str = "max",
    max_layers: Optional[int] = None,
    max_heads: Optional[int] = None,
) -> np.ndarray:
    """计算sample-level MVA特征。
    
    将token-level特征聚合为样本级别特征。
    
    Args:
        attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt长度
        response_len: Response长度
        pooling: 聚合方式 ("max", "mean", "both")
        max_layers: 最大使用层数
        max_heads: 最大使用头数
        
    Returns:
        Sample-level feature vector
    """
    token_features = compute_multi_view_attention_features(
        attention, prompt_len, response_len, 
        normalize=False,
        max_layers=max_layers,
        max_heads=max_heads,
    )
    
    token_features_np = safe_to_numpy(token_features)
    
    if token_features_np.shape[0] == 0:
        # 空序列，返回零向量
        feature_dim = token_features_np.shape[1] if len(token_features_np.shape) > 1 else 1
        if pooling == "both":
            return np.zeros(feature_dim * 2, dtype=np.float32)
        return np.zeros(feature_dim, dtype=np.float32)
    
    if pooling == "max":
        return token_features_np.max(axis=0).astype(np.float32)
    elif pooling == "mean":
        return token_features_np.mean(axis=0).astype(np.float32)
    elif pooling == "both":
        return np.concatenate([
            token_features_np.max(axis=0),
            token_features_np.mean(axis=0),
        ]).astype(np.float32)
    else:
        return token_features_np.max(axis=0).astype(np.float32)


# =============================================================================
# 批量特征提取 (用于训练数据准备)
# =============================================================================

def extract_mva_features_batch(
    attention_list: list,
    prompt_lens: list,
    response_lens: list,
    max_layers: Optional[int] = None,
    max_heads: Optional[int] = None,
) -> Tuple[list, list]:
    """批量提取MVA特征。
    
    Args:
        attention_list: 注意力矩阵列表
        prompt_lens: prompt长度列表
        response_lens: response长度列表
        max_layers: 最大使用层数
        max_heads: 最大使用头数
        
    Returns:
        (features_list, valid_indices)
    """
    features_list = []
    valid_indices = []
    
    for i, (attn, p_len, r_len) in enumerate(zip(attention_list, prompt_lens, response_lens)):
        try:
            feat = compute_multi_view_attention_features(
                attn, p_len, r_len,
                normalize=True,
                max_layers=max_layers,
                max_heads=max_heads,
            )
            features_list.append(feat)
            valid_indices.append(i)
        except Exception as e:
            print(f"Warning: Failed to extract features for sample {i}: {e}")
            continue
    
    return features_list, valid_indices