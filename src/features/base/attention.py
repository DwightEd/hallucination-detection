"""Attention Feature Extractor - 完整注意力矩阵提取。

提取模型的完整注意力矩阵，支持：
- 选择特定层
- 内存优化选项
- 分块处理长序列

Usage:
    from src.features.base.attention import AttentionExtractor
    
    extractor = AttentionExtractor(
        layers=[28, 29, 30, 31],  # 最后 4 层
        store_on_cpu=True,
    )
    
    attention = extractor.extract(model_outputs)
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttentionExtractionConfig:
    """注意力提取配置。"""
    layers: Optional[List[int]] = None  # None = 所有层
    heads: Optional[List[int]] = None   # None = 所有头
    store_on_cpu: bool = True           # 是否存储到 CPU
    half_precision: bool = True         # 是否使用 float16
    max_seq_len: Optional[int] = None   # 截断长度


class AttentionExtractor:
    """注意力矩阵提取器。
    
    支持：
    - 层选择
    - 头选择
    - 内存优化
    """
    
    def __init__(self, config: Optional[AttentionExtractionConfig] = None):
        self.config = config or AttentionExtractionConfig()
    
    def extract(
        self,
        attentions: tuple,
        prompt_len: int = 0,
        response_len: int = 0,
    ) -> Dict[str, Any]:
        """从模型输出中提取注意力矩阵。
        
        Args:
            attentions: 模型输出的 attentions tuple
            prompt_len: Prompt 长度（用于标记）
            response_len: Response 长度（用于标记）
            
        Returns:
            {
                "full_attention": Tensor [n_layers, n_heads, seq_len, seq_len],
                "layers_extracted": List[int],
                "prompt_len": int,
                "response_len": int,
            }
        """
        if attentions is None or len(attentions) == 0:
            raise ValueError("No attention outputs available")
        
        n_total_layers = len(attentions)
        
        # 确定要提取的层
        if self.config.layers is not None:
            layers_to_extract = [
                l if l >= 0 else n_total_layers + l
                for l in self.config.layers
            ]
            layers_to_extract = [l for l in layers_to_extract if 0 <= l < n_total_layers]
        else:
            layers_to_extract = list(range(n_total_layers))
        
        attention_list = []
        
        for layer_idx in layers_to_extract:
            attn = attentions[layer_idx]  # [batch, n_heads, seq_len, seq_len]
            
            # 移除 batch 维度
            if attn.dim() == 4:
                attn = attn.squeeze(0)  # [n_heads, seq_len, seq_len]
            
            # 选择特定头
            if self.config.heads is not None:
                attn = attn[self.config.heads]
            
            # 截断序列长度
            if self.config.max_seq_len is not None:
                max_len = self.config.max_seq_len
                attn = attn[:, :max_len, :max_len]
            
            # 精度转换
            if self.config.half_precision:
                attn = attn.half()
            
            # 移到 CPU
            if self.config.store_on_cpu:
                attn = attn.cpu()
            
            attention_list.append(attn)
        
        # 堆叠所有层
        full_attention = torch.stack(attention_list, dim=0)
        
        return {
            "full_attention": full_attention,
            "layers_extracted": layers_to_extract,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "n_heads": full_attention.shape[1],
            "seq_len": full_attention.shape[2],
        }
    
    def extract_and_clear(
        self,
        attentions: tuple,
        prompt_len: int = 0,
        response_len: int = 0,
    ) -> Dict[str, Any]:
        """提取注意力并清理 GPU 内存。"""
        result = self.extract(attentions, prompt_len, response_len)
        
        # 清理原始 attentions
        del attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result


def extract_full_attention(
    attentions: tuple,
    layers: Optional[List[int]] = None,
    store_on_cpu: bool = True,
) -> torch.Tensor:
    """提取完整注意力矩阵的便捷函数。
    
    Args:
        attentions: 模型输出的 attentions
        layers: 要提取的层（None = 所有层）
        store_on_cpu: 是否存储到 CPU
        
    Returns:
        Tensor [n_layers, n_heads, seq_len, seq_len]
    """
    config = AttentionExtractionConfig(
        layers=layers,
        store_on_cpu=store_on_cpu,
    )
    extractor = AttentionExtractor(config)
    result = extractor.extract(attentions)
    return result["full_attention"]


def extract_attention_for_layers(
    attentions: tuple,
    layers: List[int],
    prompt_len: int,
    response_len: int,
    response_only: bool = True,
) -> torch.Tensor:
    """提取指定层的注意力，可选只提取 response 部分。
    
    Args:
        attentions: 模型输出的 attentions
        layers: 要提取的层
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只提取 response 部分
        
    Returns:
        Tensor [n_layers, n_heads, response_len, seq_len] 或
        Tensor [n_layers, n_heads, seq_len, seq_len]
    """
    attention_list = []
    n_total_layers = len(attentions)
    
    for layer_idx in layers:
        if layer_idx < 0:
            layer_idx = n_total_layers + layer_idx
        
        if 0 <= layer_idx < n_total_layers:
            attn = attentions[layer_idx]
            
            # 移除 batch 维度
            if attn.dim() == 4:
                attn = attn.squeeze(0)
            
            # 只提取 response 部分的 attention
            if response_only and prompt_len > 0:
                resp_end = prompt_len + response_len
                attn = attn[:, prompt_len:resp_end, :]  # [n_heads, resp_len, seq_len]
            
            attention_list.append(attn.cpu())
    
    return torch.stack(attention_list, dim=0)


def compute_memory_for_attention(
    n_layers: int,
    n_heads: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """计算存储完整注意力矩阵的内存需求。
    
    Returns:
        {"bytes": int, "mb": float, "gb": float}
    """
    bytes_per_element = 2 if dtype == torch.float16 else 4
    total_bytes = n_layers * n_heads * seq_len * seq_len * bytes_per_element
    
    return {
        "bytes": total_bytes,
        "mb": total_bytes / (1024 ** 2),
        "gb": total_bytes / (1024 ** 3),
    }
