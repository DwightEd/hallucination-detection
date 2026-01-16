"""Hidden States Feature Extractor - 隐藏状态提取。

提取模型的隐藏状态，支持：
- 选择特定层
- 不同的池化策略
- 内存优化

Usage:
    from src.features.base.hidden_states import HiddenStatesExtractor
    
    extractor = HiddenStatesExtractor(
        layers=[-4, -3, -2, -1],  # 最后 4 层
        pooling="mean",           # 对 token 维度池化
    )
    
    hidden = extractor.extract(model_outputs)
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum, auto
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


class PoolingStrategy(Enum):
    """隐藏状态池化策略。"""
    NONE = auto()          # 不池化，保留所有 token
    MEAN = auto()          # 平均所有 token
    MAX = auto()           # 最大池化
    LAST = auto()          # 只取最后一个 token
    FIRST = auto()         # 只取第一个 token
    RESPONSE_MEAN = auto() # 只对 response token 平均


@dataclass
class HiddenStatesExtractionConfig:
    """隐藏状态提取配置。"""
    layers: Optional[List[int]] = None  # None = 所有层, 负数表示从后往前
    pooling: PoolingStrategy = PoolingStrategy.NONE
    store_on_cpu: bool = True
    half_precision: bool = True
    response_only: bool = False  # 是否只提取 response 部分


class HiddenStatesExtractor:
    """隐藏状态提取器。"""
    
    def __init__(self, config: Optional[HiddenStatesExtractionConfig] = None):
        self.config = config or HiddenStatesExtractionConfig()
    
    def extract(
        self,
        hidden_states: tuple,
        prompt_len: int = 0,
        response_len: int = 0,
    ) -> Dict[str, Any]:
        """从模型输出中提取隐藏状态。
        
        Args:
            hidden_states: 模型输出的 hidden_states tuple
            prompt_len: Prompt 长度
            response_len: Response 长度
            
        Returns:
            {
                "hidden_states": Tensor,
                "layers_extracted": List[int],
                "pooling": str,
                "shape": tuple,
            }
        """
        if hidden_states is None or len(hidden_states) == 0:
            raise ValueError("No hidden states available")
        
        # hidden_states 包含 embedding 层输出，所以有 n_layers + 1 个元素
        n_total_layers = len(hidden_states) - 1  # 不算 embedding 层
        
        # 确定要提取的层（从 1 开始，跳过 embedding 层）
        if self.config.layers is not None:
            layers_to_extract = []
            for l in self.config.layers:
                actual_idx = l if l >= 0 else n_total_layers + l + 1
                # 转换为 hidden_states 的索引（+1 因为 index 0 是 embedding）
                hs_idx = actual_idx + 1 if actual_idx >= 0 else actual_idx
                if 1 <= hs_idx <= n_total_layers:
                    layers_to_extract.append(hs_idx)
        else:
            layers_to_extract = list(range(1, n_total_layers + 1))
        
        hs_list = []
        
        for layer_idx in layers_to_extract:
            hs = hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
            
            # 移除 batch 维度
            if hs.dim() == 3:
                hs = hs.squeeze(0)  # [seq_len, hidden_dim]
            
            # 只提取 response 部分
            if self.config.response_only and prompt_len > 0:
                resp_end = prompt_len + response_len
                hs = hs[prompt_len:resp_end]
            
            # 池化
            hs = self._apply_pooling(hs, prompt_len, response_len)
            
            # 精度转换
            if self.config.half_precision:
                hs = hs.half()
            
            # 移到 CPU
            if self.config.store_on_cpu:
                hs = hs.cpu()
            
            hs_list.append(hs)
        
        # 堆叠所有层
        result_hs = torch.stack(hs_list, dim=0)
        
        return {
            "hidden_states": result_hs,
            "layers_extracted": layers_to_extract,
            "pooling": self.config.pooling.name,
            "shape": tuple(result_hs.shape),
            "prompt_len": prompt_len,
            "response_len": response_len,
        }
    
    def _apply_pooling(
        self,
        hs: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> torch.Tensor:
        """应用池化策略。
        
        Args:
            hs: [seq_len, hidden_dim] 或已经是 response 部分
            
        Returns:
            池化后的 tensor
        """
        if self.config.pooling == PoolingStrategy.NONE:
            return hs
        
        elif self.config.pooling == PoolingStrategy.MEAN:
            return hs.mean(dim=0)
        
        elif self.config.pooling == PoolingStrategy.MAX:
            return hs.max(dim=0)[0]
        
        elif self.config.pooling == PoolingStrategy.LAST:
            return hs[-1]
        
        elif self.config.pooling == PoolingStrategy.FIRST:
            return hs[0]
        
        elif self.config.pooling == PoolingStrategy.RESPONSE_MEAN:
            # 如果已经是 response only，直接平均
            if self.config.response_only:
                return hs.mean(dim=0)
            # 否则只对 response 部分平均
            resp_start = prompt_len
            resp_end = prompt_len + response_len
            if resp_end <= hs.shape[0]:
                return hs[resp_start:resp_end].mean(dim=0)
            return hs[resp_start:].mean(dim=0)
        
        return hs


def extract_hidden_states(
    hidden_states: tuple,
    layers: Optional[List[int]] = None,
    store_on_cpu: bool = True,
) -> torch.Tensor:
    """提取隐藏状态的便捷函数。
    
    Args:
        hidden_states: 模型输出的 hidden_states
        layers: 要提取的层（None = 所有层）
        store_on_cpu: 是否存储到 CPU
        
    Returns:
        Tensor [n_layers, seq_len, hidden_dim]
    """
    config = HiddenStatesExtractionConfig(
        layers=layers,
        store_on_cpu=store_on_cpu,
        pooling=PoolingStrategy.NONE,
    )
    extractor = HiddenStatesExtractor(config)
    result = extractor.extract(hidden_states)
    return result["hidden_states"]


def extract_hidden_states_for_layers(
    hidden_states: tuple,
    layers: List[int],
    prompt_len: int,
    response_len: int,
    response_only: bool = False,
    pooling: str = "none",
) -> torch.Tensor:
    """提取指定层的隐藏状态。
    
    Args:
        hidden_states: 模型输出的 hidden_states
        layers: 要提取的层（负数表示从后往前）
        prompt_len: Prompt 长度
        response_len: Response 长度
        response_only: 是否只提取 response 部分
        pooling: 池化策略
        
    Returns:
        Tensor [n_layers, seq_len/1, hidden_dim]
    """
    pooling_map = {
        "none": PoolingStrategy.NONE,
        "mean": PoolingStrategy.MEAN,
        "max": PoolingStrategy.MAX,
        "last": PoolingStrategy.LAST,
        "first": PoolingStrategy.FIRST,
        "response_mean": PoolingStrategy.RESPONSE_MEAN,
    }
    
    config = HiddenStatesExtractionConfig(
        layers=layers,
        pooling=pooling_map.get(pooling, PoolingStrategy.NONE),
        response_only=response_only,
    )
    extractor = HiddenStatesExtractor(config)
    result = extractor.extract(hidden_states, prompt_len, response_len)
    return result["hidden_states"]


def compute_memory_for_hidden_states(
    n_layers: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """计算存储隐藏状态的内存需求。
    
    Returns:
        {"bytes": int, "mb": float, "gb": float}
    """
    bytes_per_element = 2 if dtype == torch.float16 else 4
    total_bytes = n_layers * seq_len * hidden_dim * bytes_per_element
    
    return {
        "bytes": total_bytes,
        "mb": total_bytes / (1024 ** 2),
        "gb": total_bytes / (1024 ** 3),
    }
