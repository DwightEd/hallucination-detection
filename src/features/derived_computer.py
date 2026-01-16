"""Derived Feature Computer - 统一的衍生特征计算器。

根据方法配置，从基础特征计算方法需要的衍生特征。

设计原则：
1. 所有计算函数在此统一注册
2. 方法在配置文件中定义需要的衍生特征和参数
3. 主代码不包含方法特定的硬编码逻辑

Usage:
    from src.features.derived_computer import DerivedFeatureComputer
    
    # 从方法配置创建计算器
    computer = DerivedFeatureComputer.from_method_config("haloscope")
    
    # 计算衍生特征
    derived_features = computer.compute(base_features, prompt_len, response_len)
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml

import torch
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 衍生特征配置
# =============================================================================

@dataclass
class DerivedFeatureConfig:
    """单个衍生特征的配置。"""
    name: str
    compute_fn: str
    inputs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 计算函数注册表
# =============================================================================

class ComputeFunctionRegistry:
    """计算函数注册表。"""
    
    _functions: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """注册计算函数的装饰器。"""
        def decorator(fn: Callable) -> Callable:
            cls._functions[name] = fn
            return fn
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """获取计算函数。"""
        if name not in cls._functions:
            raise ValueError(f"Unknown compute function: {name}")
        return cls._functions[name]
    
    @classmethod
    def list_functions(cls) -> List[str]:
        """列出所有注册的函数。"""
        return list(cls._functions.keys())


# =============================================================================
# 注册计算函数
# =============================================================================

@ComputeFunctionRegistry.register("compute_laplacian_from_diags")
def compute_laplacian_from_diags(
    attention_diags: torch.Tensor,
    attention_row_sums: Optional[torch.Tensor] = None,
    scope: str = "full",
    prompt_len: int = 0,
    response_len: int = 0,
    **kwargs
) -> torch.Tensor:
    """从对角线计算 Laplacian 对角线。
    
    L[i,i] = D[i,i] - A[i,i] = row_sum[i] - diag[i]
    对于归一化的 attention: row_sum ≈ 1
    
    Args:
        attention_diags: [n_layers, n_heads, seq_len]
        attention_row_sums: [n_layers, n_heads, seq_len]，如果为 None 则假设为 1
        scope: "full", "response_only"
        prompt_len: prompt 长度
        response_len: response 长度
        
    Returns:
        Laplacian 对角线 [n_layers, n_heads, seq_len]
    """
    if attention_row_sums is not None:
        laplacian = attention_row_sums - attention_diags
    else:
        laplacian = 1.0 - attention_diags
    
    if scope == "response_only" and prompt_len > 0:
        seq_len = laplacian.shape[-1]
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
        laplacian = laplacian[..., prompt_len:end_idx]
    
    return laplacian


@ComputeFunctionRegistry.register("compute_attention_entropy")
def compute_attention_entropy(
    full_attention: torch.Tensor,
    eps: float = 1e-10,
    scope: str = "full",
    prompt_len: int = 0,
    response_len: int = 0,
    **kwargs
) -> torch.Tensor:
    """从完整注意力矩阵计算熵。
    
    H[i] = -sum_j(A[i,j] * log(A[i,j]))
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        eps: 数值稳定性
        scope: "full", "response_only"
        prompt_len: prompt 长度
        response_len: response 长度
        
    Returns:
        注意力熵 [n_layers, n_heads, seq_len]
    """
    attn = full_attention.clamp(min=eps)
    entropy = -torch.sum(attn * torch.log(attn), dim=-1)
    
    if scope == "response_only" and prompt_len > 0:
        seq_len = entropy.shape[-1]
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
        entropy = entropy[..., prompt_len:end_idx]
    
    return entropy


@ComputeFunctionRegistry.register("compute_token_entropy")
def compute_token_entropy(
    token_probs: torch.Tensor,
    eps: float = 1e-10,
    **kwargs
) -> torch.Tensor:
    """计算 token 熵。
    
    H = -p * log(p)
    
    Args:
        token_probs: [seq_len]
        eps: 数值稳定性
        
    Returns:
        Token 熵 [seq_len]
    """
    probs = token_probs.clamp(min=eps)
    return -probs * torch.log(probs)


@ComputeFunctionRegistry.register("extract_last_token_embedding")
def extract_last_token_embedding(
    hidden_states: torch.Tensor,
    layer_selection: str = "middle",
    n_layers_to_use: Optional[int] = None,
    specific_layers: Optional[List[int]] = None,
    **kwargs
) -> torch.Tensor:
    """提取最后一个 token 的 embedding。
    
    用于 HaloScope 等方法。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        layer_selection: 
            - "all": 所有层
            - "middle": 中间层（默认，HaloScope 论文推荐）
            - "last": 最后几层
            - "first": 前几层
        n_layers_to_use: 使用的层数（用于 middle/last/first）
        specific_layers: 明确指定的层索引
        
    Returns:
        Last token embedding [n_selected_layers, hidden_dim]
    """
    return extract_token_embedding(
        hidden_states,
        token_selection="last",
        layer_selection=layer_selection,
        n_layers_to_use=n_layers_to_use,
        specific_layers=specific_layers,
        **kwargs
    )


@ComputeFunctionRegistry.register("extract_token_embedding")
def extract_token_embedding(
    hidden_states: torch.Tensor,
    token_selection: str = "last",
    token_indices: Optional[List[int]] = None,
    layer_selection: str = "all",
    n_layers_to_use: Optional[int] = None,
    specific_layers: Optional[List[int]] = None,
    prompt_len: int = 0,
    response_len: int = 0,
    scope: str = "full",
    **kwargs
) -> torch.Tensor:
    """灵活的 token embedding 提取。
    
    支持多种 token 和层的选择方式。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        token_selection: token 选择策略
            - "last": 最后一个 token
            - "first": 第一个 token
            - "mean": 平均池化
            - "max": 最大池化
            - "all": 所有 token（不池化）
            - "specific": 指定的 token 索引
            - "range": 范围内的 token
        token_indices: 当 token_selection="specific" 时，指定的 token 索引列表
        layer_selection: 层选择策略
            - "all": 所有层
            - "last_n": 最后 n 层
            - "first_n": 前 n 层
            - "middle": 中间 n 层
            - "specific": 指定的层索引
        n_layers_to_use: 用于 last_n/first_n/middle 策略
        specific_layers: 当 layer_selection="specific" 时的层索引
        prompt_len: prompt 长度（用于 scope）
        response_len: response 长度（用于 scope）
        scope: 作用域 "full" / "response_only" / "prompt_only"
        
    Returns:
        提取的 embedding，形状取决于选择策略
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    # 1. 确定层索引
    if specific_layers is not None:
        layer_indices = [l if l >= 0 else n_layers + l for l in specific_layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    elif layer_selection == "all":
        layer_indices = list(range(n_layers))
    elif layer_selection == "last_n" or layer_selection == "last":
        n_use = n_layers_to_use if n_layers_to_use else 4
        layer_indices = list(range(max(0, n_layers - n_use), n_layers))
    elif layer_selection == "first_n" or layer_selection == "first":
        n_use = n_layers_to_use if n_layers_to_use else 4
        layer_indices = list(range(min(n_use, n_layers)))
    elif layer_selection == "middle":
        n_use = n_layers_to_use if n_layers_to_use else max(1, n_layers // 2)
        start = (n_layers - n_use) // 2
        layer_indices = list(range(start, start + n_use))
    else:
        layer_indices = list(range(n_layers))
    
    # 2. 确定 token 范围
    if scope == "response_only" and prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    elif scope == "prompt_only":
        start_idx = 0
        end_idx = prompt_len if prompt_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    # 选择层
    selected = hidden_states[layer_indices]  # [n_selected_layers, seq_len, hidden_dim]
    
    # 应用范围
    selected = selected[:, start_idx:end_idx, :]
    actual_seq_len = selected.shape[1]
    
    # 3. Token 选择
    if token_selection == "last":
        return selected[:, -1, :]  # [n_layers, hidden_dim]
    
    elif token_selection == "first":
        return selected[:, 0, :]
    
    elif token_selection == "mean":
        return selected.mean(dim=1)
    
    elif token_selection == "max":
        return selected.max(dim=1)[0]
    
    elif token_selection == "all":
        return selected  # [n_layers, seq_len, hidden_dim]
    
    elif token_selection == "specific":
        if not token_indices:
            raise ValueError("token_indices required for specific selection")
        # 调整索引到当前范围
        valid_indices = [i - start_idx for i in token_indices 
                        if start_idx <= i < end_idx]
        if not valid_indices:
            raise ValueError(f"No valid token indices in range [{start_idx}, {end_idx})")
        return selected[:, valid_indices, :]  # [n_layers, n_tokens, hidden_dim]
    
    elif token_selection == "range":
        # token_indices 格式: [start, end]
        if not token_indices or len(token_indices) < 2:
            raise ValueError("token_indices [start, end] required for range selection")
        t_start = max(0, token_indices[0] - start_idx)
        t_end = min(actual_seq_len, token_indices[1] - start_idx)
        return selected[:, t_start:t_end, :]
    
    else:
        raise ValueError(f"Unknown token_selection: {token_selection}")


@ComputeFunctionRegistry.register("extract_pooled_embedding")
def extract_pooled_embedding(
    hidden_states: torch.Tensor,
    layer_selection: str = "last",
    n_layers_to_use: int = 4,
    pooling: str = "mean",
    scope: str = "response_only",
    prompt_len: int = 0,
    response_len: int = 0,
    **kwargs
) -> torch.Tensor:
    """提取池化后的 embedding。
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        layer_selection: 层选择策略
        n_layers_to_use: 使用的层数
        pooling: "mean", "max", "last", "first"
        scope: "full", "response_only", "prompt_only"
        prompt_len: prompt 长度
        response_len: response 长度
        
    Returns:
        Pooled embedding [n_selected_layers, hidden_dim]
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    # 确定层
    if layer_selection == "all":
        layer_indices = list(range(n_layers))
    elif layer_selection == "last":
        layer_indices = list(range(max(0, n_layers - n_layers_to_use), n_layers))
    elif layer_selection == "first":
        layer_indices = list(range(min(n_layers_to_use, n_layers)))
    elif layer_selection == "middle":
        start = (n_layers - n_layers_to_use) // 2
        layer_indices = list(range(start, start + n_layers_to_use))
    else:
        layer_indices = list(range(n_layers))
    
    selected = hidden_states[layer_indices]  # [n_selected, seq_len, hidden_dim]
    
    # 确定范围
    if scope == "response_only" and prompt_len > 0:
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
        selected = selected[:, prompt_len:end_idx, :]
    elif scope == "prompt_only":
        selected = selected[:, :prompt_len, :]
    
    # 池化
    if pooling == "mean":
        pooled = selected.mean(dim=1)
    elif pooling == "max":
        pooled = selected.max(dim=1)[0]
    elif pooling == "last":
        pooled = selected[:, -1, :]
    elif pooling == "first":
        pooled = selected[:, 0, :]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    
    return pooled


@ComputeFunctionRegistry.register("compute_mva_features")
def compute_mva_features(
    full_attention: torch.Tensor,
    layers: Optional[List[int]] = None,
    head_aggregation: str = "mean",
    threshold: float = 0.01,
    prompt_len: int = 0,
    response_len: int = 0,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """计算 Multi-View Attention 特征。
    
    用于 HSDMVAF 方法。
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        layers: 要使用的层（None = 最后4层）
        head_aggregation: 头聚合方式
        threshold: diversity 阈值
        prompt_len: prompt 长度
        response_len: response 长度
        
    Returns:
        {"avg_in": Tensor, "div_in": Tensor, "div_out": Tensor}
    """
    n_layers, n_heads, seq_len, _ = full_attention.shape
    eps = 1e-10
    
    if layers is None:
        layer_indices = list(range(max(0, n_layers - 4), n_layers))
    else:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    
    if prompt_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len) if response_len > 0 else seq_len
    else:
        start_idx = 0
        end_idx = seq_len
    
    range_len = end_idx - start_idx
    selected_attn = full_attention[layer_indices]
    
    avg_in = torch.zeros(len(layer_indices), n_heads, range_len)
    div_in = torch.zeros(len(layer_indices), n_heads, range_len)
    div_out = torch.zeros(len(layer_indices), n_heads, range_len)
    
    for i, pos in enumerate(range(start_idx, end_idx)):
        incoming_attn = selected_attn[:, :, :pos+1, pos]
        avg_in[:, :, i] = incoming_attn.mean(dim=-1)
        significant_in = (incoming_attn > threshold).float().sum(dim=-1)
        div_in[:, :, i] = significant_in / (pos + 1 + eps)
        
        outgoing_attn = selected_attn[:, :, pos, :pos+1]
        significant_out = (outgoing_attn > threshold).float().sum(dim=-1)
        div_out[:, :, i] = significant_out / (pos + 1 + eps)
    
    if head_aggregation == "mean":
        avg_in = avg_in.mean(dim=1)
        div_in = div_in.mean(dim=1)
        div_out = div_out.mean(dim=1)
    
    return {
        "avg_in": avg_in,
        "div_in": div_in,
        "div_out": div_out,
    }


@ComputeFunctionRegistry.register("compute_lookback_ratio")
def compute_lookback_ratio(
    full_attention: torch.Tensor,
    prompt_len: int,
    response_len: int = 0,
    layers: Optional[List[int]] = None,
    include_self: bool = False,
    **kwargs
) -> torch.Tensor:
    """计算 Lookback 比率。
    
    Args:
        full_attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: prompt 长度
        response_len: response 长度
        layers: 要使用的层
        include_self: 是否包含自身注意力
        
    Returns:
        Lookback ratio [n_layers, n_heads, response_len]
    """
    n_layers, n_heads, seq_len, _ = full_attention.shape
    eps = 1e-10
    
    if layers is None:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [l if l >= 0 else n_layers + l for l in layers]
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers]
    
    if response_len <= 0:
        response_len = seq_len - prompt_len
    
    resp_end = min(prompt_len + response_len, seq_len)
    actual_resp_len = resp_end - prompt_len
    
    lookback_ratio = torch.zeros(len(layer_indices), n_heads, actual_resp_len)
    
    for li, layer_idx in enumerate(layer_indices):
        attn = full_attention[layer_idx]
        
        for i, pos in enumerate(range(prompt_len, resp_end)):
            attn_row = attn[:, pos, :]
            attn_to_prompt = attn_row[:, :prompt_len].sum(dim=-1)
            
            if include_self:
                attn_to_response = attn_row[:, prompt_len:pos+1].sum(dim=-1)
            else:
                attn_to_response = attn_row[:, prompt_len:pos].sum(dim=-1)
            
            total_attn = attn_to_prompt + attn_to_response + eps
            lookback_ratio[li, :, i] = attn_to_prompt / total_attn
    
    return lookback_ratio


# =============================================================================
# 衍生特征计算器
# =============================================================================

class DerivedFeatureComputer:
    """统一的衍生特征计算器。
    
    根据方法配置中的 derived_features 定义，
    从基础特征计算方法需要的衍生特征。
    """
    
    def __init__(self, configs: Optional[List[DerivedFeatureConfig]] = None):
        """初始化计算器。
        
        Args:
            configs: 衍生特征配置列表
        """
        self.configs = configs or []
    
    @classmethod
    def from_method_config(
        cls, 
        method_name: str, 
        config_dir: Optional[Path] = None
    ) -> "DerivedFeatureComputer":
        """从方法配置文件创建计算器。
        
        Args:
            method_name: 方法名称
            config_dir: 配置目录（默认 config/method/）
            
        Returns:
            DerivedFeatureComputer 实例
        """
        if config_dir is None:
            config_dir = Path("config/method")
        
        config_file = config_dir / f"{method_name}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Method config not found: {config_file}")
            return cls([])
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        derived_features = config.get("derived_features", {})
        
        configs = []
        for name, cfg in derived_features.items():
            configs.append(DerivedFeatureConfig(
                name=name,
                compute_fn=cfg.get("compute_fn", name),
                inputs=cfg.get("inputs", []),
                params=cfg.get("params", {}),
            ))
        
        return cls(configs)
    
    @classmethod
    def from_dict(cls, derived_features: Dict[str, Any]) -> "DerivedFeatureComputer":
        """从字典创建计算器。
        
        Args:
            derived_features: 衍生特征配置字典
            
        Returns:
            DerivedFeatureComputer 实例
        """
        configs = []
        for name, cfg in derived_features.items():
            if isinstance(cfg, dict):
                configs.append(DerivedFeatureConfig(
                    name=name,
                    compute_fn=cfg.get("compute_fn", name),
                    inputs=cfg.get("inputs", []),
                    params=cfg.get("params", {}),
                ))
        return cls(configs)
    
    def compute(
        self,
        base_features: Dict[str, torch.Tensor],
        prompt_len: int = 0,
        response_len: int = 0,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """计算所有配置的衍生特征。
        
        Args:
            base_features: 基础特征字典
            prompt_len: prompt 长度
            response_len: response 长度
            
        Returns:
            衍生特征字典
        """
        derived = {}
        
        for config in self.configs:
            try:
                fn = ComputeFunctionRegistry.get(config.compute_fn)
                
                # 收集输入
                inputs = []
                for input_name in config.inputs:
                    if input_name not in base_features:
                        logger.warning(
                            f"Missing input '{input_name}' for derived feature '{config.name}'"
                        )
                        continue
                    inputs.append(base_features[input_name])
                
                if len(inputs) != len(config.inputs):
                    continue
                
                # 添加通用参数
                params = {
                    **config.params,
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                }
                
                # 计算
                result = fn(*inputs, **params)
                derived[config.name] = result
                
            except Exception as e:
                logger.warning(f"Failed to compute '{config.name}': {e}")
        
        return derived
    
    def compute_single(
        self,
        name: str,
        base_features: Dict[str, torch.Tensor],
        prompt_len: int = 0,
        response_len: int = 0,
        **extra_params
    ) -> Optional[torch.Tensor]:
        """计算单个衍生特征。
        
        Args:
            name: 衍生特征名称
            base_features: 基础特征字典
            prompt_len: prompt 长度
            response_len: response 长度
            extra_params: 额外参数（覆盖默认参数）
            
        Returns:
            计算结果
        """
        config = None
        for cfg in self.configs:
            if cfg.name == name:
                config = cfg
                break
        
        if config is None:
            logger.warning(f"Unknown derived feature: {name}")
            return None
        
        try:
            fn = ComputeFunctionRegistry.get(config.compute_fn)
            
            inputs = [base_features[inp] for inp in config.inputs]
            
            params = {
                **config.params,
                **extra_params,
                "prompt_len": prompt_len,
                "response_len": response_len,
            }
            
            return fn(*inputs, **params)
            
        except Exception as e:
            logger.warning(f"Failed to compute '{name}': {e}")
            return None
    
    def list_features(self) -> List[str]:
        """列出所有配置的衍生特征。"""
        return [c.name for c in self.configs]
    
    def get_required_inputs(self) -> Set[str]:
        """获取所有需要的输入特征。"""
        inputs = set()
        for config in self.configs:
            inputs.update(config.inputs)
        return inputs


# =============================================================================
# 便捷函数
# =============================================================================

def compute_derived_feature(
    name: str,
    base_features: Dict[str, torch.Tensor],
    prompt_len: int = 0,
    response_len: int = 0,
    **params
) -> Optional[torch.Tensor]:
    """直接计算衍生特征（不需要配置）。
    
    Args:
        name: 计算函数名称
        base_features: 基础特征字典
        prompt_len: prompt 长度
        response_len: response 长度
        params: 计算参数
        
    Returns:
        计算结果
    """
    try:
        fn = ComputeFunctionRegistry.get(name)
        
        # 根据函数名推断输入
        input_mapping = {
            "compute_laplacian_from_diags": ["attention_diags", "attention_row_sums"],
            "compute_attention_entropy": ["full_attention"],
            "compute_token_entropy": ["token_probs"],
            "extract_last_token_embedding": ["hidden_states"],
            "extract_pooled_embedding": ["hidden_states"],
            "compute_mva_features": ["full_attention"],
            "compute_lookback_ratio": ["full_attention"],
        }
        
        input_names = input_mapping.get(name, [])
        inputs = [base_features[inp] for inp in input_names if inp in base_features]
        
        return fn(*inputs, prompt_len=prompt_len, response_len=response_len, **params)
        
    except Exception as e:
        logger.warning(f"Failed to compute '{name}': {e}")
        return None


def list_compute_functions() -> List[str]:
    """列出所有可用的计算函数。"""
    return ComputeFunctionRegistry.list_functions()
