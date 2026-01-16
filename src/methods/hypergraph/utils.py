"""Hypergraph utility functions.

包含神经网络构建和通用工具函数。
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn


def make_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation=nn.ReLU,
    dropout: float = 0.0
) -> nn.Sequential:
    """Build MLP with LayerNorm and optional dropout.
    
    Args:
        in_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        out_dim: 输出维度
        activation: 激活函数类
        dropout: Dropout比率
        
    Returns:
        nn.Sequential MLP模块
    """
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def get_device() -> str:
    """获取可用设备。
    
    Returns:
        "cuda" 或 "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device()
