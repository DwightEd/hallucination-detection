"""Hypergraph-based Hallucination Detection Module.

使用超图神经网络（HyperCHARM）检测LLM幻觉。

模块结构：
- model.py: HyperCHARM神经网络模型
- data.py: 超图数据结构和构建器
- method.py: HypergraphMethod检测方法
- utils.py: 工具函数
"""

from .method import HypergraphMethod, HypergraphTokenMethod
from .model import HyperCHARMModel, HyperCharmLayer
from .data import HypergraphData, HypergraphBuilder

__all__ = [
    "HypergraphMethod",
    "HypergraphTokenMethod",
    "HyperCHARMModel",
    "HyperCharmLayer",
    "HypergraphData",
    "HypergraphBuilder",
]
