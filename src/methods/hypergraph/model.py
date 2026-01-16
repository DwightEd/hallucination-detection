"""HyperCHARM Neural Network Models.

包含超图神经网络的层和模型定义。
"""
from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_mlp

if TYPE_CHECKING:
    from .data import HypergraphData


class HyperCharmLayer(nn.Module):
    """Hypergraph message passing layer.
    
    实现超图上的消息传递：
    1. Node -> Edge: 节点信息聚合到超边
    2. Edge -> Node: 超边信息传播回节点
    
    Attributes:
        residual: 是否使用残差连接
        node2edge: 节点到边的消息网络
        edge2node: 边到节点的消息网络
        ln_out: 输出层归一化
    """

    def __init__(
        self,
        node_dim: int,
        hedge_dim: int,
        hidden_dim: int,
        residual: bool = True
    ):
        """初始化超图层。
        
        Args:
            node_dim: 节点特征维度
            hedge_dim: 超边特征维度
            hidden_dim: 隐藏层维度
            residual: 是否使用残差连接
        """
        super().__init__()
        self.residual = residual
        self.node2edge = make_mlp(node_dim + 2, [hidden_dim], hidden_dim)
        self.edge2node = make_mlp(hedge_dim + hidden_dim, [hidden_dim], node_dim)
        self.ln_out = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,
        he_index: torch.Tensor,
        he_attr: torch.Tensor,
        he_mark: torch.Tensor,
        he_count: torch.Tensor
    ) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 节点特征 [num_nodes, node_dim]
            he_index: 超边连接索引 [2, num_connections]
            he_attr: 超边属性 [num_edges, hedge_dim]
            he_mark: 超边标记 [num_connections, 2]
            he_count: 超边节点数 [num_edges]
            
        Returns:
            更新后的节点特征 [num_nodes, node_dim]
        """
        he_ids = he_index[1]
        node_ids = he_index[0]

        # Node -> Edge aggregation
        msg_ne = self.node2edge(torch.cat([x[node_ids], he_mark[he_ids]], dim=-1))
        agg_e = torch.zeros((he_attr.size(0), msg_ne.size(-1)), device=x.device)
        agg_e.index_add_(0, he_ids, msg_ne)
        agg_e = agg_e / (he_count.unsqueeze(-1) + 1e-6)

        # Edge -> Node aggregation
        inc_msg = self.edge2node(torch.cat([he_attr[he_ids], agg_e[he_ids]], dim=-1))
        inc_msg = F.relu(inc_msg)

        out = torch.zeros_like(x)
        out.index_add_(0, node_ids, inc_msg)

        # Normalize by node degree
        num_nodes = x.size(0)
        node_deg = torch.bincount(node_ids, minlength=num_nodes).float().unsqueeze(-1).to(x.device)
        out = out / (node_deg + 1e-6)

        out = self.ln_out(out)
        return x + out if self.residual else out


class HyperCHARMModel(nn.Module):
    """Hypergraph neural network for token-level hallucination detection.
    
    使用多层超图消息传递进行token级别的幻觉检测。
    
    Attributes:
        in_proj: 输入投影层
        layers: 超图消息传递层列表
        pred: 预测头
    """

    def __init__(
        self,
        node_dim: int,
        hedge_dim: int,
        config: Dict[str, Any]
    ):
        """初始化模型。
        
        Args:
            node_dim: 节点特征维度
            hedge_dim: 超边特征维度
            config: 配置字典，包含：
                - hidden_dim: 隐藏层维度 (default: 128)
                - gnn_layers: GNN层数 (default: 2)
                - dropout: Dropout比率 (default: 0.25)
                - residual_mp: 是否使用残差连接 (default: True)
        """
        super().__init__()
        hidden_dim = config.get("hidden_dim", 128)
        n_layers = config.get("gnn_layers", 2)
        dropout = config.get("dropout", 0.25)
        residual = config.get("residual_mp", True)

        self.in_proj = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            HyperCharmLayer(
                node_dim=hidden_dim,
                hedge_dim=hedge_dim,
                hidden_dim=hidden_dim,
                residual=residual
            )
            for _ in range(n_layers)
        ])

        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.LayerNorm(max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_dim // 2), 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: 'HypergraphData') -> torch.Tensor:
        """前向传播。
        
        Args:
            data: HypergraphData对象
            
        Returns:
            预测logits [num_nodes]
        """
        h = F.relu(self.in_proj(data.x))
        for layer in self.layers:
            h = layer(h, data.he_index, data.he_attr, data.he_mark, data.he_count)
        return self.pred(h).view(-1)
