"""Hypergraph Data Structures and Builder.

包含超图数据容器和从注意力矩阵构建超图的逻辑。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging

import torch

if TYPE_CHECKING:
    from src.core import ExtractedFeatures

logger = logging.getLogger(__name__)


class HypergraphData:
    """Container for hypergraph data.
    
    存储超图的所有组件：节点特征、超边索引、超边属性等。
    
    Attributes:
        x: 节点特征 [num_nodes, node_dim]
        he_index: 超边连接索引 [2, num_connections]
        he_attr: 超边属性 [num_edges, hedge_dim]
        he_mark: 超边标记 [num_connections, 2]
        he_count: 每个超边的节点数 [num_edges]
        y: 节点标签 [num_nodes]
        node_pos: 节点位置索引 [num_nodes]
        response_idx: 响应起始位置
        sample_id: 样本ID
        batch: 批次索引 [num_nodes]
    """

    def __init__(
        self,
        x: torch.Tensor,
        he_index: torch.Tensor,
        he_attr: torch.Tensor,
        he_mark: torch.Tensor,
        he_count: torch.Tensor,
        y: torch.Tensor,
        node_pos: torch.Tensor,
        response_idx: int,
        sample_id: str = ""
    ):
        """初始化超图数据。
        
        Args:
            x: 节点特征
            he_index: 超边索引
            he_attr: 超边属性
            he_mark: 超边标记
            he_count: 超边节点计数
            y: 节点标签
            node_pos: 节点位置
            response_idx: 响应起始索引
            sample_id: 样本ID
        """
        self.x = x
        self.he_index = he_index
        self.he_attr = he_attr
        self.he_mark = he_mark
        self.he_count = he_count
        self.y = y
        self.node_pos = node_pos
        self.response_idx = torch.tensor([response_idx])
        self.sample_id = sample_id
        self.batch = torch.zeros(x.size(0), dtype=torch.long)

    def to(self, device: str) -> 'HypergraphData':
        """将数据移动到指定设备。
        
        Args:
            device: 目标设备 ("cuda" 或 "cpu")
            
        Returns:
            self (链式调用)
        """
        self.x = self.x.to(device)
        self.he_index = self.he_index.to(device)
        self.he_attr = self.he_attr.to(device)
        self.he_mark = self.he_mark.to(device)
        self.he_count = self.he_count.to(device)
        self.y = self.y.to(device)
        self.node_pos = self.node_pos.to(device)
        self.response_idx = self.response_idx.to(device)
        self.batch = self.batch.to(device)
        return self


class HypergraphBuilder:
    """Build hypergraph from attention matrices.
    
    从注意力矩阵构建超图结构，用于后续的GNN训练。
    
    Attributes:
        tau: 注意力阈值
        topk_per_row: 每行保留的top-k个连接
        min_members: 超边最小成员数
        include_center: 是否包含中心token
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化构建器。
        
        Args:
            config: 配置字典，包含：
                - attention_threshold: 注意力阈值 (default: 0.05)
                - topk_per_row: 每行top-k (default: 16)
                - min_members_in_he: 最小成员数 (default: 2)
                - include_center_token: 包含中心token (default: True)
        """
        self.tau = config.get("attention_threshold", 0.05)
        self.topk_per_row = config.get("topk_per_row", 16)
        self.min_members = config.get("min_members_in_he", 2)
        self.include_center = config.get("include_center_token", True)

    def build_from_attention(
        self,
        attention: torch.Tensor,
        response_idx: int,
        token_labels: Optional[List[int]] = None,
        sample_id: str = "",
    ) -> HypergraphData:
        """Build hypergraph from attention matrix.
        
        Args:
            attention: 注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
            response_idx: 响应起始位置
            token_labels: Token级别标签列表
            sample_id: 样本ID
            
        Returns:
            HypergraphData对象
        """
        attention = attention.float()
        n_layers, n_heads, seq_len, _ = attention.shape
        num_heads = n_layers * n_heads
        attention_flat = attention.reshape(num_heads, seq_len, seq_len)

        # Node features: self-attention values across all heads
        diag_idx = torch.arange(seq_len)
        self_att = attention_flat[:, diag_idx, diag_idx].transpose(0, 1)
        self_att = torch.clamp(self_att, 0.0, 1.0)

        # Normalize node features
        if self_att.numel() > 0:
            self_att = torch.clamp(self_att, -5.0, 5.0)
            mean_val = self_att.mean(dim=0, keepdim=True)
            std_val = self_att.std(dim=0, keepdim=True) + 1e-6
            self_att = (self_att - mean_val) / std_val

        x = self_att

        # Masks
        tri_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-1)
        valid_mask = tri_mask.clone()
        valid_mask[:response_idx, :response_idx] = False

        # Build hyperedges
        he_node_list = []
        he_id_list = []
        he_attr_list = []
        he_mark_list = []
        he_count_list = []
        he_counter = 0

        for head_idx in range(num_heads):
            for i in range(response_idx, seq_len):
                row = attention_flat[head_idx, i, :].clone()
                row[~valid_mask[i]] = 0.0

                valid_indices = (row > self.tau).nonzero(as_tuple=True)[0]
                if len(valid_indices) > self.topk_per_row:
                    topk_vals, topk_idx = torch.topk(row[valid_indices], self.topk_per_row)
                    valid_indices = valid_indices[topk_idx]

                members = valid_indices.tolist()
                if self.include_center and i not in members:
                    members.append(i)

                if len(members) >= self.min_members:
                    max_val = row.max().item()
                    mean_val = row[valid_indices].mean().item() if len(valid_indices) > 0 else 0.0
                    he_attr_list.append([max_val, mean_val])
                    he_count_list.append(len(members))

                    for member in members:
                        he_node_list.append(member)
                        he_id_list.append(he_counter)
                        is_center = 1.0 if member == i else 0.0
                        rel_pos = (member - response_idx) / max(1, seq_len - response_idx)
                        he_mark_list.append([is_center, rel_pos])

                    he_counter += 1

        # Handle empty hypergraph
        if he_counter == 0:
            for i in range(response_idx, seq_len):
                he_attr_list.append([0.1, 0.1])
                he_count_list.append(1)
                he_node_list.append(i)
                he_id_list.append(he_counter)
                he_mark_list.append([1.0, (i - response_idx) / max(1, seq_len - response_idx)])
                he_counter += 1

        he_index = torch.tensor([he_node_list, he_id_list], dtype=torch.long)
        he_attr = torch.tensor(he_attr_list, dtype=torch.float32)
        he_mark = torch.tensor(he_mark_list, dtype=torch.float32)
        he_count = torch.tensor(he_count_list, dtype=torch.float32)

        # Normalize hyperedge attributes
        if he_attr.numel() > 0:
            he_attr = torch.clamp(he_attr, 0.0, 1.0)
            mean_val = he_attr.mean(dim=0, keepdim=True)
            std_val = he_attr.std(dim=0, keepdim=True) + 1e-6
            he_attr = (he_attr - mean_val) / std_val

        if token_labels is not None:
            y = torch.tensor(token_labels, dtype=torch.float32)
        else:
            y = torch.zeros(seq_len, dtype=torch.float32)

        node_pos = torch.arange(seq_len)

        return HypergraphData(
            x=x, he_index=he_index, he_attr=he_attr, he_mark=he_mark,
            he_count=he_count, y=y, node_pos=node_pos,
            response_idx=response_idx, sample_id=sample_id,
        )

    def build_from_features(self, features: 'ExtractedFeatures') -> Optional[HypergraphData]:
        """Build hypergraph from ExtractedFeatures.
        
        Token-level labels 来自 extractor.py，它调用 hallucination_spans.py
        将字符级标注转换为 token 级标签，存入 features.hallucination_labels。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            HypergraphData对象，或None如果无法构建
        """
        from src.core import FeatureAccessor

        with FeatureAccessor(features, prefer_fast=False, allow_lazy_load=True) as accessor:
            attention = accessor.get_full_attention()

            if attention is None:
                return None

            seq_len = attention.shape[-1]
            response_idx = features.prompt_len

            # 直接使用 extractor 已计算好的 token-level labels
            token_labels = features.hallucination_labels

            # 检查 response_idx 是否超出范围
            if response_idx >= seq_len:
                logger.warning(
                    f"Sample {features.sample_id}: response_idx ({response_idx}) >= seq_len ({seq_len}), skipped."
                )
                return None

            result = self.build_from_attention(
                attention=attention,
                response_idx=response_idx,
                token_labels=token_labels,
                sample_id=features.sample_id,
            )

        return result
