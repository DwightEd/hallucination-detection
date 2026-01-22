"""HSDMVAF Model Components.

严格按照原论文实现: "Hallucinated Span Detection with Multi-View Attention Features"
GitHub: https://github.com/Ogamon958/mva_hal_det

模型架构:
    Input Features (3*L*H) → Standardization → Linear → Positional Encoding 
    → Transformer Encoder → CRF → Binary Labels (0/1)

关键组件:
- PositionalEncoding: 位置编码
- CRFLayer: 条件随机场层 (用于序列标注)
- HSDMVAFModel: 完整的Transformer+CRF模型
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple, Union
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 位置编码
# =============================================================================

class PositionalEncoding(nn.Module):
    """Transformer位置编码。
    
    使用正弦/余弦位置编码。
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional_encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# CRF层
# =============================================================================

class CRFLayer(nn.Module):
    """条件随机场层 (Conditional Random Field)。
    
    用于序列标注任务，建模相邻标签之间的依赖关系。
    """
    
    def __init__(self, num_tags: int):
        """
        Args:
            num_tags: 标签数量 (2 for binary: not hallucinated, hallucinated)
        """
        super().__init__()
        self.num_tags = num_tags
        
        # 转移矩阵: transitions[i, j] = 从tag i转移到tag j的分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 起始和结束转移分数
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重。"""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """计算负对数似然损失。
        
        Args:
            emissions: 发射分数 [batch, seq_len, num_tags]
            tags: 真实标签 [batch, seq_len]
            mask: 掩码 [batch, seq_len], 1=有效, 0=padding
            reduction: 'mean', 'sum', 或 'none'
            
        Returns:
            负对数似然损失
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        else:
            mask = mask.bool()
        
        # 计算分子 (gold score)
        gold_score = self._score_sentence(emissions, tags, mask)
        
        # 计算分母 (partition function)
        forward_score = self._forward_algorithm(emissions, mask)
        
        # 负对数似然
        nll = forward_score - gold_score
        
        if reduction == 'mean':
            return nll.mean()
        elif reduction == 'sum':
            return nll.sum()
        else:
            return nll
    
    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算给定标签序列的分数。
        
        Args:
            emissions: [batch, seq_len, num_tags]
            tags: [batch, seq_len]
            mask: [batch, seq_len]
            
        Returns:
            分数 [batch]
        """
        batch_size, seq_len, _ = emissions.shape
        
        # 起始分数
        score = self.start_transitions[tags[:, 0]]
        
        # 第一个发射分数
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        # 遍历序列
        for i in range(1, seq_len):
            # 转移分数
            transition_score = self.transitions[tags[:, i-1], tags[:, i]]
            # 发射分数
            emission_score = emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1)
            # 只在mask为True时累加
            score += (transition_score + emission_score) * mask[:, i].float()
        
        # 结束分数 (需要找到每个序列的最后一个有效位置)
        seq_lengths = mask.sum(dim=1).long()
        last_tags = tags.gather(1, (seq_lengths - 1).unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """前向算法计算partition function。
        
        Args:
            emissions: [batch, seq_len, num_tags]
            mask: [batch, seq_len]
            
        Returns:
            log partition function [batch]
        """
        batch_size, seq_len, _ = emissions.shape
        
        # 初始化: alpha[0] = start_transitions + emissions[0]
        alpha = self.start_transitions + emissions[:, 0]  # [batch, num_tags]
        
        for i in range(1, seq_len):
            # alpha_t[j] = log Σ_i exp(alpha_{t-1}[i] + transitions[i,j] + emissions_t[j])
            # 使用log-sum-exp技巧
            emit_scores = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            trans_scores = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            alpha_expand = alpha.unsqueeze(2)  # [batch, num_tags, 1]
            
            scores = alpha_expand + trans_scores + emit_scores  # [batch, num_tags, num_tags]
            new_alpha = torch.logsumexp(scores, dim=1)  # [batch, num_tags]
            
            # 只在mask为True时更新
            alpha = torch.where(mask[:, i:i+1].bool(), new_alpha, alpha)
        
        # 加上结束转移分数
        alpha = alpha + self.end_transitions
        
        return torch.logsumexp(alpha, dim=1)
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Viterbi解码找到最优标签序列。
        
        Args:
            emissions: [batch, seq_len, num_tags]
            mask: [batch, seq_len]
            
        Returns:
            最优标签序列列表
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()
        
        batch_size, seq_len, _ = emissions.shape
        
        # 初始化
        score = self.start_transitions + emissions[:, 0]  # [batch, num_tags]
        history = []
        
        # Viterbi前向
        for i in range(1, seq_len):
            # score[j] = max_i (score_{t-1}[i] + transitions[i,j]) + emissions_t[j]
            broadcast_score = score.unsqueeze(2)  # [batch, num_tags, 1]
            broadcast_trans = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            
            scores = broadcast_score + broadcast_trans  # [batch, num_tags, num_tags]
            max_scores, indices = scores.max(dim=1)  # [batch, num_tags]
            
            new_score = max_scores + emissions[:, i]
            score = torch.where(mask[:, i:i+1], new_score, score)
            history.append(indices)
        
        # 加上结束转移
        score = score + self.end_transitions
        
        # 回溯
        best_tags_list = []
        _, best_last_tag = score.max(dim=1)  # [batch]
        
        for b in range(batch_size):
            best_tags = [best_last_tag[b].item()]
            seq_length = int(mask[b].sum().item())
            
            # 回溯有效序列部分
            for hist in reversed(history[:seq_length-1]):
                best_tags.append(hist[b, best_tags[-1]].item())
            
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list


# =============================================================================
# HSDMVAF模型
# =============================================================================

class HSDMVAFModel(nn.Module):
    """HSDMVAF模型: Transformer Encoder + CRF。
    
    论文默认配置:
    - d_model: 256
    - nhead: 8
    - num_encoder_layers: 4
    - dim_feedforward: 1024
    - dropout: 0.1
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，包含:
                - input_dim: 输入特征维度 (3 * num_layers * num_heads)
                - encoder: Transformer配置
                    - hidden_dim: 隐藏层维度 (d_model)
                    - num_heads: 注意力头数
                    - num_layers: Transformer层数
                    - feedforward_dim: FFN维度
                    - dropout: dropout率
                - use_crf: 是否使用CRF层
        """
        super().__init__()
        self.config = config
        
        self.input_dim = config.get('input_dim', 128)
        
        enc_config = config.get('encoder', {})
        hidden_dim = enc_config.get('hidden_dim', 256)
        num_heads = enc_config.get('num_heads', 8)
        num_layers = enc_config.get('num_layers', 4)
        feedforward_dim = enc_config.get('feedforward_dim', 1024)
        dropout = enc_config.get('dropout', 0.1)
        
        self.hidden_dim = hidden_dim
        
        # 特征标准化参数 (训练时学习)
        self.register_buffer('feature_mean', torch.zeros(self.input_dim))
        self.register_buffer('feature_std', torch.ones(self.input_dim))
        
        # 输入投影层
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Layer Norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 输出分类层
        self.classifier = nn.Linear(hidden_dim, 2)
        
        # CRF层
        self.use_crf = config.get('use_crf', True)
        if self.use_crf:
            self.crf = CRFLayer(num_tags=2)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def set_standardization_params(self, mean: torch.Tensor, std: torch.Tensor):
        """设置特征标准化参数 (从训练数据计算)。
        
        Args:
            mean: 特征均值 [input_dim]
            std: 特征标准差 [input_dim]
        """
        self.feature_mean = mean.to(self.feature_mean.device)
        self.feature_std = std.to(self.feature_std.device)
    
    def standardize_features(self, features: torch.Tensor) -> torch.Tensor:
        """标准化特征。
        
        Args:
            features: [batch, seq_len, input_dim]
            
        Returns:
            标准化后的特征
        """
        return (features - self.feature_mean) / (self.feature_std + 1e-8)
    
    def get_emissions(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取发射分数。
        
        Args:
            features: [batch, seq_len, input_dim]
            mask: [batch, seq_len], 1=有效, 0=padding
            
        Returns:
            emissions: [batch, seq_len, num_tags]
        """
        # 标准化
        features = self.standardize_features(features)
        
        # 投影到隐藏维度
        x = self.input_proj(features)  # [batch, seq_len, hidden_dim]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        # 分类
        emissions = self.classifier(x)  # [batch, seq_len, 2]
        
        return emissions
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。
        
        Args:
            features: [batch, seq_len, input_dim]
            labels: [batch, seq_len], 真实标签 (训练时)
            mask: [batch, seq_len], 1=有效, 0=padding
            
        Returns:
            字典包含:
                - loss: 负对数似然损失 (如果提供了labels)
                - emissions: 发射分数
                - predictions: 预测标签 (推理时)
        """
        emissions = self.get_emissions(features, mask)
        
        result = {'emissions': emissions}
        
        if labels is not None:
            # 训练模式: 计算损失
            if self.use_crf:
                loss = self.crf(emissions, labels, mask, reduction='mean')
            else:
                # 不使用CRF时，使用交叉熵
                if mask is not None:
                    active_loss = mask.view(-1) == 1
                    active_logits = emissions.view(-1, 2)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = F.cross_entropy(active_logits, active_labels)
                else:
                    loss = F.cross_entropy(emissions.view(-1, 2), labels.view(-1))
            result['loss'] = loss
        else:
            # 推理模式: 解码
            if self.use_crf:
                predictions = self.crf.decode(emissions, mask)
            else:
                predictions = emissions.argmax(dim=-1).tolist()
            result['predictions'] = predictions
        
        return result
    
    def predict(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """预测标签序列。
        
        Args:
            features: [batch, seq_len, input_dim]
            mask: [batch, seq_len]
            
        Returns:
            预测的标签序列列表
        """
        self.eval()
        with torch.no_grad():
            emissions = self.get_emissions(features, mask)
            
            if self.use_crf:
                predictions = self.crf.decode(emissions, mask)
            else:
                predictions = emissions.argmax(dim=-1).tolist()
        
        return predictions
    
    def predict_proba(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """预测概率。
        
        Args:
            features: [batch, seq_len, input_dim]
            mask: [batch, seq_len]
            
        Returns:
            概率分布 [batch, seq_len, 2]
        """
        self.eval()
        with torch.no_grad():
            emissions = self.get_emissions(features, mask)
            probs = F.softmax(emissions, dim=-1)
        
        return probs
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取模型状态 (用于保存)。"""
        return {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'feature_mean': self.feature_mean.cpu(),
            'feature_std': self.feature_std.cpu(),
        }
    
    def load_state_dict_from_saved(self, state: Dict[str, Any]):
        """从保存的状态加载模型。"""
        self.load_state_dict(state['model_state_dict'])
        if 'feature_mean' in state:
            self.feature_mean = state['feature_mean'].to(self.feature_mean.device)
        if 'feature_std' in state:
            self.feature_std = state['feature_std'].to(self.feature_std.device)