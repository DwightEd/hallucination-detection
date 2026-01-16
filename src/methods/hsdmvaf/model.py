"""HSDMVAF Model Components.

包含 Transformer Encoder 和 CRF 模型定义。
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewAttentionEncoder(nn.Module):
    """多视角注意力特征编码器。
    
    使用 Transformer Encoder 对 MVA 特征进行编码。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        return x


class CRFLayer(nn.Module):
    """条件随机场层。"""
    
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        
        return (forward_score - gold_score).mean()
    
    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        for i in range(1, seq_len):
            score += self.transitions[tags[:, i-1], tags[:, i]] * mask[:, i].float()
            score += emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1) * mask[:, i].float()
        
        last_tag_indices = mask.sum(1).long() - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i:i+1].bool(), next_score, score)
        
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)
    
    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i:i+1].bool(), next_score, score)
            history.append(indices)
        
        score += self.end_transitions
        best_tags_list = []
        _, best_last_tag = score.max(dim=1)
        
        for b in range(batch_size):
            best_tags = [best_last_tag[b].item()]
            seq_length = int(mask[b].sum().item())
            
            for hist in reversed(history[:seq_length-1]):
                best_tags.append(hist[b, best_tags[-1]].item())
            
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list


class HSDMVAFModel(nn.Module):
    """HSDMVAF 模型：Transformer Encoder + CRF。"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.input_dim = config.get('input_dim', 128)
        
        enc_config = config.get('encoder', {})
        hidden_dim = enc_config.get('hidden_dim', 256)
        
        self.encoder = MultiViewAttentionEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_heads=enc_config.get('num_heads', 8),
            num_layers=enc_config.get('num_layers', 4),
            feedforward_dim=enc_config.get('feedforward_dim', 512),
            dropout=enc_config.get('dropout', 0.1)
        )
        
        self.use_crf = config.get('use_crf', True)
        self.classifier = nn.Linear(hidden_dim, 2)
        
        if self.use_crf:
            self.crf = CRFLayer(num_tags=2)
    
    def forward(self, features, labels=None, mask=None):
        hidden = self.encoder(features, mask)
        logits = self.classifier(hidden)
        
        output = {'logits': logits}
        
        if labels is not None:
            if self.use_crf:
                if mask is None:
                    mask = labels != -100
                valid_labels = labels.clone()
                valid_labels[labels == -100] = 0
                output['loss'] = self.crf(logits, valid_labels, mask)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                output['loss'] = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        if self.use_crf:
            output['predictions'] = self.crf.decode(logits, mask)
        else:
            output['predictions'] = logits.argmax(dim=-1)
        
        return output
    
    def predict_proba(self, features, mask=None):
        with torch.no_grad():
            hidden = self.encoder(features, mask)
            logits = self.classifier(hidden)
            probs = F.softmax(logits, dim=-1)
        return probs[:, :, 1]
