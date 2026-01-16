"""HSDMVAF Method - Framework Integration.

将HSDMVAF检测方法集成到框架中。
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.core import ExtractedFeatures, MethodConfig, METHODS
from src.methods.base import BaseMethod

from .features import (
    compute_multi_view_attention_features,
    compute_mva_features_from_diags,
)
from .model import HSDMVAFModel

logger = logging.getLogger(__name__)


@METHODS.register("hsdmvaf", aliases=["mva", "multi_view_attention"])
class HSDMVAFMethod(BaseMethod):
    """HSDMVAF 幻觉检测方法（样本级别）。
    
    基于论文: "Hallucinated Span Detection with Multi-View Attention Features"
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.pooling = params.get("pooling", "max")
        self.use_full_attention = params.get("use_full_attention", True)
        self.max_layers = params.get("max_layers", 4)
        self.max_heads = params.get("max_heads", 8)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取 MVA 特征并聚合为样本级别。"""
        
        full_attention = features.full_attention
        if full_attention is None:
            full_attention = features.get_full_attention()
        
        if self.use_full_attention and full_attention is not None:
            mva_features = compute_multi_view_attention_features(
                attention=full_attention,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
            )
            features.release_large_features()
        elif features.attn_diags is not None:
            mva_features = compute_mva_features_from_diags(
                attn_diags=features.attn_diags,
                attn_entropy=features.attn_entropy,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
            )
        else:
            raise ValueError("HSDMVAF requires full_attention or attn_diags")
        
        if isinstance(mva_features, torch.Tensor):
            mva_features = mva_features.float().cpu().numpy()
        
        if np.any(~np.isfinite(mva_features)):
            mva_features = np.nan_to_num(mva_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if len(mva_features) == 0:
            return np.zeros(mva_features.shape[1] * 4 if len(mva_features.shape) > 1 else 4)
        
        sample_features = []
        sample_features.append(mva_features.mean(axis=0))
        sample_features.append(mva_features.max(axis=0))
        sample_features.append(mva_features.std(axis=0))
        sample_features.append(mva_features[-1])
        
        result = np.concatenate(sample_features).astype(np.float32)
        
        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result
    
    def extract_token_features(self, features: ExtractedFeatures, token_idx: int) -> Optional[np.ndarray]:
        """Extract MVA features for a single token."""
        try:
            full_attention = features.full_attention
            if full_attention is None:
                full_attention = features.get_full_attention()
            
            if self.use_full_attention and full_attention is not None:
                mva_features = compute_multi_view_attention_features(
                    attention=full_attention,
                    prompt_len=features.prompt_len,
                    response_len=features.response_len,
                )
            elif features.attn_diags is not None:
                mva_features = compute_mva_features_from_diags(
                    attn_diags=features.attn_diags,
                    attn_entropy=features.attn_entropy,
                    prompt_len=features.prompt_len,
                    response_len=features.response_len,
                )
            else:
                return None
            
            if isinstance(mva_features, torch.Tensor):
                mva_features = mva_features.float().cpu().numpy()
            
            if token_idx >= len(mva_features):
                return None
            
            result = mva_features[token_idx].astype(np.float32)
            
            if np.any(~np.isfinite(result)):
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return result
            
        except Exception as e:
            logger.debug(f"Failed to extract MVA token features: {e}")
            return None


class HSDMVAFDetector:
    """HSDMVAF 检测器，支持 token 级别的幻觉检测。"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _extract_mva_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """从数据中提取 MVA 特征。"""
        prompt_len = data.get('prompt_len', 0)
        response_len = data.get('response_len', 100)
        
        if 'full_attention' in data and data['full_attention'] is not None:
            return compute_multi_view_attention_features(
                data['full_attention'], prompt_len, response_len
            )
        elif 'attn_diags' in data and data['attn_diags'] is not None:
            return compute_mva_features_from_diags(
                data['attn_diags'],
                data.get('attn_entropy'),
                prompt_len, response_len
            )
        else:
            raise ValueError("需要 full_attention 或 attn_diags")
    
    def fit(self, train_data: List[Dict], val_data: List[Dict]):
        """训练模型。"""
        train_features = []
        train_labels = []
        
        for item in train_data:
            try:
                feat = self._extract_mva_features(item)
                train_features.append(feat)
                
                label = item.get('hallucination_labels', item.get('label', 0))
                if isinstance(label, int):
                    label = [label] * len(feat)
                train_labels.append(torch.tensor(label))
            except Exception as e:
                logger.warning(f"跳过样本: {e}")
        
        if not train_features:
            raise ValueError("没有有效的训练数据")
        
        max_len = max(f.shape[0] for f in train_features)
        padded_features = []
        padded_labels = []
        
        for feat, label in zip(train_features, train_labels):
            pad_len = max_len - feat.shape[0]
            if pad_len > 0:
                feat = F.pad(feat, (0, 0, 0, pad_len))
                label = F.pad(label, (0, pad_len), value=-100)
            padded_features.append(feat)
            padded_labels.append(label)
        
        X = torch.stack(padded_features)
        y = torch.stack(padded_labels)
        
        self.config['input_dim'] = X.shape[-1]
        self.model = HSDMVAFModel(self.config).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(self.config.get('epochs', 50)):
            self.model.train()
            X_dev = X.to(self.device)
            y_dev = y.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(X_dev, y_dev)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    def predict(self, data: Dict) -> Dict[str, Any]:
        """预测单个样本。"""
        self.model.eval()
        
        feat = self._extract_mva_features(data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.model.predict_proba(feat)
        
        probs_np = probs[0].float().cpu().numpy()
        
        pooling = self.config.get('pooling', 'max')
        if pooling == 'max':
            sample_prob = float(probs_np.max())
        elif pooling == 'mean':
            sample_prob = float(probs_np.mean())
        else:
            sample_prob = float((probs_np > 0.5).any())
        
        return {
            'label': int(sample_prob > 0.5),
            'proba': sample_prob,
            'token_probs': probs_np.tolist()
        }
    
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config
        }, path / 'model.pt')
    
    def load(self, path: Path):
        path = Path(path)
        checkpoint = torch.load(path / 'model.pt', map_location=self.device)
        self.config = checkpoint['config']
        self.model = HSDMVAFModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
