"""HSDMVAF Method - Framework Integration.

严格按照原论文实现: "Hallucinated Span Detection with Multi-View Attention Features"
GitHub: https://github.com/Ogamon958/mva_hal_det

支持两种模式:
- sample-level: MVA特征聚合 + LogisticRegression (快速)
- token-level: Transformer+CRF序列标注 (精确)
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from src.core import ExtractedFeatures, MethodConfig, METHODS, Prediction, MethodError
from src.methods.base import BaseMethod

from .features import (
    compute_multi_view_attention_features,
    compute_mva_features_from_diags,
    compute_mva_sample_features,
    safe_to_numpy,
    safe_to_tensor,
)
from .model import HSDMVAFModel, CRFLayer

logger = logging.getLogger(__name__)


# =============================================================================
# 数据集类 (用于token-level训练)
# =============================================================================

class MVADataset(Dataset):
    """MVA特征数据集。"""
    
    def __init__(self, features: List[torch.Tensor], labels: List[torch.Tensor]):
        """
        Args:
            features: 特征张量列表 [seq_len, feature_dim]
            labels: 标签张量列表 [seq_len]
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    """带padding的批处理函数。"""
    features, labels = zip(*batch)
    
    # 获取最大长度
    max_len = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]
    
    # Padding
    batch_features = torch.zeros(len(features), max_len, feature_dim)
    batch_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    batch_mask = torch.zeros(len(features), max_len)
    
    for i, (f, l) in enumerate(zip(features, labels)):
        seq_len = f.shape[0]
        batch_features[i, :seq_len] = f
        batch_labels[i, :seq_len] = l
        batch_mask[i, :seq_len] = 1
    
    return batch_features, batch_labels, batch_mask


# =============================================================================
# HSDMVAF检测器 (Token-level)
# =============================================================================

class HSDMVAFDetector:
    """HSDMVAF独立检测器 (用于token-level检测)。
    
    包含完整的Transformer+CRF模型和训练逻辑。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，包含encoder配置、训练参数等
        """
        self.config = config
        self.model: Optional[HSDMVAFModel] = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征提取参数
        self.max_layers = config.get('max_layers', 4)
        self.max_heads = config.get('max_heads', 8)
    
    def _extract_mva_features(
        self,
        data: Union[Dict[str, Any], ExtractedFeatures],
    ) -> torch.Tensor:
        """从数据提取MVA特征。
        
        Args:
            data: 数据字典或ExtractedFeatures对象
            
        Returns:
            MVA特征张量 [seq_len, feature_dim]
        """
        if isinstance(data, ExtractedFeatures):
            prompt_len = data.prompt_len
            response_len = data.response_len
            full_attention = data.full_attention
            if full_attention is None:
                full_attention = data.get_full_attention()
            attn_diags = data.attn_diags
            attn_entropy = data.attn_entropy
        else:
            prompt_len = data.get('prompt_len', 0)
            response_len = data.get('response_len', 100)
            full_attention = data.get('full_attention')
            attn_diags = data.get('attn_diags')
            attn_entropy = data.get('attn_entropy')
        
        if full_attention is not None:
            return compute_multi_view_attention_features(
                full_attention, prompt_len, response_len,
                normalize=False,
                max_layers=self.max_layers,
                max_heads=self.max_heads,
            )
        elif attn_diags is not None:
            return compute_mva_features_from_diags(
                attn_diags, attn_entropy,
                prompt_len, response_len,
            )
        else:
            raise ValueError("需要full_attention或attn_diags")
    
    def compute_standardization_params(
        self,
        features_list: List[torch.Tensor],
    ) -> tuple:
        """从训练数据计算标准化参数。
        
        Args:
            features_list: 特征张量列表
            
        Returns:
            (mean, std)
        """
        all_features = []
        for feat in features_list:
            all_features.append(feat.reshape(-1, feat.shape[-1]))
        
        all_features = torch.cat(all_features, dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        
        return mean, std
    
    def fit(
        self,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """训练模型。
        
        Args:
            train_data: 训练数据列表，每个元素包含:
                - full_attention 或 attn_diags
                - prompt_len, response_len
                - hallucination_labels: [seq_len] token-level标签
            val_data: 验证数据 (可选)
            
        Returns:
            训练指标
        """
        # 提取特征
        train_features = []
        train_labels = []
        
        for item in train_data:
            try:
                feat = self._extract_mva_features(item)
                labels = item.get('hallucination_labels')
                
                if labels is None:
                    continue
                
                labels = safe_to_tensor(labels, dtype=torch.long)
                
                # 确保长度匹配
                min_len = min(feat.shape[0], labels.shape[0])
                feat = feat[:min_len]
                labels = labels[:min_len]
                
                train_features.append(feat)
                train_labels.append(labels)
            except Exception as e:
                logger.warning(f"特征提取失败: {e}")
                continue
        
        if len(train_features) == 0:
            raise MethodError("没有有效的训练数据")
        
        logger.info(f"提取了 {len(train_features)} 个训练样本的特征")
        
        # 计算标准化参数
        mean, std = self.compute_standardization_params(train_features)
        
        # 初始化模型
        input_dim = train_features[0].shape[-1]
        model_config = {
            'input_dim': input_dim,
            'encoder': self.config.get('encoder', {}),
            'use_crf': self.config.get('use_crf', True),
        }
        
        self.model = HSDMVAFModel(model_config).to(self.device)
        self.model.set_standardization_params(mean.to(self.device), std.to(self.device))
        
        # 准备数据加载器
        train_dataset = MVADataset(train_features, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 16),
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        # 训练参数
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 50)
        lr = training_config.get('learning_rate', 1e-4)
        weight_decay = training_config.get('weight_decay', 0.01)
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 训练循环
        best_loss = float('inf')
        patience = training_config.get('early_stopping', {}).get('patience', 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for features, labels, mask in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(features, labels, mask)
                loss = output['loss']
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停于epoch {epoch+1}")
                    break
        
        self.is_fitted = True
        
        return {'final_loss': best_loss}
    
    def predict(self, data: Union[Dict[str, Any], ExtractedFeatures]) -> Dict[str, Any]:
        """预测单个样本。
        
        Args:
            data: 数据字典或ExtractedFeatures对象
            
        Returns:
            包含predictions和probabilities的字典
        """
        if not self.is_fitted:
            raise ValueError("检测器未训练")
        
        feat = self._extract_mva_features(data).unsqueeze(0).to(self.device)
        mask = torch.ones(1, feat.shape[1]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(feat, mask)
            probs = self.model.predict_proba(feat, mask)
        
        return {
            'predictions': predictions[0],
            'probabilities': probs[0].cpu().numpy(),
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取状态字典 (用于保存)。"""
        if self.model is None:
            return {}
        
        return {
            'model_state': self.model.get_state_dict(),
            'config': self.config,
            'is_fitted': self.is_fitted,
            'max_layers': self.max_layers,
            'max_heads': self.max_heads,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """加载状态字典。"""
        self.config = state.get('config', self.config)
        self.is_fitted = state.get('is_fitted', False)
        self.max_layers = state.get('max_layers', 4)
        self.max_heads = state.get('max_heads', 8)
        
        if 'model_state' in state and state['model_state']:
            model_state = state['model_state']
            model_config = model_state.get('config', {})
            
            self.model = HSDMVAFModel(model_config).to(self.device)
            self.model.load_state_dict_from_saved(model_state)


# =============================================================================
# Multi-View Attention Encoder (辅助模块)
# =============================================================================

class MultiViewAttentionEncoder(nn.Module):
    """多视角注意力特征编码器。
    
    用于将MVA特征编码为固定维度表示。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len], 1=有效, 0=padding
            
        Returns:
            encoded: [batch, seq_len, hidden_dim]
        """
        x = self.input_proj(x)
        
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        return x


# =============================================================================
# HSDMVAF Method (框架集成)
# =============================================================================

@METHODS.register("hsdmvaf", aliases=["mva", "multi_view_attention"])
class HSDMVAFMethod(BaseMethod):
    """HSDMVAF幻觉检测方法。
    
    基于论文: "Hallucinated Span Detection with Multi-View Attention Features"
    GitHub: https://github.com/Ogamon958/mva_hal_det
    
    特点:
    - 从注意力矩阵提取三种互补特征 (avg_in, div_in, div_out)
    - 支持sample-level和token-level检测
    - Sample-level使用LogisticRegression
    - Token-level使用Transformer+CRF
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.pooling = params.get("pooling", "max")
        self.use_full_attention = params.get("use_full_attention", True)
        self.max_layers = params.get("max_layers", 4)
        self.max_heads = params.get("max_heads", 8)
        
        # 分类器参数
        classifier_params = params.get("classifier_params", {})
        self._classifier_max_iter = classifier_params.get("max_iter", 1000)
        self._classifier_C = classifier_params.get("C", 1.0)
        self._classifier_class_weight = classifier_params.get("class_weight", "balanced")
        
        self._scaler = StandardScaler()
        self._detector: Optional[HSDMVAFDetector] = None
        self._feature_dim: Optional[int] = None
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取MVA特征并聚合为样本级别。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            样本级别特征向量
        """
        full_attention = features.full_attention
        if full_attention is None:
            full_attention = features.get_full_attention()
        
        if self.use_full_attention and full_attention is not None:
            # 安全转换 - 处理BFloat16
            full_attention = safe_to_numpy(full_attention)
            
            # 限制层和头数量
            n_layers, n_heads = full_attention.shape[:2]
            layer_start = max(0, n_layers - self.max_layers)
            head_end = min(self.max_heads, n_heads)
            
            limited_attn = full_attention[layer_start:, :head_end]
            
            feat_vec = compute_mva_sample_features(
                torch.from_numpy(limited_attn),
                prompt_len=features.prompt_len,
                response_len=features.response_len,
                pooling=self.pooling,
                max_layers=None,  # 已经限制过了
                max_heads=None,
            )
            
            # 释放大内存
            features.release_large_features()
            
        elif features.attn_diags is not None:
            # 降级模式: 使用对角线近似
            mva_features = compute_mva_features_from_diags(
                features.attn_diags,
                features.attn_entropy,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
            )
            
            # 聚合
            mva_np = safe_to_numpy(mva_features)
            if self.pooling == "max":
                feat_vec = mva_np.max(axis=0)
            elif self.pooling == "mean":
                feat_vec = mva_np.mean(axis=0)
            else:
                feat_vec = np.concatenate([mva_np.max(axis=0), mva_np.mean(axis=0)])
        else:
            raise MethodError("HSDMVAF需要full_attention或attn_diags")
        
        # 处理无效值
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec.astype(np.float32)
    
    def extract_token_features(self, features: ExtractedFeatures) -> np.ndarray:
        """提取token-level MVA特征。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            Token-level特征 [seq_len, feature_dim]
        """
        full_attention = features.full_attention
        if full_attention is None:
            full_attention = features.get_full_attention()
        
        if full_attention is not None:
            full_attention = safe_to_tensor(full_attention)
            
            mva_features = compute_multi_view_attention_features(
                full_attention,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
                max_layers=self.max_layers,
                max_heads=self.max_heads,
            )
            
            features.release_large_features()
            return safe_to_numpy(mva_features)
        
        elif features.attn_diags is not None:
            mva_features = compute_mva_features_from_diags(
                features.attn_diags,
                features.attn_entropy,
                prompt_len=features.prompt_len,
                response_len=features.response_len,
            )
            return safe_to_numpy(mva_features)
        
        else:
            raise MethodError("HSDMVAF需要full_attention或attn_diags")
    
    def fit(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[int]] = None,
        cv: bool = True,
    ) -> Dict[str, float]:
        """训练HSDMVAF方法 (sample-level)。
        
        Args:
            features_list: ExtractedFeatures列表
            labels: 标签 (如果不在features中)
            cv: 是否进行交叉验证
            
        Returns:
            训练指标
        """
        X = []
        y = []
        
        n_total = len(features_list)
        log_interval = max(1, n_total // 10)
        
        for i, feat in enumerate(features_list):
            if i % log_interval == 0:
                logger.info(f"提取特征: {i}/{n_total} ({100*i/n_total:.0f}%)")
            
            try:
                x = self.extract_method_features(feat)
                if x is not None and len(x) > 0:
                    X.append(x)
                    label = labels[i] if labels else feat.label
                    if label is not None:
                        y.append(label)
                    else:
                        logger.warning(f"样本 {feat.sample_id} 没有标签")
            except Exception as e:
                logger.warning(f"特征提取失败 {feat.sample_id}: {e}")
        
        if len(X) == 0:
            raise MethodError("没有有效的特征")
        
        X = np.array(X)
        y = np.array(y)
        
        self._feature_dim = X.shape[1]
        logger.info(f"特征维度: {X.shape}, 标签: {y.shape}")
        logger.info(f"类别分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
        
        # 标准化
        X_scaled = self._scaler.fit_transform(X)
        
        # 训练分类器
        self.classifier = LogisticRegression(
            max_iter=self._classifier_max_iter,
            C=self._classifier_C,
            class_weight=self._classifier_class_weight,
            random_state=42,
        )
        
        # 交叉验证
        metrics = {}
        if cv and len(y) >= 5:
            try:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(
                    self.classifier, X_scaled, y,
                    cv=min(5, len(y)),
                    scoring='roc_auc'
                )
                metrics['cv_auroc_mean'] = float(np.mean(cv_scores))
                metrics['cv_auroc_std'] = float(np.std(cv_scores))
                logger.info(f"CV AUROC: {metrics['cv_auroc_mean']:.4f} ± {metrics['cv_auroc_std']:.4f}")
            except Exception as e:
                logger.warning(f"交叉验证失败: {e}")
        
        # 最终训练
        self.classifier.fit(X_scaled, y)
        self.is_fitted = True
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> Prediction:
        """预测单个样本。
        
        Args:
            features: ExtractedFeatures对象
            
        Returns:
            Prediction对象
        """
        if not self.is_fitted:
            raise MethodError("方法未训练，请先调用fit()")
        
        # 提取特征
        x = self.extract_method_features(features)
        
        # 标准化
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        
        # 预测
        proba = self.classifier.predict_proba(x_scaled)[0]
        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        return Prediction(
            sample_id=features.sample_id,
            score=score,
            label=1 if score > 0.5 else 0,
            confidence=abs(score - 0.5) * 2,
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """保存模型到单个文件 (统一model.pkl格式)。
        
        Args:
            path: 保存路径
        """
        if not self.is_fitted:
            raise MethodError("无法保存未训练的方法")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "classifier": self.classifier,
            "scaler": self._scaler,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
            # 保存参数以便恢复
            "pooling": self.pooling,
            "use_full_attention": self.use_full_attention,
            "max_layers": self.max_layers,
            "max_heads": self.max_heads,
            # 检测器状态 (如果有)
            "detector_state": self._detector.get_state_dict() if self._detector else None,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"保存方法到 {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """从文件加载模型。
        
        支持新格式 (单文件) 和旧格式 (目录) 的兼容加载。
        
        Args:
            path: 模型路径
        """
        path = Path(path)
        
        # 检查旧格式 (目录)
        if path.is_dir():
            self._load_legacy_format(path)
            return
        
        if not path.exists():
            raise MethodError(f"模型文件不存在: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.classifier = state["classifier"]
        self._scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        self._feature_dim = state.get("feature_dim")
        
        # 恢复参数
        self.pooling = state.get("pooling", "max")
        self.use_full_attention = state.get("use_full_attention", True)
        self.max_layers = state.get("max_layers", 4)
        self.max_heads = state.get("max_heads", 8)
        
        # 恢复检测器 (如果有)
        if state.get("detector_state"):
            self._detector = HSDMVAFDetector(state.get("config", {}).params or {})
            self._detector.load_state_dict(state["detector_state"])
        
        logger.info(f"从 {path} 加载方法")
    
    def _load_legacy_format(self, path: Path) -> None:
        """加载旧格式模型 (目录结构)。
        
        Args:
            path: 模型目录
        """
        logger.info(f"检测到旧格式模型，从目录加载: {path}")
        
        # 尝试加载各个文件
        method_state_path = path / "method_state.pkl"
        detector_path = path / "hsdmvaf_detector.pkl"
        
        if method_state_path.exists():
            with open(method_state_path, "rb") as f:
                method_state = pickle.load(f)
            
            self.config = method_state.get("config", self.config)
            self.classifier = method_state.get("classifier")
            self._scaler = method_state.get("scaler", StandardScaler())
            self.is_fitted = method_state.get("is_fitted", False)
            self._feature_dim = method_state.get("feature_dim")
            
            self.pooling = method_state.get("pooling", "max")
            self.use_full_attention = method_state.get("use_full_attention", True)
            self.max_layers = method_state.get("max_layers", 4)
            self.max_heads = method_state.get("max_heads", 8)
        
        if detector_path.exists():
            with open(detector_path, "rb") as f:
                detector_state = pickle.load(f)
            
            self._detector = HSDMVAFDetector(self.config.params or {})
            self._detector.load_state_dict(detector_state)
        
        logger.info("旧格式模型加载完成")