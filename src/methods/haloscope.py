"""
HaloScope - Harnessing Unlabeled LLM Generations for Hallucination Detection

基于隐藏状态SVD的无监督幻觉检测方法
论文: https://arxiv.org/abs/2409.17504 (NeurIPS'24)
GitHub: https://github.com/deeplearning-wisc/haloscope

核心思想：
1. 将问题前置于生成的答案，提取最后一个token的hidden state
2. 对隐藏状态进行SVD分解，识别幻觉子空间
3. 使用加权投影平方作为成员估计分数
4. 训练两层MLP分类器

⚠️ 重要修正（相对于之前的实现）：
- Token选择: 使用最后一个token (last token)，而非mean pooling
- SVD分数: 使用投影平方 (projection²)，而非绝对值
- 层选择: 使用中间层 (8-14)，而非后半层
- 分类器: 使用两层MLP (d->1024->1)，而非LogisticRegression
- 随机种子: 固定为41
"""

import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

# =============================================================================
# 原始HaloScope使用的两层MLP分类器
# =============================================================================

class HaloScopeMLP(nn.Module):
    """
    原始HaloScope使用的两层MLP架构
    
    Architecture: input_dim -> 1024 -> 1
    Activation: ReLU
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class HaloScopeDetector:
    """
    HaloScope检测器 - 忠实复现原始论文实现
    
    关键设计选择 (来自论文):
    1. 使用最后一个token的embedding (last-token)
    2. 使用中间层 (8-14层 for 32层模型)
    3. SVD分数使用投影平方: ζ_i = (1/k) Σ σ_j · ⟨f̄_i, v_j⟩²
    4. 两层MLP分类器 (d -> 1024 -> 1)
    5. 随机种子固定为41
    """
    
    # 原始论文使用的随机种子
    PAPER_SEED = 41
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检测器
        
        Args:
            config: 方法配置
        """
        self.config = config
        self.detection_cfg = config.get('detection', {})
        self.feature_cfg = config.get('feature_config', {})
        
        # SVD相关参数
        self.weighted_svd = self.detection_cfg.get('weighted_svd', True)
        self.feat_loc_svd = self.detection_cfg.get('feat_loc_svd', 3)
        
        # ⚠️ 修正: 使用specific层选择，默认中间层
        self.layer_selection = self.detection_cfg.get('layer_selection', 'middle')
        self.specific_layers = self.detection_cfg.get('specific_layers', [8, 9, 10, 11, 12, 13, 14])
        
        # SVD配置
        svd_cfg = config.get('svd_config', {})
        self.n_components = svd_cfg.get('n_components', 10)
        self.k = svd_cfg.get('k', 5)  # 用于分数计算的主成分数量
        self.center = svd_cfg.get('center', True)
        
        # ⚠️ 修正: Token选择策略 - 必须使用last token
        self.token_selection = self.feature_cfg.get('token_selection', 'last')
        
        # MLP训练配置 (来自原始实现)
        mlp_cfg = config.get('mlp_config', {})
        self.mlp_hidden_dim = mlp_cfg.get('hidden_dim', 1024)
        self.mlp_lr = mlp_cfg.get('learning_rate', 0.05)
        self.mlp_epochs = mlp_cfg.get('epochs', 50)
        self.mlp_batch_size = mlp_cfg.get('batch_size', 512)
        self.mlp_weight_decay = mlp_cfg.get('weight_decay', 3e-4)
        
        # 用于存储分布估计
        self.mean_hidden = None
        self.svd_components = None  # V^T (右奇异向量)
        self.singular_values = None
        self.reference_scores = None
        
        # MLP分类器
        self.mlp_classifier = None
        self.is_fitted = False
    
    def _select_layers(self, num_layers: int) -> List[int]:
        """
        选择要使用的层
        
        ⚠️ 修正: 默认使用中间层 (8-14)，这是论文发现的最优范围
        """
        if self.layer_selection == 'all':
            return list(range(num_layers))
        elif self.layer_selection == 'last_half':
            return list(range(num_layers // 2, num_layers))
        elif self.layer_selection == 'middle':
            # 论文发现的最优层范围: 8-14 for 32层模型
            # 按比例缩放到其他模型
            start = max(0, int(num_layers * 0.25))  # ~25%
            end = min(num_layers, int(num_layers * 0.5))  # ~50%
            return list(range(start, end))
        elif self.layer_selection == 'specific':
            # 过滤掉超出范围的层
            return [l for l in self.specific_layers if 0 <= l < num_layers]
        else:
            logger.warning(f"Unknown layer_selection: {self.layer_selection}, using middle layers")
            start = max(0, int(num_layers * 0.25))
            end = min(num_layers, int(num_layers * 0.5))
            return list(range(start, end))
    
    def _extract_last_token_embedding(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        提取最后一个token的embedding
        
        ⚠️ 这是HaloScope的关键设计: 只使用last token
        
        论文原文: "we prepend the question to the generated answer and 
        use the last-token embedding to identify the subspace"
        
        Args:
            hidden_states: [num_layers, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
            
        Returns:
            [num_selected_layers, hidden_dim] 或 [hidden_dim]
        """
        if len(hidden_states.shape) == 2:
            # [seq_len, hidden_dim] -> 取最后一个token
            return hidden_states[-1, :]
        elif len(hidden_states.shape) == 3:
            # [num_layers, seq_len, hidden_dim] -> 每层取最后一个token
            num_layers = hidden_states.shape[0]
            selected_layers = self._select_layers(num_layers)
            # 取每层的最后一个token
            result = hidden_states[selected_layers, -1, :]  # [n_selected, hidden_dim]
            return result
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
    
    def _extract_hidden_states(self, data: Dict[str, Any]) -> np.ndarray:
        """
        从数据中提取隐藏状态
        
        ⚠️ 修正: 使用last token而非mean pooling
        
        Args:
            data: 包含hidden_states的数据字典
            
        Returns:
            提取后的特征向量
        """
        hidden_states = data.get('hidden_states')
        
        if hidden_states is None:
            raise ValueError("hidden_states not found in data")
        
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        # Handle NaN and Inf values
        if np.any(~np.isfinite(hidden_states)):
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 确保是3D: [num_layers, seq_len, hidden_dim]
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[np.newaxis, ...]
        
        # ⚠️ 关键修正: 使用last token embedding
        last_token_emb = self._extract_last_token_embedding(hidden_states)
        
        return last_token_emb
    
    def _compute_svd_score(self, hidden: np.ndarray) -> float:
        """
        计算SVD成员估计分数
        
        ⚠️ 修正: 使用投影平方公式
        
        原始公式: ζ_i = (1/k) Σ_{j=1}^{k} σ_j · ⟨f̄_i, v_j⟩²
        
        Args:
            hidden: 隐藏状态 [num_layers, hidden_dim] 或 [hidden_dim]
            
        Returns:
            成员估计分数 (越高越可能是幻觉)
        """
        # 展平
        hidden_flat = hidden.flatten()
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(hidden_flat)):
            hidden_flat = np.nan_to_num(hidden_flat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 中心化
        if self.center and self.mean_hidden is not None:
            hidden_flat = hidden_flat - self.mean_hidden
        
        if self.svd_components is None:
            # 如果没有预计算的SVD，直接使用hidden的norm
            norm = np.linalg.norm(hidden_flat)
            return float(norm) if np.isfinite(norm) else 0.0
        
        # 计算投影到前k个主成分
        k = min(self.k, len(self.svd_components))
        
        # ⚠️ 关键修正: 使用投影的平方
        projections_squared = []
        for j in range(k):
            v_j = self.svd_components[j]
            proj = np.dot(hidden_flat, v_j)
            projections_squared.append(proj ** 2)  # 使用平方！
        
        projections_squared = np.array(projections_squared)
        
        # 计算加权分数
        if self.weighted_svd and self.singular_values is not None:
            # 加权版本: ζ = (1/k) Σ σ_j · ⟨f, v_j⟩²
            weights = self.singular_values[:k]
            score = np.sum(weights * projections_squared) / k
        else:
            # 非加权版本
            score = np.mean(projections_squared)
        
        return float(score) if np.isfinite(score) else 0.0
    
    def fit(self, unlabeled_data: List[Dict[str, Any]]):
        """
        使用unlabeled数据估计幻觉子空间
        
        这是HaloScope的第一阶段: 成员估计 (Membership Estimation)
        
        Args:
            unlabeled_data: 无标签数据列表
        """
        logger.info(f"Fitting HaloScope SVD on {len(unlabeled_data)} samples")
        
        # 设置随机种子
        np.random.seed(self.PAPER_SEED)
        
        # 提取所有隐藏状态
        all_hidden = []
        for item in unlabeled_data:
            try:
                hidden = self._extract_hidden_states(item)
                all_hidden.append(hidden.flatten())
            except Exception as e:
                logger.warning(f"Failed to extract hidden states: {e}")
        
        if not all_hidden:
            logger.warning("No hidden states extracted, using default parameters")
            return
        
        all_hidden = np.array(all_hidden)
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(all_hidden)):
            logger.warning("Found NaN/Inf in hidden states, replacing with zeros")
            all_hidden = np.nan_to_num(all_hidden, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 计算均值用于中心化
        self.mean_hidden = all_hidden.mean(axis=0)
        
        # 中心化
        centered = all_hidden - self.mean_hidden
        
        # SVD分解
        try:
            # 使用numpy的SVD (与原始实现一致)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            
            n_components = min(self.n_components, len(S))
            self.svd_components = Vt[:n_components]  # 前n个右奇异向量
            self.singular_values = S[:n_components]
            
            logger.info(f"SVD completed, top singular values: {self.singular_values[:5]}")
        except Exception as e:
            logger.warning(f"SVD failed: {e}, using fallback mode")
            self.svd_components = None
            self.singular_values = None
        
        # 计算参考分数分布 (用于归一化和阈值选择)
        self.reference_scores = []
        for hidden_flat in all_hidden:
            score = self._compute_svd_score(hidden_flat.reshape(-1))
            self.reference_scores.append(score)
        
        self.reference_scores = np.array(self.reference_scores)
        
        valid_scores = self.reference_scores[np.isfinite(self.reference_scores)]
        if len(valid_scores) > 0:
            logger.info(f"Reference scores: mean={valid_scores.mean():.4f}, std={valid_scores.std():.4f}")
    
    def train_classifier(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        训练两层MLP分类器
        
        这是HaloScope的第二阶段: 分类器训练
        使用SVD分数作为伪标签或真实标签
        
        Args:
            features: 训练特征 [N, feature_dim]
            labels: 训练标签 [N]
            val_features: 验证特征 (可选)
            val_labels: 验证标签 (可选)
            
        Returns:
            训练指标
        """
        # 设置随机种子
        torch.manual_seed(self.PAPER_SEED)
        np.random.seed(self.PAPER_SEED)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建MLP
        input_dim = features.shape[1]
        self.mlp_classifier = HaloScopeMLP(input_dim, self.mlp_hidden_dim).to(device)
        
        # 准备数据
        X_train = torch.FloatTensor(features).to(device)
        y_train = torch.FloatTensor(labels).to(device)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.mlp_batch_size,
            shuffle=True,
        )
        
        # 优化器: SGD with cosine decay (原始实现)
        optimizer = optim.SGD(
            self.mlp_classifier.parameters(),
            lr=self.mlp_lr,
            weight_decay=self.mlp_weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.mlp_epochs)
        
        # 损失函数
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环
        self.mlp_classifier.train()
        metrics = {'train_losses': []}
        
        for epoch in range(self.mlp_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.mlp_classifier(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            metrics['train_losses'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.mlp_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        
        # 验证集评估
        if val_features is not None and val_labels is not None:
            val_preds = self.predict_proba(val_features)
            from sklearn.metrics import roc_auc_score
            metrics['val_auroc'] = roc_auc_score(val_labels, val_preds)
            logger.info(f"Validation AUROC: {metrics['val_auroc']:.4f}")
        
        return metrics
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        使用MLP预测幻觉概率
        
        Args:
            features: 特征 [N, feature_dim] 或 [feature_dim]
            
        Returns:
            幻觉概率 [N] (始终返回1D数组)
        """
        if self.mlp_classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")
        
        # 确保输入是2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        device = next(self.mlp_classifier.parameters()).device
        X = torch.FloatTensor(features).to(device)
        
        self.mlp_classifier.eval()
        with torch.no_grad():
            logits = self.mlp_classifier(X)  # [N, 1]
            probs = torch.sigmoid(logits)    # [N, 1]
        
        # ⚠️ 修复: 使用 flatten() 而非 squeeze()，确保始终返回1D数组
        # squeeze() 在单样本时会返回0-D数组，导致 probs[0] 报错
        result = probs.cpu().numpy().flatten()  # [N]
        
        return result
    
    def predict_score(self, data: Dict[str, Any]) -> float:
        """
        预测单个样本的幻觉分数 (使用SVD分数)
        
        Args:
            data: 包含hidden_states的数据
            
        Returns:
            成员估计分数
        """
        hidden = self._extract_hidden_states(data)
        return self._compute_svd_score(hidden)
    
    def save(self, path: Path):
        """保存检测器状态"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': self.config,
            'mean_hidden': self.mean_hidden,
            'svd_components': self.svd_components,
            'singular_values': self.singular_values,
            'reference_scores': self.reference_scores,
            'is_fitted': self.is_fitted,
        }
        
        np.savez(path / 'haloscope_svd_state.npz', **{
            k: v for k, v in state.items() if v is not None and k not in ['config', 'is_fitted']
        })
        
        # 保存MLP
        if self.mlp_classifier is not None:
            torch.save(self.mlp_classifier.state_dict(), path / 'mlp_classifier.pt')
            # 保存MLP配置
            mlp_config = {
                'input_dim': self.mlp_classifier.fc1.in_features,
                'hidden_dim': self.mlp_hidden_dim,
            }
            import json
            with open(path / 'mlp_config.json', 'w') as f:
                json.dump(mlp_config, f)
        
        import json
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"HaloScope state saved to {path}")
    
    def load(self, path: Path):
        """加载检测器状态"""
        path = Path(path)
        
        import json
        with open(path / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # 重新初始化配置
        self.__init__(self.config)
        
        # 加载SVD状态
        state_file = path / 'haloscope_svd_state.npz'
        if state_file.exists():
            state = np.load(state_file, allow_pickle=True)
            self.mean_hidden = state.get('mean_hidden')
            self.svd_components = state.get('svd_components')
            self.singular_values = state.get('singular_values')
            self.reference_scores = state.get('reference_scores')
        
        # 加载MLP
        mlp_path = path / 'mlp_classifier.pt'
        mlp_config_path = path / 'mlp_config.json'
        if mlp_path.exists() and mlp_config_path.exists():
            with open(mlp_config_path, 'r') as f:
                mlp_config = json.load(f)
            self.mlp_classifier = HaloScopeMLP(
                mlp_config['input_dim'],
                mlp_config['hidden_dim']
            )
            self.mlp_classifier.load_state_dict(torch.load(mlp_path))
            self.is_fitted = True
        
        logger.info(f"HaloScope state loaded from {path}")


# =============================================================================
# HaloScopeMethod - BaseMethod 兼容的包装类
# =============================================================================

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod


@METHODS.register("haloscope", aliases=["halo", "haloscope_svd"])
class HaloScopeMethod(BaseMethod):
    """
    HaloScope方法的BaseMethod兼容包装类
    
    忠实复现原始论文实现:
    - 使用last token embedding
    - 使用SVD投影平方作为成员估计分数
    - 使用两层MLP分类器
    - 随机种子固定为41
    """
    
    # 原始论文使用的随机种子
    PAPER_SEED = 41
    
    def __init__(self, config: Optional[MethodConfig] = None):
        """初始化方法"""
        super().__init__(config)
        
        # 创建底层检测器配置
        detector_config = {
            'detection': {
                'layer_selection': 'middle',  # ⚠️ 修正: 使用中间层
                'specific_layers': [8, 9, 10, 11, 12, 13, 14],
                'weighted_svd': True,
                'feat_loc_svd': 3,
            },
            'feature_config': {
                'token_selection': 'last',  # ⚠️ 修正: 使用last token
                'response_only': False,  # 不单独处理response
            },
            'svd_config': {
                'n_components': 10,
                'k': 5,  # 用于分数计算的主成分数
                'center': True,
            },
            'mlp_config': {
                'hidden_dim': 1024,
                'learning_rate': 0.05,
                'epochs': 50,
                'batch_size': 512,
                'weight_decay': 3e-4,
            }
        }
        
        # 如果config中有params，覆盖默认值
        if config and config.params:
            for key, value in config.params.items():
                if key in detector_config:
                    if isinstance(value, dict):
                        detector_config[key].update(value)
                    else:
                        detector_config[key] = value
        
        self.detector = HaloScopeDetector(detector_config)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """从ExtractedFeatures提取方法特定的特征向量
        
        ⚠️ 关键: 使用last token embedding
        
        Args:
            features: 提取的特征
            
        Returns:
            特征向量 [feature_dim]
        """
        # 使用懒加载获取 hidden_states
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            raise ValueError(f"Sample {features.sample_id} has no hidden_states. "
                           "Make sure hidden_states are extracted and saved.")
        
        # 转换为numpy
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        # Handle NaN/Inf in hidden states
        if np.any(~np.isfinite(hidden_states)):
            hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # ⚠️ 关键修正: 提取last token embedding
        data = {'hidden_states': hidden_states}
        processed = self.detector._extract_hidden_states(data)
        
        # 展平作为特征
        result = processed.flatten()
        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result
    
    def extract_token_features(self, features: ExtractedFeatures, token_idx: int) -> Optional[np.ndarray]:
        """Extract features for a single token.
        
        ⚠️ 注意: HaloScope是sample-level方法，token级特征仅用于分析
        
        Args:
            features: Extracted features from model
            token_idx: Token index in the response
            
        Returns:
            Feature vector for this token
        """
        hidden_states = features.hidden_states
        if hidden_states is None:
            hidden_states = features.get_hidden_states()
        
        if hidden_states is None:
            return None
        
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.cpu().float().numpy()
        
        if len(hidden_states.shape) != 3:
            return None
        
        n_layers, seq_len, hidden_dim = hidden_states.shape
        
        # 计算绝对索引
        prompt_len = features.prompt_len if hasattr(features, 'prompt_len') else 0
        abs_idx = prompt_len + token_idx
        
        if abs_idx >= seq_len:
            return None
        
        # 层选择
        layer_selection = self.detector.config.get('detection', {}).get('layer_selection', 'middle')
        
        if layer_selection == 'middle':
            start_layer = max(0, int(n_layers * 0.25))
            end_layer = min(n_layers, int(n_layers * 0.5))
        elif layer_selection == 'last_half':
            start_layer = n_layers // 2
            end_layer = n_layers
        else:
            start_layer = 0
            end_layer = n_layers
        
        # 提取该位置的hidden states
        token_hidden = hidden_states[start_layer:end_layer, abs_idx, :]
        
        result = token_hidden.flatten().astype(np.float32)
        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result
    
    def fit(self, features_list: List[ExtractedFeatures], labels: Optional[List[int]] = None, cv: bool = True) -> Dict[str, float]:
        """训练方法
        
        两阶段训练:
        1. 使用所有数据进行SVD拟合 (无监督)
        2. 训练MLP分类器 (监督/半监督)
        """
        # 设置随机种子
        np.random.seed(self.PAPER_SEED)
        
        # 提取隐藏状态用于SVD拟合
        all_data = []
        for feat in features_list:
            hs = feat.hidden_states
            if hs is None:
                hs = feat.get_hidden_states()
            
            if hs is not None:
                if isinstance(hs, torch.Tensor):
                    hs = hs.cpu().float().numpy()
                
                all_data.append({'hidden_states': hs})
                feat.release_large_features()
        
        # 阶段1: 拟合SVD（无监督）
        if all_data:
            self.detector.fit(all_data)
        
        # 阶段2: 提取特征并训练分类器
        X = []
        y = []
        
        for i, feat in enumerate(features_list):
            try:
                x = self.extract_method_features(feat)
                if x is not None and not np.any(np.isnan(x)):
                    X.append(x)
                    label = labels[i] if labels else feat.label
                    if label is not None:
                        y.append(label)
            except Exception as e:
                logger.warning(f"Failed to extract features for {feat.sample_id}: {e}")
        
        if len(X) == 0:
            raise ValueError("No valid features extracted")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training MLP on {len(X)} samples, feature dim={X.shape[1]}")
        self._feature_dim = X.shape[1]
        
        # 训练MLP分类器
        metrics = self.detector.train_classifier(X, y)
        
        self.is_fitted = True
        
        return metrics
    
    def predict(self, features: ExtractedFeatures) -> 'Prediction':
        """预测单个样本"""
        from src.core import Prediction
        
        if not self.is_fitted:
            raise ValueError("Method not fitted. Call fit() first.")
        
        x = self.extract_method_features(features)
        proba = self.detector.predict_proba(x.reshape(1, -1))[0]
        
        return Prediction(
            sample_id=features.sample_id,
            score=float(proba),
            label=1 if proba > 0.5 else 0,
            confidence=abs(proba - 0.5) * 2,
        )
    
    def save(self, path: Path) -> None:
        """保存方法"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为目录
        if path.suffix:
            # 如果有后缀，使用父目录/文件名作为目录
            save_dir = path.parent / path.stem
        else:
            save_dir = path
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存检测器状态
        self.detector.save(save_dir)
        
        # 保存方法配置
        state = {
            "config": self.config,
            "is_fitted": self.is_fitted,
            "feature_dim": self._feature_dim,
        }
        
        with open(save_dir / "method_state.pkl", "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved HaloScope method to {save_dir}")
    
    def load(self, path: Path) -> None:
        """加载方法"""
        path = Path(path)
        
        # 确定加载目录
        if path.is_file():
            load_dir = path.parent / path.stem
        else:
            load_dir = path
        
        # 加载检测器状态
        self.detector.load(load_dir)
        
        # 加载方法状态
        method_state_path = load_dir / "method_state.pkl"
        if method_state_path.exists():
            with open(method_state_path, "rb") as f:
                state = pickle.load(f)
            self.config = state["config"]
            self.is_fitted = state["is_fitted"]
            self._feature_dim = state["feature_dim"]
        
        logger.info(f"Loaded HaloScope method from {load_dir}")