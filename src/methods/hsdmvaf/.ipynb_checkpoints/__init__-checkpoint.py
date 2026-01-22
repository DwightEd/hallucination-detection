"""HSDMVAF - Hallucinated Span Detection with Multi-View Attention Features.

严格按照原论文实现:
- 论文: "Hallucinated Span Detection with Multi-View Attention Features"
- GitHub: https://github.com/Ogamon958/mva_hal_det
- 会议: *SEM 2025

核心创新:
- 从注意力矩阵提取三种互补特征 (avg_in, div_in, div_out)
- Transformer Encoder + CRF 用于序列标注
- 支持 sample-level 和 token-level 检测

=============================================================================
模块结构:
=============================================================================
- features.py: Multi-View Attention 特征计算
  - avg_in (μ): 平均入向注意力
  - div_in (β): 入向注意力多样性 (归一化熵)
  - div_out (γ): 出向注意力多样性 (归一化熵)

- model.py: Transformer+CRF 模型
  - PositionalEncoding: 位置编码
  - CRFLayer: 条件随机场层
  - HSDMVAFModel: 完整模型

- method.py: 框架集成
  - HSDMVAFMethod: 主方法类 (sample-level)
  - HSDMVAFDetector: Token-level 检测器
  - MultiViewAttentionEncoder: 特征编码器

=============================================================================
修复内容:
=============================================================================
1. BFloat16 安全转换 - 解决 "Got unsupported ScalarType BFloat16" 错误
2. 严格按照原论文公式实现 MVA 特征计算
3. 统一 model.pkl 保存格式
4. 支持新旧格式兼容加载

=============================================================================
使用示例:
=============================================================================
```python
from src.methods.hsdmvaf import HSDMVAFMethod, HSDMVAFDetector

# Sample-level 检测
method = HSDMVAFMethod()
method.fit(features_list)
prediction = method.predict(features)

# Token-level 检测 (需要 hallucination_labels)
detector = HSDMVAFDetector(config)
detector.fit(train_data)
result = detector.predict(data)
```
"""

from .method import HSDMVAFMethod, HSDMVAFDetector, MultiViewAttentionEncoder
from .model import HSDMVAFModel, CRFLayer, PositionalEncoding
from .features import (
    compute_multi_view_attention_features,
    compute_mva_features_from_diags,
    compute_mva_sample_features,
    compute_average_incoming_attention,
    compute_incoming_attention_entropy,
    compute_outgoing_attention_entropy,
    safe_to_numpy,
    safe_to_tensor,
)

__all__ = [
    # 主要类
    "HSDMVAFMethod",
    "HSDMVAFDetector",
    "MultiViewAttentionEncoder",
    
    # 模型组件
    "HSDMVAFModel",
    "CRFLayer",
    "PositionalEncoding",
    
    # 特征函数
    "compute_multi_view_attention_features",
    "compute_mva_features_from_diags",
    "compute_mva_sample_features",
    "compute_average_incoming_attention",
    "compute_incoming_attention_entropy",
    "compute_outgoing_attention_entropy",
    
    # 工具函数
    "safe_to_numpy",
    "safe_to_tensor",
]

# 版本信息
__version__ = "1.1.0"
__author__ = "Based on Ogamon958/mva_hal_det"
__paper__ = "Hallucinated Span Detection with Multi-View Attention Features (*SEM 2025)"