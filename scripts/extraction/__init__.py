"""Feature extraction pipeline modules.

重构版本：使用统一的 PathManager 进行路径管理。

该模块将 generate_activations.py 的具体实现抽象为独立子模块：
- data_loader: 样本加载和过滤
- output_manager: 输出路径构建和结果整合
- progress: 进度跟踪
- extraction_loop: 主提取循环

Usage:
    from scripts.extraction import (
        load_samples_from_splits,
        build_output_dir,
        ProgressTracker,
        run_extraction_loop,
    )
"""

from .data_loader import (
    load_samples_from_splits,
    get_task_types_from_config,
)

from .output_manager import (
    build_output_dir,
    get_task_suffix,
    get_model_short_name,
    finalize_outputs,
    extract_features_dict,
    save_sample_answer,
)

from .progress import ProgressTracker

from .extraction_loop import run_extraction_loop

# 从 src.core.paths 导入统一的解析函数
from src.core.paths import parse_task_types

__all__ = [
    # Data loading
    "load_samples_from_splits",
    "get_task_types_from_config",
    
    # Path utilities (from centralized module)
    "parse_task_types",
    
    # Output management
    "build_output_dir",
    "get_task_suffix",
    "get_model_short_name",
    "finalize_outputs",
    "extract_features_dict",
    "save_sample_answer",
    
    # Progress tracking
    "ProgressTracker",
    
    # Main extraction
    "run_extraction_loop",
]
