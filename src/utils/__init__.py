"""Utility modules for hallucination detection.

设备管理、内存管理和张量操作工具模块。

This module provides:
- Memory management utilities (GPU memory clearing, tracking)
- Tensor operations (attention, hidden states, token probabilities)
- Device management utilities (multi-GPU support)

Usage:
    from src.utils import (
        clear_gpu_memory,
        extract_attention_diagonal,
        compute_laplacian_diagonal,
        get_model_device,
        get_device_info,
        get_all_cuda_devices,
        synchronize_all_gpus,
    )
"""

from .memory import (
    clear_gpu_memory,
    get_gpu_memory_info,
    log_gpu_memory,
    get_tensor_memory_mb,
    MemoryTracker,
    gpu_memory_scope,
)

from .tensor_ops import (
    # Attention operations
    extract_attention_diagonal,
    compute_attention_row_sums,
    compute_laplacian_diagonal,
    compute_attention_entropy,
    stack_layer_attentions,
    normalize_attention,
    # Hidden state operations
    pool_hidden_states,
    stack_layer_hidden_states,
    # Token probability operations
    compute_token_probs,
    compute_token_entropy,
    compute_top_k_probs,
    compute_perplexity,
)

from .device import (
    # 基础设备操作
    get_model_device,
    to_device,
    ensure_cpu,
    get_available_device,
    set_device,
    # 多GPU支持
    get_device_info,
    get_all_cuda_devices,
    get_gpu_memory_info as get_device_memory_info,
    synchronize_all_gpus,
    select_best_device,
    balance_memory_check,
    # 多GPU模型辅助
    get_model_device_map,
    is_multi_gpu_model,
    log_device_distribution,
)

from .metrics_tracker import (
    MetricsTracker,
    PerformanceMetrics,
    track_metrics,
    measure_model_size,
    format_metrics_table,
)

__all__ = [
    # Memory management
    "clear_gpu_memory",
    "get_gpu_memory_info",
    "log_gpu_memory",
    "get_tensor_memory_mb",
    "MemoryTracker",
    "gpu_memory_scope",
    
    # Attention operations
    "extract_attention_diagonal",
    "compute_attention_row_sums",
    "compute_laplacian_diagonal",
    "compute_attention_entropy",
    "stack_layer_attentions",
    "normalize_attention",
    
    # Hidden state operations
    "pool_hidden_states",
    "stack_layer_hidden_states",
    
    # Token probability operations
    "compute_token_probs",
    "compute_token_entropy",
    "compute_top_k_probs",
    "compute_perplexity",
    
    # Device management - 基础
    "get_model_device",
    "to_device",
    "ensure_cpu",
    "get_available_device",
    "set_device",
    
    # Device management - 多GPU支持
    "get_device_info",
    "get_all_cuda_devices",
    "get_device_memory_info",
    "synchronize_all_gpus",
    "select_best_device",
    "balance_memory_check",
    
    # Device management - 多GPU模型辅助
    "get_model_device_map",
    "is_multi_gpu_model",
    "log_device_distribution",
    
    # Performance metrics tracking
    "MetricsTracker",
    "PerformanceMetrics",
    "track_metrics",
    "measure_model_size",
    "format_metrics_table",
]
