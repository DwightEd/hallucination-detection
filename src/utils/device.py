"""Device management utilities for multi-GPU support.

设备管理工具模块，提供：
- 多GPU设备检测和管理
- 模型设备映射处理
- GPU显存信息查询
- 设备同步操作

Usage:
    from src.utils.device import (
        get_model_device,
        get_all_cuda_devices,
        get_gpu_memory_info,
        synchronize_all_gpus,
        select_best_device,
    )
"""
from typing import Optional, Union, Dict, List, Any
import logging
import torch
from torch import nn, Tensor

logger = logging.getLogger(__name__)


# =============================================================================
# 基础设备操作
# =============================================================================

def get_model_device(model: nn.Module) -> torch.device:
    """获取模型所在的设备，支持多GPU设备映射。
    
    对于使用 device_map 加载的多GPU模型，会查找嵌入层所在设备作为输入设备。
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型的主设备（用于输入）
    """
    # 检查是否有 hf_device_map（Hugging Face 多GPU模型）
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        
        # 查找嵌入层设备（通常是输入设备）
        embedding_keys = ['embed_tokens', 'wte', 'word_embeddings', 'embed_in', 'transformer.wte']
        for key in embedding_keys:
            if key in device_map:
                device = device_map[key]
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                elif isinstance(device, str):
                    return torch.device(device)
        
        # 如果没找到嵌入层，使用第一个设备
        first_device = list(device_map.values())[0]
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        elif isinstance(first_device, str):
            return torch.device(first_device)
    
    # 标准方式：从模型参数获取设备
    try:
        return next(model.parameters()).device
    except StopIteration:
        # 模型没有参数，默认CPU
        return torch.device("cpu")


def to_device(tensor: Tensor, model: nn.Module) -> Tensor:
    """将张量移动到模型所在设备。
    
    Args:
        tensor: 要移动的张量
        model: 目标模型
        
    Returns:
        移动后的张量
    """
    device = get_model_device(model)
    return tensor.to(device)


def ensure_cpu(tensor: Optional[Tensor]) -> Optional[Tensor]:
    """将张量移动到CPU（如果不为None）。
    
    Args:
        tensor: 张量或None
        
    Returns:
        CPU上的张量或None
    """
    if tensor is None:
        return None
    return tensor.cpu()


def get_available_device() -> str:
    """获取最佳可用设备。
    
    Returns:
        "cuda" 如果CUDA可用，否则 "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_device(device: Union[str, torch.device]) -> torch.device:
    """设置张量创建的默认设备。
    
    Args:
        device: 设备字符串或torch.device
        
    Returns:
        设置的设备
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    return device


# =============================================================================
# 多GPU支持
# =============================================================================

def get_all_cuda_devices() -> List[torch.device]:
    """获取所有可用的CUDA设备列表。
    
    Returns:
        CUDA设备列表，如果CUDA不可用则返回空列表
    """
    if not torch.cuda.is_available():
        return []
    
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def get_gpu_memory_info(device_id: Optional[int] = None) -> Dict[str, Any]:
    """获取GPU显存信息。
    
    Args:
        device_id: 指定GPU ID，None表示获取所有GPU的汇总信息
        
    Returns:
        显存信息字典，包含：
        - total_gb: 总显存(GB)
        - allocated_gb: 已分配显存(GB)
        - reserved_gb: 已保留显存(GB)
        - free_gb: 可用显存(GB)
    """
    if not torch.cuda.is_available():
        return {
            "total_gb": 0,
            "allocated_gb": 0,
            "reserved_gb": 0,
            "free_gb": 0,
            "available": False,
        }
    
    if device_id is not None:
        # 单个GPU
        props = torch.cuda.get_device_properties(device_id)
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        total = props.total_memory
        
        return {
            "device_id": device_id,
            "device_name": props.name,
            "total_gb": total / 1024**3,
            "allocated_gb": allocated / 1024**3,
            "reserved_gb": reserved / 1024**3,
            "free_gb": (total - reserved) / 1024**3,
            "available": True,
        }
    else:
        # 所有GPU汇总
        total = 0
        allocated = 0
        reserved = 0
        devices = []
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            dev_allocated = torch.cuda.memory_allocated(i)
            dev_reserved = torch.cuda.memory_reserved(i)
            dev_total = props.total_memory
            
            total += dev_total
            allocated += dev_allocated
            reserved += dev_reserved
            
            devices.append({
                "id": i,
                "name": props.name,
                "total_gb": dev_total / 1024**3,
                "free_gb": (dev_total - dev_reserved) / 1024**3,
            })
        
        return {
            "device_count": torch.cuda.device_count(),
            "devices": devices,
            "total_gb": total / 1024**3,
            "allocated_gb": allocated / 1024**3,
            "reserved_gb": reserved / 1024**3,
            "free_gb": (total - reserved) / 1024**3,
            "available": True,
        }


def get_device_info() -> Dict[str, Any]:
    """获取所有可用设备的详细信息。
    
    Returns:
        设备信息字典
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": get_available_device(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_info = get_gpu_memory_info(i)
            info[f"device_{i}"] = {
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "free_memory_gb": mem_info["free_gb"],
                "major": props.major,
                "minor": props.minor,
            }
    
    return info


def synchronize_all_gpus() -> None:
    """同步所有GPU设备。
    
    等待所有GPU上的操作完成，用于多GPU场景下的同步点。
    """
    if not torch.cuda.is_available():
        return
    
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()


def select_best_device(min_free_gb: float = 4.0) -> torch.device:
    """选择空闲显存最大的GPU设备。
    
    Args:
        min_free_gb: 最小空闲显存要求(GB)
        
    Returns:
        最佳设备，如果没有满足条件的GPU则返回CPU
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    best_device = None
    max_free = 0
    
    for i in range(torch.cuda.device_count()):
        mem_info = get_gpu_memory_info(i)
        free_gb = mem_info["free_gb"]
        
        if free_gb > max_free and free_gb >= min_free_gb:
            max_free = free_gb
            best_device = i
    
    if best_device is not None:
        return torch.device(f"cuda:{best_device}")
    
    logger.warning(f"No GPU with >= {min_free_gb}GB free memory, falling back to CPU")
    return torch.device("cpu")


def balance_memory_check(
    required_gb: float,
    strategy: str = "any"
) -> Dict[str, Any]:
    """检查GPU显存是否满足需求。
    
    Args:
        required_gb: 所需显存(GB)
        strategy: 检查策略
            - "any": 任一GPU满足即可
            - "all": 所有GPU都需满足
            - "total": 总显存满足即可
            
    Returns:
        检查结果字典，包含：
        - satisfied: 是否满足
        - available_devices: 满足条件的设备列表
        - message: 说明信息
    """
    if not torch.cuda.is_available():
        return {
            "satisfied": False,
            "available_devices": [],
            "message": "CUDA not available",
        }
    
    mem_info = get_gpu_memory_info()
    available_devices = []
    
    for dev in mem_info["devices"]:
        if dev["free_gb"] >= required_gb:
            available_devices.append(dev["id"])
    
    if strategy == "any":
        satisfied = len(available_devices) > 0
        message = f"{len(available_devices)} GPU(s) have >= {required_gb}GB free"
    elif strategy == "all":
        satisfied = len(available_devices) == mem_info["device_count"]
        message = f"{len(available_devices)}/{mem_info['device_count']} GPUs have >= {required_gb}GB free"
    elif strategy == "total":
        satisfied = mem_info["free_gb"] >= required_gb
        message = f"Total free: {mem_info['free_gb']:.1f}GB, required: {required_gb}GB"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return {
        "satisfied": satisfied,
        "available_devices": available_devices,
        "message": message,
    }


# =============================================================================
# 多GPU模型辅助
# =============================================================================

def get_model_device_map(model: nn.Module) -> Optional[Dict[str, Any]]:
    """获取模型的设备映射（如果存在）。
    
    Args:
        model: PyTorch模型
        
    Returns:
        设备映射字典，如果不是多GPU模型则返回None
    """
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        return dict(model.hf_device_map)
    return None


def is_multi_gpu_model(model: nn.Module) -> bool:
    """检查模型是否分布在多个GPU上。
    
    Args:
        model: PyTorch模型
        
    Returns:
        是否是多GPU模型
    """
    device_map = get_model_device_map(model)
    if device_map is None:
        return False
    
    # 收集所有唯一设备
    devices = set()
    for device in device_map.values():
        if isinstance(device, int):
            devices.add(device)
        elif isinstance(device, str) and device.startswith("cuda"):
            devices.add(device)
    
    return len(devices) > 1


def log_device_distribution(model: nn.Module, logger_instance: Optional[logging.Logger] = None) -> None:
    """记录模型的设备分布情况。
    
    Args:
        model: PyTorch模型
        logger_instance: 日志记录器，None则使用模块默认logger
    """
    log = logger_instance or logger
    
    device_map = get_model_device_map(model)
    if device_map is None:
        device = get_model_device(model)
        log.info(f"Model on single device: {device}")
        return
    
    # 统计每个设备上的层数
    device_counts: Dict[Any, int] = {}
    for layer, device in device_map.items():
        device_counts[device] = device_counts.get(device, 0) + 1
    
    log.info("Model device distribution:")
    for device, count in sorted(device_counts.items(), key=lambda x: str(x[0])):
        if isinstance(device, int):
            device_str = f"cuda:{device}"
        else:
            device_str = str(device)
        log.info(f"  {device_str}: {count} layers")
