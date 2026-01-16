"""Model loading utilities with multi-GPU support.

模型加载工具，支持：
- 多GPU自动分布
- 自定义设备映射策略
- 量化加载（4bit/8bit）
- 模型信息查询

Usage:
    from src.models.loader import load_model, LoadedModel
    
    # 基本加载
    loaded = load_model(config)
    
    # 多GPU配置示例
    config.multi_gpu = {
        "enabled": True,
        "strategy": "balanced",
        "max_memory": {0: "20GB", 1: "20GB"}
    }
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.core import ModelConfig, ModelError

logger = logging.getLogger(__name__)


# =============================================================================
# GPU 信息获取
# =============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """获取所有GPU的详细信息。
    
    Returns:
        GPU信息字典，包含：
        - count: GPU数量
        - devices: 各GPU详情列表
        - total_memory_gb: 总显存
    """
    if not torch.cuda.is_available():
        return {
            "count": 0,
            "devices": [],
            "total_memory_gb": 0,
            "available": False,
        }
    
    devices = []
    total_memory = 0
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        
        devices.append({
            "id": i,
            "name": props.name,
            "total_gb": props.total_memory / 1024**3,
            "allocated_gb": allocated / 1024**3,
            "free_gb": (props.total_memory - reserved) / 1024**3,
        })
        total_memory += props.total_memory
    
    return {
        "count": len(devices),
        "devices": devices,
        "total_memory_gb": total_memory / 1024**3,
        "available": True,
    }


def build_device_map(
    strategy: str = "auto",
    max_memory: Optional[Dict[int, str]] = None,
    num_gpus: Optional[int] = None,
) -> Union[str, Dict[str, int]]:
    """根据策略构建设备映射。
    
    Args:
        strategy: 分布策略
            - "auto": 使用transformers自动分布
            - "balanced": 均衡分布到所有GPU
            - "sequential": 按顺序填充GPU
            - "single": 仅使用单个GPU
        max_memory: 每个GPU的最大显存限制，如 {0: "20GB", 1: "20GB"}
        num_gpus: 限制使用的GPU数量
        
    Returns:
        设备映射字符串或字典
    """
    if not torch.cuda.is_available():
        return "cpu"
    
    gpu_count = torch.cuda.device_count()
    if num_gpus is not None:
        gpu_count = min(gpu_count, num_gpus)
    
    if gpu_count == 0:
        return "cpu"
    
    if strategy == "auto":
        # 使用transformers自动分布
        if max_memory:
            return {"device_map": "auto", "max_memory": max_memory}
        return "auto"
    
    elif strategy == "balanced":
        # 均衡分布（由transformers处理）
        if max_memory:
            return {"device_map": "balanced", "max_memory": max_memory}
        return "balanced"
    
    elif strategy == "sequential":
        # 按顺序填充
        if max_memory:
            return {"device_map": "sequential", "max_memory": max_memory}
        return "sequential"
    
    elif strategy == "single":
        # 单GPU
        return {"": 0}
    
    else:
        logger.warning(f"Unknown strategy '{strategy}', falling back to 'auto'")
        return "auto"


# =============================================================================
# LoadedModel 数据类
# =============================================================================

@dataclass
class LoadedModel:
    """已加载模型的容器，包含模型、分词器和配置。
    
    Attributes:
        model: Hugging Face 模型实例
        tokenizer: 分词器实例
        config: 模型配置
    """
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: ModelConfig
    
    @property
    def num_layers(self) -> int:
        """模型层数。"""
        return self.config.n_layers
    
    @property
    def num_heads(self) -> int:
        """注意力头数。"""
        return self.config.n_heads
    
    @property
    def hidden_size(self) -> int:
        """隐藏层维度。"""
        return self.config.hidden_size
    
    @property
    def is_multi_gpu(self) -> bool:
        """检查模型是否分布在多个GPU上。"""
        if not hasattr(self.model, 'hf_device_map'):
            return False
        if not self.model.hf_device_map:
            return False
        
        # 收集唯一设备
        devices = set()
        for device in self.model.hf_device_map.values():
            if isinstance(device, int):
                devices.add(device)
            elif isinstance(device, str) and device.startswith("cuda"):
                devices.add(device)
        
        return len(devices) > 1
    
    def get_device(self) -> torch.device:
        """获取模型的输入设备。
        
        对于多GPU模型，返回嵌入层所在设备。
        """
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            device_map = self.model.hf_device_map
            
            # 查找嵌入层设备
            embedding_keys = ['embed_tokens', 'wte', 'word_embeddings', 
                            'embed_in', 'transformer.wte', 'model.embed_tokens']
            for key in embedding_keys:
                if key in device_map:
                    device = device_map[key]
                    if isinstance(device, int):
                        return torch.device(f"cuda:{device}")
                    return torch.device(device)
            
            # 查找包含 'embed' 的键
            for key, device in device_map.items():
                if 'embed' in key.lower():
                    if isinstance(device, int):
                        return torch.device(f"cuda:{device}")
                    return torch.device(device)
            
            # 使用第一个设备
            first_device = list(device_map.values())[0]
            if isinstance(first_device, int):
                return torch.device(f"cuda:{first_device}")
            return torch.device(first_device)
        
        # 标准方式
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_device_map_info(self) -> Optional[Dict[str, Any]]:
        """获取模型的设备分布信息。
        
        Returns:
            设备分布信息字典，如果是单GPU模型则返回None
        """
        if not hasattr(self.model, 'hf_device_map') or not self.model.hf_device_map:
            return None
        
        device_map = self.model.hf_device_map
        
        # 统计每个设备上的层数
        device_counts: Dict[Any, int] = {}
        for layer, device in device_map.items():
            device_counts[device] = device_counts.get(device, 0) + 1
        
        return {
            "device_map": dict(device_map),
            "device_layer_counts": device_counts,
            "num_devices": len(set(device_counts.keys())),
        }
    
    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """将文本编码为token IDs。"""
        return self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """将token IDs解码为文本。"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """生成文本。
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样
            **kwargs: 其他生成参数
            
        Returns:
            包含生成结果的字典
        """
        device = self.get_device()
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                **kwargs,
            )
        return {"generated_ids": outputs.sequences}


# =============================================================================
# 模型加载函数
# =============================================================================

def load_model(config: ModelConfig) -> LoadedModel:
    """从配置加载模型和分词器。
    
    支持多GPU分布加载，配置示例：
    ```yaml
    model:
      name: Qwen/Qwen2.5-7B-Instruct
      multi_gpu:
        enabled: true
        strategy: auto  # auto, balanced, sequential, single
        max_memory:
          0: "20GB"
          1: "20GB"
    ```
    
    Args:
        config: 模型配置
        
    Returns:
        LoadedModel 实例
        
    Raises:
        ModelError: 加载失败时抛出
    """
    logger.info(f"Loading model: {config.name}")
    
    # 强制使用 eager attention 以支持注意力提取
    if config.attn_implementation != "eager":
        logger.warning("Forcing attn_implementation='eager' for attention extraction.")
        config.attn_implementation = "eager"
    
    try:
        # =====================================================================
        # 加载分词器
        # =====================================================================
        tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # =====================================================================
        # 构建模型加载参数
        # =====================================================================
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code,
            "attn_implementation": config.attn_implementation,
        }
        
        # 数据类型
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if config.dtype in dtype_map:
            model_kwargs["torch_dtype"] = dtype_map[config.dtype]
        
        # 量化配置
        if config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_map.get(config.dtype, torch.float16),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        
        # =====================================================================
        # 多GPU配置
        # =====================================================================
        multi_gpu_config = getattr(config, 'multi_gpu', None)
        
        # 支持字典或 MultiGPUConfig 对象
        if multi_gpu_config is not None:
            # 获取 enabled 状态（支持字典和对象两种方式）
            if isinstance(multi_gpu_config, dict):
                multi_gpu_enabled = multi_gpu_config.get('enabled', False)
                strategy = multi_gpu_config.get('strategy', 'auto')
                max_memory = multi_gpu_config.get('max_memory', None)
                num_gpus = multi_gpu_config.get('num_gpus', None)
            else:
                # Pydantic 对象
                multi_gpu_enabled = getattr(multi_gpu_config, 'enabled', False)
                strategy = getattr(multi_gpu_config, 'strategy', 'auto')
                max_memory = getattr(multi_gpu_config, 'max_memory', None)
                num_gpus = getattr(multi_gpu_config, 'num_gpus', None)
        else:
            multi_gpu_enabled = False
            strategy = 'auto'
            max_memory = None
            num_gpus = None
        
        if multi_gpu_enabled:
            
            # 转换 max_memory 格式
            if max_memory:
                max_memory = {int(k): v for k, v in max_memory.items()}
            
            device_map_result = build_device_map(
                strategy=strategy,
                max_memory=max_memory,
                num_gpus=num_gpus,
            )
            
            if isinstance(device_map_result, dict):
                if "device_map" in device_map_result:
                    model_kwargs["device_map"] = device_map_result["device_map"]
                    if "max_memory" in device_map_result:
                        model_kwargs["max_memory"] = device_map_result["max_memory"]
                else:
                    model_kwargs["device_map"] = device_map_result
            else:
                model_kwargs["device_map"] = device_map_result
            
            # 记录GPU信息
            gpu_info = get_gpu_info()
            if gpu_info["available"]:
                logger.info(f"Multi-GPU enabled: {gpu_info['count']} GPUs available")
                for dev in gpu_info["devices"]:
                    logger.info(f"  GPU {dev['id']} ({dev['name']}): "
                              f"{dev['total_gb']:.1f}GB total, {dev['free_gb']:.1f}GB free")
        else:
            # 默认使用 auto device_map
            model_kwargs["device_map"] = "auto"
        
        # =====================================================================
        # 加载模型
        # =====================================================================
        logger.info(f"Loading with device_map: {model_kwargs.get('device_map', 'N/A')}")
        model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)
        model.eval()
        
        # =====================================================================
        # 更新配置
        # =====================================================================
        if hasattr(model.config, "num_hidden_layers"):
            config.n_layers = model.config.num_hidden_layers
        if hasattr(model.config, "num_attention_heads"):
            config.n_heads = model.config.num_attention_heads
        if hasattr(model.config, "hidden_size"):
            config.hidden_size = model.config.hidden_size
        
        loaded = LoadedModel(model=model, tokenizer=tokenizer, config=config)
        
        # 记录加载信息
        logger.info(f"Model loaded: {config.n_layers} layers, {config.n_heads} heads, "
                   f"hidden_size={config.hidden_size}")
        
        if loaded.is_multi_gpu:
            device_info = loaded.get_device_map_info()
            if device_info:
                logger.info(f"Model distributed across {device_info['num_devices']} devices")
        else:
            logger.info(f"Model on device: {loaded.get_device()}")
        
        return loaded
        
    except Exception as e:
        raise ModelError(f"Failed to load model: {e}", details={"model": config.name})


# =============================================================================
# 模型管理器
# =============================================================================

class ModelManager:
    """管理多个已加载模型，支持缓存。
    
    Usage:
        manager = get_model_manager()
        loaded = manager.get(config)
        manager.unload_all()
    """
    
    def __init__(self):
        self._cache: Dict[str, LoadedModel] = {}
    
    def get(self, config: ModelConfig) -> LoadedModel:
        """获取模型，如果已缓存则返回缓存版本。"""
        # 生成缓存键
        key = f"{config.name}:{config.dtype}:{config.load_in_4bit}:{config.load_in_8bit}"
        
        if key not in self._cache:
            self._cache[key] = load_model(config)
        
        return self._cache[key]
    
    def unload(self, config: ModelConfig) -> bool:
        """卸载指定模型。"""
        key = f"{config.name}:{config.dtype}:{config.load_in_4bit}:{config.load_in_8bit}"
        
        if key in self._cache:
            del self._cache[key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False
    
    def unload_all(self) -> None:
        """卸载所有模型并清理GPU缓存。"""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def list_loaded(self) -> List[str]:
        """列出所有已加载的模型。"""
        return list(self._cache.keys())


# 全局模型管理器实例
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """获取全局模型管理器。"""
    return _model_manager


def unload_all_models() -> None:
    """卸载所有模型。"""
    _model_manager.unload_all()