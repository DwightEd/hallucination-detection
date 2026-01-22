#!/usr/bin/env python3
"""将项目特征转换为原版 LapEigvals 项目格式。

放置位置: scripts/convert_to_lapeigvals.py

用法:
    # 转换单个特征文件
    python scripts/convert_to_lapeigvals.py \
        --input data/features/ragtruth/llama2-7b/train/features.pt \
        --output data/lapeigvals_format/train.pkl
    
    # 转换整个目录
    python scripts/convert_to_lapeigvals.py \
        --input-dir data/features/ragtruth/llama2-7b/train/individual \
        --output data/lapeigvals_format/train.pkl
    
    # 使用 Hydra 配置
    python scripts/convert_to_lapeigvals.py \
        dataset=ragtruth model=llama2-7b split=train

原版 LapEigvals 项目: https://github.com/graphml-lab-pwr/lapeigvals
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json

import numpy as np
import torch

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据格式定义
# =============================================================================

"""
原版 LapEigvals 期望的数据格式:

1. 输入数据 (用于特征提取):
   {
       "attention": np.ndarray,  # [n_layers, n_heads, seq_len, seq_len]
       "input_ids": np.ndarray,  # [seq_len] (可选)
       "prompt_length": int,
       "response_length": int,
   }

2. 预计算特征 (用于直接训练):
   {
       "eigenvalues": np.ndarray,  # [n_layers * n_heads * top_k]
       "label": int,
       "sample_id": str,
   }

3. 批量数据格式:
   List[Dict] 或 pickle 文件
"""


# =============================================================================
# 转换函数
# =============================================================================

def load_project_features(path: Union[str, Path]) -> Dict[str, Any]:
    """加载项目格式的特征文件。
    
    支持:
    - .pt / .pth (PyTorch)
    - .pkl / .pickle (Pickle)
    - .npz (NumPy)
    """
    path = Path(path)
    
    if path.suffix in ['.pt', '.pth']:
        data = torch.load(path, map_location='cpu')
    elif path.suffix in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif path.suffix == '.npz':
        data = dict(np.load(path, allow_pickle=True))
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    return data


def convert_single_sample(
    features: Dict[str, Any],
    include_raw_attention: bool = True,
    compute_eigenvalues: bool = True,
    top_k: int = 100,
) -> Dict[str, Any]:
    """将单个样本从项目格式转换为原版 LapEigvals 格式。
    
    Args:
        features: 项目格式的特征字典
        include_raw_attention: 是否包含原始注意力矩阵
        compute_eigenvalues: 是否预计算特征值
        top_k: 每个 (layer, head) 的 top-k 特征值
        
    Returns:
        原版 LapEigvals 格式的字典
    """
    result = {
        "sample_id": features.get("sample_id", "unknown"),
        "label": features.get("label", 0),
        "prompt_length": features.get("prompt_len", 0),
        "response_length": features.get("response_len", 0),
    }
    
    # 获取注意力矩阵
    full_attention = None
    for key in ['full_attention', 'full_attentions', 'attention']:
        if key in features and features[key] is not None:
            full_attention = features[key]
            break
    
    if full_attention is not None:
        # 转换为 numpy
        if isinstance(full_attention, torch.Tensor):
            full_attention = full_attention.detach().cpu().numpy()
        
        if include_raw_attention:
            result["attention"] = full_attention.astype(np.float32)
        
        # 计算特征值
        if compute_eigenvalues:
            eigenvalues = compute_laplacian_eigenvalues(
                full_attention,
                prompt_len=result["prompt_length"],
                response_len=result["response_length"],
                top_k=top_k,
            )
            result["eigenvalues"] = eigenvalues
    
    # 如果已有预计算的 laplacian_diags
    elif "laplacian_diags" in features and features["laplacian_diags"] is not None:
        laplacian_diags = features["laplacian_diags"]
        if isinstance(laplacian_diags, torch.Tensor):
            laplacian_diags = laplacian_diags.detach().cpu().numpy()
        
        if compute_eigenvalues:
            eigenvalues = extract_eigenvalues_from_diag(
                laplacian_diags,
                prompt_len=result["prompt_length"],
                response_len=result["response_length"],
                top_k=top_k,
            )
            result["eigenvalues"] = eigenvalues
        
        result["laplacian_diags"] = laplacian_diags.astype(np.float32)
    
    return result


def compute_laplacian_eigenvalues(
    attention: np.ndarray,
    prompt_len: int,
    response_len: int,
    top_k: int = 100,
    response_only: bool = True,
) -> np.ndarray:
    """从完整注意力矩阵计算 Laplacian 特征值。
    
    严格按照 EMNLP 2025 论文公式:
    - d_ii = Σ_{u>i} a_ui / (T - i)  [式(2)]
    - λ_i = d_ii - a_ii              [式(3)]
    
    Args:
        attention: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        top_k: 每个 (layer, head) 的 top-k 特征值
        response_only: 是否只使用 response 部分
        
    Returns:
        特征向量 [n_layers * n_heads * top_k]
    """
    n_layers, n_heads, seq_len, _ = attention.shape
    
    # 确定范围
    if response_only and prompt_len > 0 and response_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len)
    else:
        start_idx = 0
        end_idx = seq_len
    
    actual_len = end_idx - start_idx
    if actual_len <= 0:
        return np.zeros(n_layers * n_heads * top_k, dtype=np.float32)
    
    actual_k = min(top_k, actual_len)
    all_eigenvalues = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # 获取 response 部分
            A = attention[layer, head, start_idx:end_idx, start_idx:end_idx]
            T = A.shape[0]
            
            # 计算 Laplacian 对角线 (即特征值)
            eigenvalues = np.zeros(T, dtype=np.float64)
            for i in range(T):
                # 出度: 后续 token 对当前 token 的平均注意力
                subsequent_sum = A[i+1:, i].sum() if i < T - 1 else 0.0
                denominator = T - i
                d_ii = subsequent_sum / denominator if denominator > 0 else 0.0
                
                # 特征值 = 出度 - 自注意力
                eigenvalues[i] = d_ii - A[i, i]
            
            # 排序取 top-k
            sorted_eigvals = np.sort(eigenvalues)[::-1]  # 降序
            top_eigvals = sorted_eigvals[:actual_k]
            
            # Padding
            if len(top_eigvals) < top_k:
                top_eigvals = np.pad(
                    top_eigvals, 
                    (0, top_k - len(top_eigvals)), 
                    mode='constant', 
                    constant_values=0
                )
            
            all_eigenvalues.append(top_eigvals)
    
    return np.concatenate(all_eigenvalues).astype(np.float32)


def extract_eigenvalues_from_diag(
    laplacian_diags: np.ndarray,
    prompt_len: int,
    response_len: int,
    top_k: int = 100,
    response_only: bool = True,
) -> np.ndarray:
    """从预计算的 Laplacian 对角线提取特征值。
    
    Args:
        laplacian_diags: [n_layers, n_heads, seq_len]
        prompt_len: Prompt 长度
        response_len: Response 长度
        top_k: 每个 (layer, head) 的 top-k 特征值
        response_only: 是否只使用 response 部分
        
    Returns:
        特征向量 [n_layers * n_heads * top_k]
    """
    n_layers, n_heads, seq_len = laplacian_diags.shape
    
    # 确定范围
    if response_only and prompt_len > 0 and response_len > 0:
        start_idx = prompt_len
        end_idx = min(prompt_len + response_len, seq_len)
    else:
        start_idx = 0
        end_idx = seq_len
    
    actual_len = end_idx - start_idx
    if actual_len <= 0:
        return np.zeros(n_layers * n_heads * top_k, dtype=np.float32)
    
    actual_k = min(top_k, actual_len)
    all_eigenvalues = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            eigvals = laplacian_diags[layer, head, start_idx:end_idx]
            
            # 排序取 top-k
            sorted_eigvals = np.sort(eigvals)[::-1]
            top_eigvals = sorted_eigvals[:actual_k]
            
            # Padding
            if len(top_eigvals) < top_k:
                top_eigvals = np.pad(
                    top_eigvals,
                    (0, top_k - len(top_eigvals)),
                    mode='constant',
                    constant_values=0
                )
            
            all_eigenvalues.append(top_eigvals)
    
    return np.concatenate(all_eigenvalues).astype(np.float32)


# =============================================================================
# 批量转换
# =============================================================================

def convert_feature_file(
    input_path: Path,
    top_k: int = 100,
    include_raw_attention: bool = False,
) -> List[Dict[str, Any]]:
    """转换单个特征文件 (可能包含多个样本)。"""
    data = load_project_features(input_path)
    
    # 判断是单样本还是多样本
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        # 检查是否是批量格式
        if "samples" in data:
            samples = data["samples"]
        elif "features" in data:
            samples = data["features"]
        else:
            # 单样本
            samples = [data]
    else:
        raise ValueError(f"无法解析数据格式: {type(data)}")
    
    converted = []
    for sample in samples:
        try:
            converted_sample = convert_single_sample(
                sample,
                include_raw_attention=include_raw_attention,
                compute_eigenvalues=True,
                top_k=top_k,
            )
            converted.append(converted_sample)
        except Exception as e:
            logger.warning(f"转换样本失败: {e}")
            continue
    
    return converted


def convert_directory(
    input_dir: Path,
    top_k: int = 100,
    include_raw_attention: bool = False,
    file_pattern: str = "*.pt",
) -> List[Dict[str, Any]]:
    """转换目录中的所有特征文件。"""
    input_dir = Path(input_dir)
    
    all_converted = []
    files = list(input_dir.glob(file_pattern))
    
    if not files:
        # 尝试其他格式
        for pattern in ["*.pt", "*.pth", "*.pkl", "*.npz"]:
            files = list(input_dir.glob(pattern))
            if files:
                break
    
    logger.info(f"找到 {len(files)} 个文件")
    
    for i, file_path in enumerate(files):
        logger.info(f"[{i+1}/{len(files)}] 处理 {file_path.name}")
        try:
            converted = convert_feature_file(
                file_path,
                top_k=top_k,
                include_raw_attention=include_raw_attention,
            )
            all_converted.extend(converted)
        except Exception as e:
            logger.warning(f"处理文件失败 {file_path}: {e}")
            continue
    
    return all_converted


def save_converted_data(
    data: List[Dict[str, Any]],
    output_path: Path,
    format: str = "pickle",
) -> None:
    """保存转换后的数据。
    
    Args:
        data: 转换后的数据列表
        output_path: 输出路径
        format: 输出格式 ("pickle", "numpy", "json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
    elif format == "numpy":
        # 分离特征和标签
        eigenvalues = np.array([d["eigenvalues"] for d in data])
        labels = np.array([d["label"] for d in data])
        sample_ids = [d["sample_id"] for d in data]
        
        np.savez(
            output_path,
            eigenvalues=eigenvalues,
            labels=labels,
            sample_ids=sample_ids,
        )
    
    elif format == "json":
        # JSON 不支持 numpy array，需要转换
        json_data = []
        for d in data:
            json_d = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in d.items()}
            json_data.append(json_d)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"已保存 {len(data)} 个样本到 {output_path}")


# =============================================================================
# 生成原版项目可直接使用的训练数据
# =============================================================================

def export_for_sklearn(
    data: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """导出为 sklearn 可直接使用的格式。
    
    生成:
    - X_train.npy: 特征矩阵 [n_samples, n_features]
    - y_train.npy: 标签向量 [n_samples]
    - sample_ids.json: 样本 ID 列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X = np.array([d["eigenvalues"] for d in data])
    y = np.array([d["label"] for d in data])
    sample_ids = [d["sample_id"] for d in data]
    
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    
    with open(output_dir / "sample_ids.json", 'w') as f:
        json.dump(sample_ids, f)
    
    logger.info(f"已导出 sklearn 格式:")
    logger.info(f"  X: {X.shape}")
    logger.info(f"  y: {y.shape}")
    logger.info(f"  路径: {output_dir}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="将项目特征转换为原版 LapEigvals 格式"
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        help="输入特征文件路径"
    )
    input_group.add_argument(
        "--input-dir", "-d",
        type=Path,
        help="输入特征目录路径"
    )
    
    # 输出选项
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="输出路径"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["pickle", "numpy", "json", "sklearn"],
        default="pickle",
        help="输出格式 (默认: pickle)"
    )
    
    # 参数选项
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="每个 (layer, head) 的 top-k 特征值 (默认: 100)"
    )
    parser.add_argument(
        "--include-attention",
        action="store_true",
        help="是否包含原始注意力矩阵 (会增大文件体积)"
    )
    parser.add_argument(
        "--file-pattern",
        default="*.pt",
        help="文件匹配模式 (默认: *.pt)"
    )
    
    args = parser.parse_args()
    
    # 转换
    if args.input:
        logger.info(f"转换文件: {args.input}")
        converted = convert_feature_file(
            args.input,
            top_k=args.top_k,
            include_raw_attention=args.include_attention,
        )
    else:
        logger.info(f"转换目录: {args.input_dir}")
        converted = convert_directory(
            args.input_dir,
            top_k=args.top_k,
            include_raw_attention=args.include_attention,
            file_pattern=args.file_pattern,
        )
    
    logger.info(f"成功转换 {len(converted)} 个样本")
    
    # 保存
    if args.format == "sklearn":
        export_for_sklearn(converted, args.output)
    else:
        save_converted_data(converted, args.output, format=args.format)


if __name__ == "__main__":
    main()