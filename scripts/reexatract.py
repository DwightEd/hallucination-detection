#!/usr/bin/env python3
"""
标记并重新提取被截断样本的基础特征和衍生特征。

问题：某些样本因为 prompt_len >= max_length 导致 response 被完全截断，
需要用更大的 max_length 重新提取这些样本的特征。

流程：
1. 扫描所有样本，找出 prompt_len >= seq_len 的问题样本
2. 将问题样本ID保存到文件
3. 用更大的 max_length 重新提取这些样本的基础特征
4. 计算衍生特征
5. 更新合并的特征文件

用法:
    # 步骤1: 标记问题样本
    python scripts/reextract_truncated_samples.py mark \
        --features-dir data/ragtruth/features/train \
        --output truncated_samples.json

    # 步骤2: 重新提取这些样本的特征 (需要模型)
    python scripts/reextract_truncated_samples.py extract \
        --features-dir data/ragtruth/features/train \
        --samples-file truncated_samples.json \
        --new-max-length 8192 \
        --model Qwen/Qwen2.5-7B-Instruct

    # 一步完成
    python scripts/reextract_truncated_samples.py all \
        --features-dir data/ragtruth/features/train \
        --new-max-length 8192 \
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
import sys
import os

import torch

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def sanitize_sample_id(sample_id: str) -> str:
    """清理样本ID，使其可以安全用作文件名"""
    return str(sample_id).replace("/", "_").replace("\\", "_").replace(":", "_")


@dataclass
class TruncatedSampleInfo:
    """被截断样本的信息"""
    sample_id: str
    original_prompt_len: int  # 原始 prompt token 数
    original_response_len: int  # 原始 response token 数
    original_total_len: int  # 原始总长度
    current_seq_len: int  # 当前（截断后的）序列长度
    issue_type: str  # "prompt_truncated", "response_truncated", "response_too_short"
    
    @property
    def min_required_length(self) -> int:
        """计算需要的最小 max_length"""
        return self.original_total_len + 10  # 加一点余量


def load_individual_feature(features_dir: Path, sample_id: str) -> Optional[Dict]:
    """加载 individual .pt 文件"""
    individual_dir = features_dir / "features_individual"
    if not individual_dir.exists():
        return None
    
    safe_id = sanitize_sample_id(sample_id)
    pt_file = individual_dir / f"{safe_id}.pt"
    
    if not pt_file.exists():
        return None
    
    try:
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        logger.warning(f"Failed to load {pt_file}: {e}")
        return None


def get_attention_seq_len(features_data: Dict) -> Optional[int]:
    """从特征数据中获取注意力矩阵的序列长度"""
    feat = features_data.get('features', {})
    
    for key in ['full_attentions', 'attn_diags', 'laplacian_diags', 'attn_entropy']:
        tensor = feat.get(key)
        if tensor is not None and isinstance(tensor, torch.Tensor):
            return tensor.shape[-1]
    
    return None


def load_answers_json(features_dir: Path) -> List[Dict]:
    """加载 answers.json"""
    answers_path = features_dir / "answers.json"
    if not answers_path.exists():
        raise FileNotFoundError(f"answers.json not found: {answers_path}")
    
    with open(answers_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def mark_truncated_samples(features_dir: Path, verbose: bool = False) -> List[TruncatedSampleInfo]:
    """
    扫描所有样本，标记被截断的问题样本。
    
    问题样本包括：
    1. prompt_len >= seq_len (response 完全被截断)
    2. response_len <= 1 (response 几乎被完全截断)
    """
    logger.info(f"Scanning samples in {features_dir}...")
    
    answers = load_answers_json(features_dir)
    truncated_samples = []
    
    for i, answer in enumerate(answers):
        if (i + 1) % 1000 == 0:
            logger.info(f"Progress: {i + 1}/{len(answers)}")
        
        sample_id = answer['id']
        
        # 从 individual 文件获取实际数据
        features_data = load_individual_feature(features_dir, sample_id)
        
        if features_data is None:
            if verbose:
                logger.warning(f"Sample {sample_id}: individual file not found")
            continue
        
        feat = features_data.get('features', {})
        
        # 获取实际的 prompt_len 和 response_len（特征提取时记录的）
        stored_prompt_len = feat.get('prompt_len', 0)
        stored_response_len = feat.get('response_len', 0)
        
        # 获取实际序列长度
        seq_len = get_attention_seq_len(features_data)
        
        if seq_len is None:
            if verbose:
                logger.warning(f"Sample {sample_id}: no attention data")
            continue
        
        # 从 answers.json 获取原始长度（可能未截断）
        original_prompt_len = answer.get('prompt_len', stored_prompt_len)
        original_response_len = answer.get('response_len', stored_response_len)
        original_total_len = original_prompt_len + original_response_len
        
        # 判断问题类型
        issue_type = None
        
        if stored_prompt_len >= seq_len:
            # response_idx >= seq_len，response 完全被截断
            issue_type = "prompt_truncated"
        elif stored_response_len <= 1:
            # response 几乎被完全截断
            issue_type = "response_too_short"
        elif original_total_len > seq_len and original_response_len > stored_response_len:
            # response 被部分截断
            issue_type = "response_truncated"
        
        if issue_type:
            info = TruncatedSampleInfo(
                sample_id=sample_id,
                original_prompt_len=original_prompt_len,
                original_response_len=original_response_len,
                original_total_len=original_total_len,
                current_seq_len=seq_len,
                issue_type=issue_type,
            )
            truncated_samples.append(info)
            
            if verbose:
                logger.info(
                    f"Found truncated sample: {sample_id} "
                    f"(type={issue_type}, original_total={original_total_len}, seq_len={seq_len})"
                )
    
    return truncated_samples


def save_truncated_samples_list(
    samples: List[TruncatedSampleInfo],
    output_path: Path
) -> None:
    """保存问题样本列表"""
    data = {
        "total_truncated": len(samples),
        "by_type": {},
        "max_required_length": max((s.min_required_length for s in samples), default=0),
        "samples": [asdict(s) for s in samples]
    }
    
    # 按类型统计
    for s in samples:
        data["by_type"][s.issue_type] = data["by_type"].get(s.issue_type, 0) + 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(samples)} truncated samples to {output_path}")
    logger.info(f"By type: {data['by_type']}")
    logger.info(f"Max required length: {data['max_required_length']}")


def load_truncated_samples_list(input_path: Path) -> Tuple[List[str], int]:
    """加载问题样本列表，返回 (sample_ids, max_required_length)"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sample_ids = [s['sample_id'] for s in data['samples']]
    max_required_length = data.get('max_required_length', 8192)
    
    return sample_ids, max_required_length


def reextract_samples(
    features_dir: Path,
    sample_ids: List[str],
    new_max_length: int,
    model_name: str,
    dataset_name: str = "ragtruth",
    split_name: str = "train",
    batch_size: int = 1,
    device: str = "cuda",
) -> Dict[str, int]:
    """
    重新提取指定样本的特征。
    
    Returns:
        统计信息 {"success": n, "failed": n}
    """
    # 延迟导入，避免在只标记时加载大量依赖
    from omegaconf import OmegaConf
    
    from src.core import Sample, ModelConfig, FeaturesConfig, sanitize_sample_id
    from src.models import get_model, unload_all_models
    from src.features import create_extractor_from_requirements
    from src.utils import clear_gpu_memory
    from utils.feature_manager import create_feature_manager
    
    logger.info(f"Re-extracting {len(sample_ids)} samples with max_length={new_max_length}")
    
    # 1. 加载原始样本数据
    answers = load_answers_json(features_dir)
    answers_by_id = {a['id']: a for a in answers}
    
    samples_to_extract = []
    for sid in sample_ids:
        if sid not in answers_by_id:
            logger.warning(f"Sample {sid} not found in answers.json")
            continue
        
        ans = answers_by_id[sid]
        samples_to_extract.append(Sample(
            id=sid,
            prompt=ans.get('prompt', ''),
            response=ans.get('response', ''),
            label=ans.get('label', 0),
            metadata={
                'hallucination_spans': ans.get('labels', []),
                'source_model': ans.get('source_model'),
            }
        ))
    
    if not samples_to_extract:
        logger.error("No samples to extract!")
        return {"success": 0, "failed": 0}
    
    logger.info(f"Loaded {len(samples_to_extract)} samples for re-extraction")
    
    # 2. 创建模型和提取器
    model_config = ModelConfig(
        name=model_name,
        device=device,
    )
    
    logger.info(f"Loading model: {model_name}")
    model = get_model(model_config)
    
    # 创建特征管理器（hypergraph 需要 full_attention）
    feature_manager = create_feature_manager(
        methods=["hypergraph"],
        allow_full_attention=True,
    )
    
    features_config = FeaturesConfig(
        mode="teacher_forcing",
        max_length=new_max_length,
        attention_enabled=True,
        attention_layers="all",
        hidden_states_enabled=True,
        hidden_states_layers="-4,-3,-2,-1",
        token_probs_enabled=True,
    )
    
    extractor = create_extractor_from_requirements(
        model=model,
        config=features_config,
        feature_requirements=feature_manager.get_combined_requirements().to_dict(),
        allow_full_attention=True,
    )
    
    # 3. 提取特征
    individual_dir = features_dir / "features_individual"
    individual_dir.mkdir(exist_ok=True)
    
    stats = {"success": 0, "failed": 0}
    
    for i, sample in enumerate(samples_to_extract):
        logger.info(f"Processing [{i+1}/{len(samples_to_extract)}]: {sample.id}")
        
        try:
            # 提取特征
            features = extractor.extract(sample)
            
            # 验证
            if features.prompt_len >= new_max_length:
                logger.warning(
                    f"Sample {sample.id}: prompt_len ({features.prompt_len}) >= max_length ({new_max_length}). "
                    f"Need even larger max_length!"
                )
            
            # 构建保存数据
            features_dict = {
                "sample_id": features.sample_id,
                "prompt_len": features.prompt_len,
                "response_len": features.response_len,
                "label": features.label,
                "layers": features.layers,
            }
            
            # 添加各类特征
            for attr in ['attn_diags', 'attn_row_sums', 'laplacian_diags', 'attn_entropy',
                         'hidden_states', 'token_probs', 'token_entropy']:
                value = getattr(features, attr, None)
                if value is not None:
                    features_dict[attr] = value
            
            if features.full_attention is not None:
                features_dict['full_attentions'] = features.full_attention
            
            if features.hallucination_labels is not None:
                features_dict['hallucination_labels'] = features.hallucination_labels
            if features.hallucination_token_spans is not None:
                features_dict['hallucination_token_spans'] = features.hallucination_token_spans
            
            # 保存到 individual 文件
            save_data = {
                "features": features_dict,
                "metadata": {
                    "model_name": model_name,
                    "max_length": new_max_length,
                    "reextracted": True,
                }
            }
            
            safe_id = sanitize_sample_id(sample.id)
            pt_file = individual_dir / f"{safe_id}.pt"
            torch.save(save_data, pt_file)
            
            # 更新 answers.json 中的 prompt_len 和 response_len
            if sample.id in answers_by_id:
                answers_by_id[sample.id]['prompt_len'] = features.prompt_len
                answers_by_id[sample.id]['response_len'] = features.response_len
            
            stats["success"] += 1
            logger.info(
                f"  ✓ Extracted: prompt_len={features.prompt_len}, "
                f"response_len={features.response_len}, "
                f"total={features.prompt_len + features.response_len}"
            )
            
            del features
            
        except Exception as e:
            logger.error(f"  ✗ Failed to extract {sample.id}: {e}")
            stats["failed"] += 1
        
        # 定期清理内存
        if (i + 1) % 10 == 0:
            clear_gpu_memory()
    
    # 4. 保存更新后的 answers.json
    answers_path = features_dir / "answers.json"
    updated_answers = [answers_by_id[a['id']] for a in answers if a['id'] in answers_by_id]
    
    with open(answers_path, 'w', encoding='utf-8') as f:
        json.dump(updated_answers, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Updated answers.json with new prompt_len/response_len values")
    
    # 5. 清理
    unload_all_models()
    clear_gpu_memory()
    
    return stats


def update_consolidated_features(
    features_dir: Path,
    sample_ids: List[str],
) -> None:
    """更新合并的特征文件"""
    logger.info("Updating consolidated features...")
    
    consolidated_dir = features_dir / "features"
    if not consolidated_dir.exists():
        consolidated_dir.mkdir(exist_ok=True)
    
    # 需要更新的特征类型
    feature_mapping = {
        "attn_diags": "attn_diags.pt",
        "laplacian_diags": "laplacian_diags.pt",
        "attn_entropy": "attn_entropy.pt",
        "attn_row_sums": "attn_row_sums.pt",
        "token_probs": "token_probs.pt",
        "token_entropy": "token_entropy.pt",
        "hallucination_labels": "hallucination_labels.pt",
    }
    
    # 加载需要更新的样本数据
    updated_features = {}
    for sid in sample_ids:
        data = load_individual_feature(features_dir, sid)
        if data:
            updated_features[sid] = data.get('features', {})
    
    if not updated_features:
        logger.warning("No features to update")
        return
    
    # 更新每个特征文件
    for feature_key, filename in feature_mapping.items():
        filepath = consolidated_dir / filename
        
        # 加载现有数据或创建新的
        if filepath.exists():
            try:
                feature_data = torch.load(filepath, weights_only=False)
                if not isinstance(feature_data, dict):
                    feature_data = {}
            except:
                feature_data = {}
        else:
            feature_data = {}
        
        # 更新
        updated_count = 0
        for sid, feat in updated_features.items():
            if feature_key in feat:
                feature_data[sid] = feat[feature_key]
                updated_count += 1
        
        # 保存
        if updated_count > 0:
            torch.save(feature_data, filepath)
            logger.info(f"Updated {filename}: {updated_count} samples")


def cmd_mark(args):
    """标记命令"""
    features_dir = Path(args.features_dir)
    
    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        return 1
    
    # 标记问题样本
    truncated_samples = mark_truncated_samples(features_dir, verbose=args.verbose)
    
    if not truncated_samples:
        logger.info("✓ No truncated samples found!")
        return 0
    
    # 保存列表
    output_path = Path(args.output) if args.output else features_dir / "truncated_samples.json"
    save_truncated_samples_list(truncated_samples, output_path)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("截断样本摘要")
    print("=" * 60)
    print(f"发现问题样本: {len(truncated_samples)}")
    print(f"建议的最小 max_length: {max(s.min_required_length for s in truncated_samples)}")
    print(f"问题样本列表已保存到: {output_path}")
    print("=" * 60)
    
    return 0


def cmd_extract(args):
    """重新提取命令"""
    features_dir = Path(args.features_dir)
    samples_file = Path(args.samples_file)
    
    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        return 1
    
    if not samples_file.exists():
        logger.error(f"Samples file not found: {samples_file}")
        return 1
    
    # 加载问题样本列表
    sample_ids, suggested_max_length = load_truncated_samples_list(samples_file)
    
    new_max_length = args.new_max_length or suggested_max_length
    logger.info(f"Will use max_length={new_max_length}")
    
    if new_max_length < suggested_max_length:
        logger.warning(
            f"Specified max_length ({new_max_length}) < suggested ({suggested_max_length}). "
            f"Some samples may still be truncated!"
        )
    
    # 重新提取
    stats = reextract_samples(
        features_dir=features_dir,
        sample_ids=sample_ids,
        new_max_length=new_max_length,
        model_name=args.model,
        device=args.device,
    )
    
    logger.info(f"Extraction complete: {stats['success']} success, {stats['failed']} failed")
    
    # 更新合并的特征文件
    if args.update_consolidated:
        update_consolidated_features(features_dir, sample_ids)
    
    return 0 if stats['failed'] == 0 else 1


def cmd_all(args):
    """一步完成：标记 + 提取"""
    features_dir = Path(args.features_dir)
    
    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        return 1
    
    # 1. 标记问题样本
    logger.info("Step 1: Marking truncated samples...")
    truncated_samples = mark_truncated_samples(features_dir, verbose=args.verbose)
    
    if not truncated_samples:
        logger.info("✓ No truncated samples found! Nothing to do.")
        return 0
    
    # 保存列表
    samples_file = features_dir / "truncated_samples.json"
    save_truncated_samples_list(truncated_samples, samples_file)
    
    # 2. 确定 max_length
    suggested_max_length = max(s.min_required_length for s in truncated_samples)
    new_max_length = args.new_max_length or suggested_max_length
    
    if new_max_length < suggested_max_length:
        logger.warning(
            f"Specified max_length ({new_max_length}) < suggested ({suggested_max_length}). "
            f"Some samples may still be truncated!"
        )
    
    # 3. 重新提取
    logger.info(f"\nStep 2: Re-extracting {len(truncated_samples)} samples with max_length={new_max_length}...")
    
    sample_ids = [s.sample_id for s in truncated_samples]
    
    stats = reextract_samples(
        features_dir=features_dir,
        sample_ids=sample_ids,
        new_max_length=new_max_length,
        model_name=args.model,
        device=args.device,
    )
    
    logger.info(f"Extraction complete: {stats['success']} success, {stats['failed']} failed")
    
    # 4. 更新合并的特征文件
    if args.update_consolidated:
        logger.info("\nStep 3: Updating consolidated features...")
        update_consolidated_features(features_dir, sample_ids)
    
    # 打印最终摘要
    print("\n" + "=" * 60)
    print("重新提取完成")
    print("=" * 60)
    print(f"问题样本数: {len(truncated_samples)}")
    print(f"成功提取: {stats['success']}")
    print(f"失败: {stats['failed']}")
    print(f"使用的 max_length: {new_max_length}")
    print("=" * 60)
    
    return 0 if stats['failed'] == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="标记并重新提取被截断样本的特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # mark 子命令
    mark_parser = subparsers.add_parser('mark', help='标记被截断的样本')
    mark_parser.add_argument('--features-dir', '-d', type=str, required=True,
                            help='特征目录路径')
    mark_parser.add_argument('--output', '-o', type=str, default=None,
                            help='输出文件路径 (默认: features_dir/truncated_samples.json)')
    mark_parser.add_argument('--verbose', '-v', action='store_true',
                            help='详细输出')
    
    # extract 子命令
    extract_parser = subparsers.add_parser('extract', help='重新提取指定样本的特征')
    extract_parser.add_argument('--features-dir', '-d', type=str, required=True,
                               help='特征目录路径')
    extract_parser.add_argument('--samples-file', '-s', type=str, required=True,
                               help='问题样本列表文件')
    extract_parser.add_argument('--new-max-length', '-l', type=int, default=None,
                               help='新的 max_length (默认使用建议值)')
    extract_parser.add_argument('--model', '-m', type=str, required=True,
                               help='模型名称或路径')
    extract_parser.add_argument('--device', type=str, default='cuda',
                               help='设备 (默认: cuda)')
    extract_parser.add_argument('--update-consolidated', action='store_true',
                               help='同时更新合并的特征文件')
    
    # all 子命令
    all_parser = subparsers.add_parser('all', help='一步完成：标记 + 提取')
    all_parser.add_argument('--features-dir', '-d', type=str, required=True,
                           help='特征目录路径')
    all_parser.add_argument('--new-max-length', '-l', type=int, default=None,
                           help='新的 max_length (默认使用建议值)')
    all_parser.add_argument('--model', '-m', type=str, required=True,
                           help='模型名称或路径')
    all_parser.add_argument('--device', type=str, default='cuda',
                           help='设备 (默认: cuda)')
    all_parser.add_argument('--update-consolidated', action='store_true',
                           help='同时更新合并的特征文件')
    all_parser.add_argument('--verbose', '-v', action='store_true',
                           help='详细输出')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'mark':
        return cmd_mark(args)
    elif args.command == 'extract':
        return cmd_extract(args)
    elif args.command == 'all':
        return cmd_all(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())