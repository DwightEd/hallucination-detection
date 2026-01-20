#!/usr/bin/env python3
"""
正确的截断检测脚本

检测逻辑：
1. 从 answers.json 读取原始 prompt 和 response 文本
2. 用 tokenizer 计算原始 token 长度
3. 从 features_individual/*.pt 读取实际存储的 response_len（截断后的）
4. 对比：如果 原始response_len > 存储的response_len，说明被截断了

用法：
    python scripts/reextract_truncated.py \
        --features-dir outputs/features/.../train \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --new-max-length 16384
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def sanitize_sample_id(sample_id: str) -> str:
    """清理样本ID用于文件名"""
    return str(sample_id).replace("/", "_").replace("\\", "_").replace(":", "_")


@dataclass
class TruncatedSample:
    sample_id: str
    original_prompt_len: int
    original_response_len: int
    stored_prompt_len: int
    stored_response_len: int
    
    @property
    def is_truncated(self) -> bool:
        return self.stored_response_len < self.original_response_len


def detect_truncated_samples(
    features_dir: Path,
    model_name: str,
    min_response_threshold: int = 5,
) -> List[TruncatedSample]:
    """
    检测被截断的样本
    
    对比原始文本的 token 长度与实际存储的长度
    """
    from transformers import AutoTokenizer
    
    logger.info(f"加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载 answers.json（包含原始文本）
    answers_file = features_dir / "answers.json"
    if not answers_file.exists():
        logger.error(f"answers.json 不存在: {answers_file}")
        return []
    
    with open(answers_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)
    logger.info(f"加载 {len(answers)} 个样本")
    
    # 检查 features_individual 目录
    individual_dir = features_dir / "features_individual"
    if not individual_dir.exists():
        logger.error(f"features_individual 目录不存在: {individual_dir}")
        return []
    
    truncated_samples = []
    checked = 0
    
    for answer in answers:
        sample_id = answer['id']
        prompt = answer.get('prompt', '')
        response = answer.get('response', '')
        
        # 计算原始 token 长度
        original_prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        original_response_len = len(tokenizer.encode(response, add_special_tokens=False))
        
        # 读取存储的长度
        safe_id = sanitize_sample_id(sample_id)
        pt_file = individual_dir / f"{safe_id}.pt"
        
        if not pt_file.exists():
            logger.warning(f"特征文件不存在: {pt_file}")
            continue
        
        try:
            data = torch.load(pt_file, map_location='cpu', weights_only=False)
            feat = data.get('features', data)
            stored_prompt_len = feat.get('prompt_len', 0) or 0
            stored_response_len = feat.get('response_len', 0) or 0
        except Exception as e:
            logger.warning(f"无法读取 {pt_file}: {e}")
            continue
        
        checked += 1
        if checked % 500 == 0:
            logger.info(f"  进度: {checked}/{len(answers)}")
        
        # 判断是否被截断或 response 太短
        is_truncated = stored_response_len < original_response_len
        is_too_short = stored_response_len <= min_response_threshold
        
        if is_truncated or is_too_short:
            truncated_samples.append(TruncatedSample(
                sample_id=sample_id,
                original_prompt_len=original_prompt_len,
                original_response_len=original_response_len,
                stored_prompt_len=stored_prompt_len,
                stored_response_len=stored_response_len,
            ))
    
    logger.info(f"检查完成: {checked} 个样本")
    logger.info(f"发现问题样本: {len(truncated_samples)}")
    
    return truncated_samples


def reextract_samples(
    features_dir: Path,
    truncated_samples: List[TruncatedSample],
    answers: List[Dict],
    model_name: str,
    new_max_length: int,
    device: str = "cuda",
) -> Tuple[int, int]:
    """重新提取被截断的样本"""
    from src.core import Sample, ModelConfig, FeaturesConfig, sanitize_sample_id
    from src.models import get_model, unload_all_models
    from src.features import create_extractor_from_requirements
    from src.utils import clear_gpu_memory
    from utils.feature_manager import create_feature_manager
    
    individual_dir = features_dir / "features_individual"
    consolidated_dir = features_dir / "features"
    
    # 构建待提取样本
    answers_by_id = {a['id']: a for a in answers}
    truncated_ids = {s.sample_id for s in truncated_samples}
    
    samples_to_extract = []
    for sid in truncated_ids:
        if sid not in answers_by_id:
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
    
    logger.info(f"重新提取 {len(samples_to_extract)} 个样本 (max_length={new_max_length})")
    
    # 加载模型
    logger.info(f"加载模型: {model_name}")
    model_config = ModelConfig(name=model_name, device=device)
    model = get_model(model_config)
    
    # 创建特征提取器
    feature_manager = create_feature_manager(
        methods=["hypergraph", "haloscope", "lapeigvals", "lookback_lens", "hsdmvaf"],
        allow_full_attention=True,
    )
    
    features_config = FeaturesConfig(
        mode="teacher_forcing",
        max_length=new_max_length,
        attention_enabled=True,
        attention_layers="all",
        attention_storage="diag",
        store_full_attention=True,
        hidden_states_enabled=True,
        hidden_states_layers="all",
        hidden_states_pooling="none",
        token_probs_enabled=True,
    )
    
    extractor = create_extractor_from_requirements(
        model=model,
        config=features_config,
        feature_requirements=feature_manager.get_combined_requirements().to_dict(),
        allow_full_attention=True,
    )
    
    # 提取特征
    individual_dir.mkdir(exist_ok=True)
    extracted_features = {}
    success_count = 0
    failed_count = 0
    
    for i, sample in enumerate(samples_to_extract):
        logger.info(f"  [{i+1}/{len(samples_to_extract)}] {sample.id}")
        
        try:
            features = extractor.extract(sample)
            
            # 构建特征字典
            features_dict = {
                "sample_id": features.sample_id,
                "prompt_len": features.prompt_len,
                "response_len": features.response_len,
                "label": features.label,
                "layers": features.layers,
            }
            
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
            
            # 保存到 features_individual/
            save_data = {
                "sample_id": sample.id,
                "features": features_dict,
                "metadata": {"model_name": model_name, "max_length": new_max_length, "reextracted": True}
            }
            
            safe_id = sanitize_sample_id(sample.id)
            pt_file = individual_dir / f"{safe_id}.pt"
            torch.save(save_data, pt_file)
            
            extracted_features[sample.id] = features_dict
            logger.info(f"    ✓ prompt={features.prompt_len}, response={features.response_len}")
            success_count += 1
            
            del features
            
        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
            failed_count += 1
        
        if (i + 1) % 10 == 0:
            clear_gpu_memory()
    
    # 清理模型
    unload_all_models()
    clear_gpu_memory()
    
    logger.info(f"提取完成: 成功 {success_count}, 失败 {failed_count}")
    
    # 更新合并特征文件
    if consolidated_dir.exists() and extracted_features:
        logger.info("更新合并特征文件...")
        update_consolidated_features(consolidated_dir, individual_dir, extracted_features)
    
    return success_count, failed_count


def update_consolidated_features(
    consolidated_dir: Path,
    individual_dir: Path,
    extracted_features: Dict[str, Dict],
):
    """更新 features/ 目录下的合并特征文件"""
    # 常规特征
    regular_features = [
        "attn_diags", "laplacian_diags", "attn_entropy",
        "token_probs", "token_entropy",
        "hallucination_labels", "hallucination_token_spans",
    ]
    
    for feat_key in regular_features:
        feat_file = consolidated_dir / f"{feat_key}.pt"
        if not feat_file.exists():
            continue
        
        samples_with_feat = [
            (sid, data[feat_key]) 
            for sid, data in extracted_features.items() 
            if feat_key in data and data[feat_key] is not None
        ]
        
        if not samples_with_feat:
            continue
        
        try:
            feat_data = torch.load(feat_file, map_location='cpu', weights_only=False)
            if not isinstance(feat_data, dict):
                feat_data = {}
        except:
            feat_data = {}
        
        for sid, value in samples_with_feat:
            feat_data[sid] = value
        
        torch.save(feat_data, feat_file)
        logger.info(f"  更新 {feat_key}.pt: {len(samples_with_feat)} 个样本")
    
    # 大特征索引
    large_features = ["hidden_states", "full_attentions"]
    
    for feat_key in large_features:
        index_file = consolidated_dir / f"{feat_key}_index.json"
        if not index_file.exists():
            continue
        
        samples_with_feat = [
            sid for sid, data in extracted_features.items() 
            if feat_key in data and data[feat_key] is not None
        ]
        
        if not samples_with_feat:
            continue
        
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            index = index_data.get('index', {})
        except:
            index = {}
        
        for sid in samples_with_feat:
            safe_id = sanitize_sample_id(sid)
            index[sid] = str(individual_dir / f"{safe_id}.pt")
        
        index_data['index'] = index
        index_data['sample_count'] = len(index)
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"  更新 {feat_key}_index.json: {len(samples_with_feat)} 个样本")


def main():
    parser = argparse.ArgumentParser(description="检测并重新提取被截断的样本")
    parser.add_argument('--features-dir', '-d', type=str, required=True,
                       help='特征目录路径')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='模型名称')
    parser.add_argument('--new-max-length', '-l', type=int, default=16384,
                       help='新的 max_length (默认: 16384)')
    parser.add_argument('--min-response-threshold', '-t', type=int, default=5,
                       help='response_len <= 此值也视为问题 (默认: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (默认: cuda)')
    parser.add_argument('--detect-only', action='store_true',
                       help='只检测，不重新提取')
    
    args = parser.parse_args()
    features_dir = Path(args.features_dir)
    
    if not features_dir.exists():
        logger.error(f"目录不存在: {features_dir}")
        return 1
    
    # 步骤 1: 检测
    logger.info("=" * 60)
    logger.info("步骤 1: 检测被截断或 response 过短的样本")
    logger.info("=" * 60)
    
    truncated = detect_truncated_samples(
        features_dir=features_dir,
        model_name=args.model,
        min_response_threshold=args.min_response_threshold,
    )
    
    if not truncated:
        logger.info("✅ 没有发现问题样本")
        return 0
    
    # 打印详情
    print("\n问题样本列表:")
    print("-" * 80)
    print(f"{'sample_id':<40} {'orig_resp':<10} {'stored_resp':<12} {'状态'}")
    print("-" * 80)
    for s in sorted(truncated, key=lambda x: x.stored_response_len)[:30]:
        status = "截断" if s.is_truncated else "原始就短"
        print(f"{s.sample_id:<40} {s.original_response_len:<10} {s.stored_response_len:<12} {status}")
    if len(truncated) > 30:
        print(f"... 还有 {len(truncated) - 30} 个")
    print("-" * 80)
    
    # 保存列表
    output_file = features_dir / "truncated_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'count': len(truncated),
            'samples': [
                {
                    'sample_id': s.sample_id,
                    'original_prompt_len': s.original_prompt_len,
                    'original_response_len': s.original_response_len,
                    'stored_prompt_len': s.stored_prompt_len,
                    'stored_response_len': s.stored_response_len,
                    'is_truncated': s.is_truncated,
                }
                for s in truncated
            ]
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"问题样本列表已保存到: {output_file}")
    
    if args.detect_only:
        logger.info("检测完成 (--detect-only 模式)")
        return 0
    
    # 步骤 2: 重新提取
    logger.info("")
    logger.info("=" * 60)
    logger.info("步骤 2: 重新提取问题样本")
    logger.info("=" * 60)
    
    # 加载 answers
    answers_file = features_dir / "answers.json"
    with open(answers_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)
    
    success, failed = reextract_samples(
        features_dir=features_dir,
        truncated_samples=truncated,
        answers=answers,
        model_name=args.model,
        new_max_length=args.new_max_length,
        device=args.device,
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ 完成: 成功 {success}, 失败 {failed}")
    logger.info("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())