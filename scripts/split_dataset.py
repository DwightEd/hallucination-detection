#!/usr/bin/env python3
"""
数据集分割脚本

功能：
1. 从配置文件读取参数（不使用argparse）
2. 加载原始数据集
3. 检查是否已有split字段
4. 如果没有split字段，进行分割
5. 保存train.json和test.json到 outputs/splits/{dataset}/

重要说明：
- 此脚本只读取原始数据，不会修改或删除任何原始文件
- 分割后的数据保存到 outputs/splits/ 目录
- RAGTruth等已有split字段的数据集会直接使用原有分割

使用方法:
    python scripts/split_dataset.py
    
配置在params.yaml中:
    datasets:
      - name: ragtruth
    split:
      test_ratio: 0.2
      random_seed: 42
      stratify: true
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import yaml
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = project_root / 'params.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    return params


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """加载数据集配置"""
    dataset_config_path = project_root / 'config' / 'dataset' / f'{dataset_name}.yaml'
    
    if dataset_config_path.exists():
        with open(dataset_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    return {'name': dataset_name}


def load_ragtruth_data(dataset_path: Path, config: Dict[str, Any]) -> List[Dict]:
    """加载RAGTruth数据集"""
    response_file = dataset_path / "response.jsonl"
    source_file = dataset_path / "source_info.jsonl"
    
    if not response_file.exists():
        raise FileNotFoundError(f"response.jsonl not found: {response_file}")
    
    # 加载source_info
    source_map = {}
    if source_file.exists():
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                source_id = item.get("source_id")
                if source_id:
                    source_map[source_id] = item
    
    # 加载response数据
    data = []
    exclude_quality = set(config.get('exclude_quality', ['incorrect_refusal', 'truncated']))
    
    with open(response_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            
            # 质量过滤
            if item.get('quality', 'good') in exclude_quality:
                continue
            
            # 获取source信息
            source_id = item.get('source_id', '')
            source_info = source_map.get(source_id, {})
            
            # 获取标签
            labels = item.get('labels', [])
            label = 1 if labels else 0
            
            # 构建数据项
            data_item = {
                'id': str(item.get('id', '')),
                'prompt': source_info.get('prompt', ''),
                'response': item.get('response', ''),
                'label': label,
                'split': item.get('split', ''),  # RAGTruth已有split字段
                'model': item.get('model', ''),
                'task_type': source_info.get('task_type', ''),
                'metadata': {
                    'source_id': source_id,
                    'quality': item.get('quality'),
                    'temperature': item.get('temperature'),
                    'n_hallucinations': len(labels),
                    'hallucination_spans': labels,
                }
            }
            data.append(data_item)
    
    logger.info(f"Loaded {len(data)} samples from RAGTruth")
    return data


def load_generic_data(dataset_path: Path) -> List[Dict]:
    """加载通用数据集（JSON或JSONL格式）"""
    if dataset_path.is_file():
        if dataset_path.suffix == '.jsonl':
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        elif dataset_path.suffix == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                if all(isinstance(v, dict) for v in data.values()):
                    return [{'id': k, **v} for k, v in data.items()]
                return [data]
            return data
    
    # 尝试查找数据文件
    for filename in ['data.jsonl', 'data.json', 'samples.jsonl', 'samples.json']:
        filepath = dataset_path / filename
        if filepath.exists():
            return load_generic_data(filepath)
    
    raise FileNotFoundError(f"No data file found in {dataset_path}")


def load_dataset_data(dataset_name: str, dataset_config: Dict[str, Any]) -> List[Dict]:
    """根据数据集名称加载数据"""
    dataset_path = Path(dataset_config.get('path', f'data/{dataset_name}'))
    
    if dataset_name.lower() == 'ragtruth':
        return load_ragtruth_data(dataset_path, dataset_config)
    else:
        return load_generic_data(dataset_path)


def check_existing_splits(data: List[Dict]) -> Dict[str, List[Dict]]:
    """检查数据是否已有split字段"""
    splits = defaultdict(list)
    has_split = False
    
    for item in data:
        split = item.get('split')
        if split:
            has_split = True
            split_lower = split.lower()
            if split_lower in ['val', 'validation', 'dev']:
                split = 'test'
            elif split_lower in ['train', 'training']:
                split = 'train'
            elif split_lower in ['test', 'testing']:
                split = 'test'
            splits[split].append(item)
    
    if has_split:
        logger.info(f"Found existing splits: { {k: len(v) for k, v in splits.items()} }")
        return dict(splits)
    
    return {}


def perform_split(
    data: List[Dict],
    test_ratio: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True,
    label_field: str = 'label'
) -> Tuple[List[Dict], List[Dict]]:
    """执行数据分割"""
    if len(data) == 0:
        logger.warning("Empty dataset, returning empty splits")
        return [], []
    
    labels = None
    if stratify:
        labels = []
        for item in data:
            label = item.get(label_field, 0)
            if isinstance(label, bool):
                label = int(label)
            elif isinstance(label, str):
                label = 1 if label.lower() in ['true', '1', 'hallucinated', 'yes'] else 0
            labels.append(label)
        
        label_counts = defaultdict(int)
        for l in labels:
            label_counts[l] += 1
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        min_count = min(label_counts.values()) if label_counts else 0
        if min_count < 2:
            logger.warning(f"Minimum class count ({min_count}) < 2, disabling stratify")
            labels = None
    
    train_data, test_data = train_test_split(
        data,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels
    )
    
    # 添加split字段
    for item in train_data:
        item['split'] = 'train'
    for item in test_data:
        item['split'] = 'test'
    
    logger.info(f"Split result: train={len(train_data)}, test={len(test_data)}")
    
    return train_data, test_data


def save_splits(
    train_data: List[Dict],
    test_data: List[Dict],
    output_dir: Path
):
    """保存分割后的数据（不会删除任何原始文件）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def ensure_complete_data(data: List[Dict]) -> List[Dict]:
        """确保每条数据包含所有必要字段"""
        complete_data = []
        for i, item in enumerate(data):
            complete_item = {
                'id': item.get('id', str(i)),
                'prompt': item.get('prompt', ''),
                'response': item.get('response', ''),
                'label': item.get('label', 0),
                'split': item.get('split', ''),
                'model': item.get('model', ''),
                'task_type': item.get('task_type', ''),
            }
            for key, value in item.items():
                if key not in complete_item:
                    complete_item[key] = value
            if 'metadata' not in complete_item:
                complete_item['metadata'] = {}
            complete_data.append(complete_item)
        return complete_data
    
    train_data = ensure_complete_data(train_data)
    test_data = ensure_complete_data(test_data)
    
    # 保存train.json
    train_path = output_dir / 'train.json'
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved train data to {train_path} ({len(train_data)} samples)")
    
    # 保存test.json
    test_path = output_dir / 'test.json'
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved test data to {test_path} ({len(test_data)} samples)")
    
    # 统计标签
    train_label_counts = defaultdict(int)
    test_label_counts = defaultdict(int)
    for item in train_data:
        train_label_counts[str(item.get('label', 0))] += 1
    for item in test_data:
        test_label_counts[str(item.get('label', 0))] += 1
    
    # 保存分割信息
    info = {
        'train_count': len(train_data),
        'test_count': len(test_data),
        'total': len(train_data) + len(test_data),
        'train_labels': dict(train_label_counts),
        'test_labels': dict(test_label_counts),
        'preserved_fields': list(train_data[0].keys()) if train_data else [],
    }
    
    info_path = output_dir / 'split_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved split info to {info_path}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Starting dataset split")
    logger.info("=" * 60)
    
    # 加载配置
    params = load_config()
    datasets_config = params.get('datasets', [])
    split_config = params.get('split', {})
    
    # 获取split参数
    test_ratio = split_config.get('test_ratio', 0.2)
    random_seed = split_config.get('random_seed', params.get('seed', 42))
    stratify = split_config.get('stratify', True)
    
    logger.info(f"Split config: test_ratio={test_ratio}, seed={random_seed}, stratify={stratify}")
    
    # 处理每个数据集
    for ds_cfg in datasets_config:
        dataset_name = ds_cfg.get('name') if isinstance(ds_cfg, dict) else ds_cfg
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*40}")
        
        # 加载数据集配置
        dataset_config = load_dataset_config(dataset_name)
        if isinstance(ds_cfg, dict):
            dataset_config.update(ds_cfg)
        
        # 加载数据
        try:
            data = load_dataset_data(dataset_name, dataset_config)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
        
        logger.info(f"Loaded {len(data)} samples")
        
        if len(data) == 0:
            logger.warning("No data loaded, skipping")
            continue
        
        # 检查是否已有split字段
        existing_splits = check_existing_splits(data)
        
        if existing_splits:
            logger.info("Using existing splits from data (no re-splitting needed)")
            train_data = existing_splits.get('train', [])
            test_data = existing_splits.get('test', [])
        else:
            logger.info("No existing splits found, performing split")
            train_data, test_data = perform_split(
                data,
                test_ratio=test_ratio,
                random_seed=random_seed,
                stratify=stratify,
                label_field=dataset_config.get('label_field', 'label')
            )
        
        # 确定输出目录
        output_base = Path(params.get('output_dir', 'outputs'))
        output_dir = output_base / 'splits' / dataset_name
        
        # 保存（不删除任何原始文件）
        save_splits(train_data, test_data, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Dataset split completed successfully")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())