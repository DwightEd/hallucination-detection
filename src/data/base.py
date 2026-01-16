"""Base dataset interface and factory functions."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any, Union, Callable
import json
import logging

from src.core import Sample, TaskType, SplitType, DatasetConfig, DatasetError, DATASETS

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for all datasets."""
    
    def __init__(self, path: Union[str, Path], config: Optional[DatasetConfig] = None):
        self.path = Path(path)
        self.config = config or DatasetConfig(path=str(path))
        self._validate()
    
    def _validate(self) -> None:
        if not self.path.exists():
            raise DatasetError(f"Dataset path not found: {self.path}", details={"path": str(self.path)})
    
    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        pass
    
    def __len__(self) -> int:
        return sum(1 for _ in self)
    
    def load(self, max_samples: Optional[int] = None) -> List[Sample]:
        samples = []
        for i, sample in enumerate(self):
            if max_samples and i >= max_samples:
                break
            samples.append(sample)
        return samples
    
    def filter(self, split: Optional[SplitType] = None, task_type: Optional[TaskType] = None,
               has_label: Optional[bool] = None, predicate: Optional[Callable[[Sample], bool]] = None) -> Iterator[Sample]:
        for sample in self:
            if split and sample.split != split:
                continue
            if task_type and sample.task_type != task_type:
                continue
            if has_label is not None:
                if has_label and sample.label is None:
                    continue
                if not has_label and sample.label is not None:
                    continue
            if predicate and not predicate(sample):
                continue
            yield sample
    
    def statistics(self) -> Dict[str, Any]:
        stats = {"total": 0, "by_split": {}, "by_task": {}, "by_label": {"0": 0, "1": 0, "none": 0}}
        for sample in self:
            stats["total"] += 1
            if sample.split:
                key = sample.split.value
                stats["by_split"][key] = stats["by_split"].get(key, 0) + 1
            key = sample.task_type.value
            stats["by_task"][key] = stats["by_task"].get(key, 0) + 1
            if sample.label is None:
                stats["by_label"]["none"] += 1
            else:
                stats["by_label"][str(sample.label)] += 1
        return stats


@DATASETS.register("jsonl", aliases=["jsonlines"])
class JsonlDataset(BaseDataset):
    """Generic JSONL file dataset."""
    
    def __init__(self, path: Union[str, Path], config: Optional[DatasetConfig] = None,
                 field_mapping: Optional[Dict[str, str]] = None, task_type: TaskType = TaskType.QA):
        super().__init__(path, config)
        self.field_mapping = field_mapping or {}
        self.default_task_type = task_type
    
    def __iter__(self) -> Iterator[Sample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    yield self._parse_item(item, idx)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {idx}: {e}")
    
    def _parse_item(self, item: Dict[str, Any], idx: int) -> Sample:
        def get_field(key: str, default: Any = "") -> Any:
            mapped = self.field_mapping.get(key, key)
            return item.get(mapped, item.get(key, default))
        
        return Sample(
            id=str(get_field("id", idx)),
            prompt=str(get_field("prompt", get_field("question", ""))),
            response=str(get_field("response", get_field("answer", ""))),
            reference=str(get_field("reference", get_field("gold_answer", ""))),
            label=get_field("label") if "label" in item else None,
            task_type=self.default_task_type,
            metadata={"raw": item},
        )


@DATASETS.register("json")
class JsonDataset(BaseDataset):
    """Generic JSON file dataset."""
    
    def __init__(self, path: Union[str, Path], config: Optional[DatasetConfig] = None,
                 data_key: Optional[str] = None, field_mapping: Optional[Dict[str, str]] = None,
                 task_type: TaskType = TaskType.QA):
        super().__init__(path, config)
        self.data_key = data_key
        self.field_mapping = field_mapping or {}
        self.default_task_type = task_type
    
    def __iter__(self) -> Iterator[Sample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if self.data_key:
            data = data.get(self.data_key, [])
        if isinstance(data, dict):
            data = [data]
        for idx, item in enumerate(data):
            yield self._parse_item(item, idx)
    
    def _parse_item(self, item: Dict[str, Any], idx: int) -> Sample:
        def get_field(key: str, default: Any = "") -> Any:
            mapped = self.field_mapping.get(key, key)
            return item.get(mapped, item.get(key, default))
        return Sample(
            id=str(get_field("id", idx)),
            prompt=str(get_field("prompt", get_field("question", ""))),
            response=str(get_field("response", get_field("answer", ""))),
            reference=str(get_field("reference", "")),
            label=get_field("label"),
            task_type=self.default_task_type,
            metadata={"raw": item},
        )


def create_dataset(name_or_path: Union[str, Path], config: Optional[DatasetConfig] = None, **kwargs: Any) -> BaseDataset:
    """Create dataset by name or path."""
    if isinstance(name_or_path, str) and DATASETS.contains(name_or_path):
        if config is None:
            raise ValueError(f"Config with path required for dataset '{name_or_path}'")
        return DATASETS.create(name_or_path, path=config.path, config=config, **kwargs)
    
    path = Path(name_or_path)
    if not path.exists():
        raise DatasetError(f"Path not found: {path}")
    
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return JsonlDataset(path, config, **kwargs)
        elif suffix == ".json":
            return JsonDataset(path, config, **kwargs)
        raise DatasetError(f"Unsupported file format: {suffix}")
    
    if (path / "response.jsonl").exists():
        return DATASETS.create("ragtruth", path=path, config=config, **kwargs)
    
    raise DatasetError(f"Cannot determine dataset type for: {path}")


def load_samples(name_or_path: Union[str, Path], config: Optional[DatasetConfig] = None,
                 max_samples: Optional[int] = None, **kwargs: Any) -> List[Sample]:
    """Load samples from dataset."""
    dataset = create_dataset(name_or_path, config, **kwargs)
    return dataset.load(max_samples=max_samples)
