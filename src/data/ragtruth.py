"""RAGTruth dataset parser."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Set
import json
import logging

from src.core import Sample, TaskType, SplitType, DatasetConfig, DatasetError, DATASETS
from .base import BaseDataset

logger = logging.getLogger(__name__)

_TASK_TYPE_MAP = {"QA": TaskType.QA, "Summary": TaskType.SUMMARY, "Data2txt": TaskType.DATA2TXT}


@DATASETS.register("ragtruth", aliases=["rag_truth", "RAGTruth"])
class RAGTruthDataset(BaseDataset):
    """RAGTruth hallucination detection dataset."""
    
    def __init__(self, path: Path, config: Optional[DatasetConfig] = None,
                 splits: Optional[List[str]] = None, task_types: Optional[List[str]] = None,
                 models: Optional[List[str]] = None, exclude_quality: Optional[List[str]] = None):
        super().__init__(path, config)
        if config:
            splits = splits or config.splits
            task_types = task_types or config.task_types
        self.split_filter: Optional[Set[str]] = set(splits) if splits else None
        self.task_filter: Optional[Set[str]] = set(task_types) if task_types else None
        self.model_filter: Optional[Set[str]] = set(models) if models else None
        self.exclude_quality: Set[str] = set(exclude_quality or ["incorrect_refusal", "truncated"])
        self._source_cache: Optional[Dict[str, Dict]] = None
        self._validate_structure()
    
    def _validate_structure(self) -> None:
        if not (self.path / "response.jsonl").exists():
            raise DatasetError(f"RAGTruth response.jsonl not found", details={"path": str(self.path)})
        if not (self.path / "source_info.jsonl").exists():
            raise DatasetError(f"RAGTruth source_info.jsonl not found", details={"path": str(self.path)})
    
    def _load_source_info(self) -> Dict[str, Dict]:
        if self._source_cache is not None:
            return self._source_cache
        self._source_cache = {}
        with open(self.path / "source_info.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if source_id := item.get("source_id"):
                        self._source_cache[source_id] = item
        return self._source_cache
    
    def __iter__(self) -> Iterator[Sample]:
        source_map = self._load_source_info()
        with open(self.path / "response.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not self._should_include(item):
                    continue
                if sample := self._parse_response(item, source_map):
                    yield sample
    
    def _should_include(self, item: Dict[str, Any]) -> bool:
        if self.split_filter and item.get("split", "") not in self.split_filter:
            return False
        if item.get("quality", "good") in self.exclude_quality:
            return False
        if self.model_filter and item.get("model", "") not in self.model_filter:
            return False
        return True
    
    def _parse_response(self, item: Dict[str, Any], source_map: Dict[str, Dict]) -> Optional[Sample]:
        source_id = item.get("source_id", "")
        source_info = source_map.get(source_id, {})
        if not source_info:
            return None
        
        task_type_str = source_info.get("task_type", "QA")
        task_type = _TASK_TYPE_MAP.get(task_type_str, TaskType.QA)
        
        if self.task_filter and task_type_str not in self.task_filter and task_type.value not in self.task_filter:
            return None
        
        prompt = self._build_prompt(source_info, task_type)
        labels = item.get("labels", [])
        split = SplitType.TRAIN if item.get("split") == "train" else SplitType.TEST if item.get("split") == "test" else None
        
        return Sample(
            id=item.get("id", ""),
            prompt=prompt,
            response=item.get("response", ""),
            reference="",
            label=1 if labels else 0,
            task_type=task_type,
            split=split,
            metadata={
                "source_id": source_id,
                "source_model": item.get("model"),
                "temperature": item.get("temperature"),
                "hallucination_spans": [{"text": l.get("text", ""), "type": l.get("label_type", "")} for l in labels],
                "n_hallucinations": len(labels),
            }
        )
    
    def _build_prompt(self, source_info: Dict[str, Any], task_type: TaskType) -> str:
        source_data = source_info.get("source_info", {})
        original_prompt = source_info.get("prompt", "")
        
        if task_type == TaskType.QA and isinstance(source_data, dict):
            question = source_data.get("question", "")
            passages = source_data.get("passages", "")
            if isinstance(passages, list):
                passages = "\n\n".join(str(p) for p in passages)
            return f"Context:\n{passages}\n\nQuestion: {question}" if passages else question
        elif task_type == TaskType.SUMMARY:
            return f"Summarize the following:\n\n{source_data}" if source_data else original_prompt
        elif task_type == TaskType.DATA2TXT and isinstance(source_data, dict):
            return f"Describe the following data:\n\n{json.dumps(source_data, ensure_ascii=False, indent=2)}"
        return original_prompt
    
    def get_hallucinated(self) -> Iterator[Sample]:
        for sample in self:
            if sample.label == 1:
                yield sample
    
    def get_clean(self) -> Iterator[Sample]:
        for sample in self:
            if sample.label == 0:
                yield sample
