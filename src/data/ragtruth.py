"""RAGTruth dataset parser. 

Supports both original RAGTruth format and lapeigvals-compatible processing.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Set
import json
import logging

from src.core import Sample, TaskType, SplitType, DatasetConfig, DatasetError, DATASETS
from .base import BaseDataset

logger = logging.getLogger(__name__)

_TASK_TYPE_MAP = {
    "QA": TaskType.QA,
    "Summary": TaskType.SUMMARY,
    "Data2txt": TaskType.DATA2TXT,
}


@DATASETS.register("ragtruth", aliases=["rag_truth", "RAGTruth"])
class RAGTruthDataset(BaseDataset):
    """RAGTruth hallucination detection dataset. 
    
    RAGTruth contains: 
    - response.jsonl: Model responses with hallucination labels
    - source_info.jsonl: Source context and prompts
    
    Labels are at span level, converted to response level: 
    - label=1 if any hallucination span exists
    - label=0 if no hallucination
    """
    
    def __init__(
        self,
        path: Path,
        config: Optional[DatasetConfig] = None,
        splits: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        exclude_quality: Optional[List[str]] = None,
    ):
        # Handle path
        if path is None and config and config.path:
            path = Path(config.path)
        super().__init__(path, config)
        
        # Filters from config
        if config: 
            splits = splits or config.splits
            task_types = task_types or config.task_types
            models = models or getattr(config, 'models', None)
            exclude_quality = exclude_quality or getattr(config, 'exclude_quality', None)
            
        self.split_filter: Optional[Set[str]] = set(splits) if splits else None
        self.task_filter: Optional[Set[str]] = set(task_types) if task_types else None
        self.model_filter: Optional[List[str]] = list(models) if models else None
        self.exclude_quality: Set[str] = set(exclude_quality or ["incorrect_refusal", "truncated"])
        
        # Cache
        self._source_cache: Optional[Dict[str, Dict]] = None
        
        self._validate_structure()
        
        logger.info(f"RAGTruth filters: splits={self.split_filter}, tasks={self.task_filter}, models={self.model_filter}")
    
    def _validate(self) -> None:
        """Override base validation."""
        if self.path is None:
            raise DatasetError("RAGTruth path not specified")
        if not self.path.exists():
            raise DatasetError(f"RAGTruth path not found: {self.path}")
    
    def _validate_structure(self) -> None:
        """Validate RAGTruth directory structure."""
        if not (self.path / "response.jsonl").exists():
            raise DatasetError(
                f"RAGTruth response.jsonl not found",
                details={"path": str(self.path)}
            )
        if not (self.path / "source_info.jsonl").exists():
            raise DatasetError(
                f"RAGTruth source_info.jsonl not found",
                details={"path": str(self.path)}
            )
    
    def _load_source_info(self) -> Dict[str, Dict]:
        """Load and cache source information."""
        if self._source_cache is not None:
            return self._source_cache
        
        self._source_cache = {}
        source_file = self.path / "source_info.jsonl"
        
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                item = json.loads(line)
                source_id = item.get("source_id")
                if source_id:
                    self._source_cache[source_id] = item
        
        logger.info(f"Loaded {len(self._source_cache)} source entries")
        return self._source_cache
    
    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples."""
        source_map = self._load_source_info()
        response_file = self.path / "response.jsonl"
        
        with open(response_file, 'r', encoding='utf-8') as f:
            for line in f: 
                line = line.strip()
                if not line:
                    continue
                    
                item = json.loads(line)
                
                if not self._should_include(item):
                    continue
                
                sample = self._parse_response(item, source_map)
                if sample:
                    yield sample
    
    def _should_include(self, item: Dict[str, Any]) -> bool:
        """Check if item should be included based on filters."""
        # Split filter
        if self.split_filter and item.get("split", "") not in self.split_filter:
            return False
        
        # Quality filter
        if item.get("quality", "good") in self.exclude_quality:
            return False
        
        # Model filter (partial match: "gpt-4" matches "gpt-4-0613")
        if self.model_filter:
            model_name = item.get("model", "")
            if not any(f in model_name or model_name in f for f in self.model_filter):
                return False
        
        return True
    
    def _parse_response(
        self,
        item: Dict[str, Any],
        source_map: Dict[str, Dict]
    ) -> Optional[Sample]:
        """Parse response item to Sample."""
        source_id = item.get("source_id", "")
        source_info = source_map.get(source_id, {})
        
        if not source_info: 
            logger.warning(f"Source not found for {source_id}")
            return None
        
        # Get task type
        task_type_str = source_info.get("task_type", "QA")
        task_type = _TASK_TYPE_MAP.get(task_type_str, TaskType.QA)
        
        # Apply task filter
        if self.task_filter: 
            if task_type_str not in self.task_filter and task_type.value not in self.task_filter:
                return None
        
        # Build prompt
        prompt = self._build_prompt(source_info, task_type)
        
        # Get labels (span-level to response-level)
        labels = item.get("labels", [])
        response_label = 1 if labels else 0
        
        # Get split
        split_str = item.get("split", "")
        split = None
        if split_str == "train":
            split = SplitType.TRAIN
        elif split_str == "test":
            split = SplitType.TEST
        elif split_str == "validation":
            split = SplitType.VALIDATION
        
        return Sample(
            id=str(item.get("id", "")),
            prompt=prompt,
            response=item.get("response", ""),
            reference="",
            label=response_label,
            task_type=task_type,
            split=split,
            metadata={
                "source_id": source_id,
                "source_model": item.get("model"),
                "temperature": item.get("temperature"),
                "quality": item.get("quality"),
                "hallucination_spans": [
                    {
                        "text": l.get("text", ""),
                        "type": l.get("label_type", ""),
                        "start": l.get("start"),
                        "end": l.get("end"),
                    }
                    for l in labels
                ],
                "n_hallucinations": len(labels),
                "task_type_str": task_type_str,
            }
        )
    
    def _build_prompt(self, source_info:  Dict[str, Any], task_type:  TaskType) -> str:
        """Build prompt from source info."""
        source_data = source_info.get("source_info", {})
        original_prompt = source_info.get("prompt", "")
        
        if task_type == TaskType.QA and isinstance(source_data, dict):
            question = source_data.get("question", "")
            passages = source_data.get("passages", "")
            
            if isinstance(passages, list):
                passages = "\n\n".join(str(p) for p in passages)
            
            if passages:
                return f"Context:\n{passages}\n\nQuestion:  {question}"
            return question
        
        elif task_type == TaskType.SUMMARY: 
            if source_data: 
                return f"Summarize the following:\n\n{source_data}"
            return original_prompt
        
        elif task_type == TaskType.DATA2TXT and isinstance(source_data, dict):
            return f"Describe the following data:\n\n{json.dumps(source_data, ensure_ascii=False, indent=2)}"
        
        return original_prompt
    
    def get_hallucinated(self) -> Iterator[Sample]:
        """Get only hallucinated samples."""
        for sample in self:
            if sample.label == 1:
                yield sample
    
    def get_clean(self) -> Iterator[Sample]:
        """Get only clean (non-hallucinated) samples."""
        for sample in self: 
            if sample.label == 0:
                yield sample
    
    def get_by_task(self, task_type: TaskType) -> Iterator[Sample]: 
        """Get samples by task type."""
        for sample in self:
            if sample.task_type == task_type: 
                yield sample
    
    def statistics(self) -> Dict[str, Any]: 
        """Get dataset statistics."""
        stats = super().statistics()
        
        # Add RAGTruth specific stats
        stats["by_source_model"] = {}
        stats["by_quality"] = {}
        
        source_map = self._load_source_info()
        response_file = self.path / "response.jsonl"
        
        with open(response_file, 'r', encoding='utf-8') as f:
            for line in f: 
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                
                model = item.get("model", "unknown")
                stats["by_source_model"][model] = stats["by_source_model"].get(model, 0) + 1
                
                quality = item.get("quality", "good")
                stats["by_quality"][quality] = stats["by_quality"].get(quality, 0) + 1
        
        return stats