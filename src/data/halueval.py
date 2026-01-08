"""HaluEval dataset parser."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Union
import json
import logging

from src.core import Sample, TaskType, DatasetConfig, DatasetError, DATASETS
from .base import BaseDataset

logger = logging.getLogger(__name__)

_SUBTASK_MAP = {"qa": TaskType.QA, "summarization": TaskType.SUMMARY, "dialogue": TaskType.DIALOGUE}


@DATASETS.register("halueval", aliases=["halu_eval", "HaluEval"])
class HaluEvalDataset(BaseDataset):
    """HaluEval hallucination evaluation dataset."""
    
    def __init__(self, path: Union[str, Path], config: Optional[DatasetConfig] = None, subtask: str = "qa"):
        self.subtask = subtask.lower()
        if self.subtask not in _SUBTASK_MAP:
            raise DatasetError(f"Unknown subtask: {subtask}", details={"valid": list(_SUBTASK_MAP.keys())})
        super().__init__(path, config)
        
        if self.path.is_dir():
            patterns = [f"{self.subtask}_samples.json", f"{self.subtask}.json", f"{self.subtask}.jsonl"]
            for pattern in patterns:
                if (candidate := self.path / pattern).exists():
                    self.file_path = candidate
                    break
            else:
                raise DatasetError(f"No {self.subtask} data file found in {self.path}")
        else:
            self.file_path = self.path
    
    def __iter__(self) -> Iterator[Sample]:
        suffix = self.file_path.suffix.lower()
        if suffix == ".jsonl":
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        yield self._parse_item(json.loads(line), idx)
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("data") or data.get("samples") or [data]
            for idx, item in enumerate(data):
                yield self._parse_item(item, idx)
    
    def _parse_item(self, item: Dict[str, Any], idx: int) -> Sample:
        label = self._parse_label(item)
        
        if self.subtask == "qa":
            question = item.get("question") or item.get("user_query") or ""
            response = item.get("hallucinated_answer") or item.get("response") or item.get("answer") or ""
            reference = item.get("right_answer") or item.get("ground_truth") or ""
            context = item.get("knowledge") or item.get("context") or ""
            prompt = f"Context: {context}\n\nQuestion: {question}" if context else question
            task_type = TaskType.QA
        elif self.subtask == "summarization":
            document = item.get("document") or item.get("source") or ""
            response = item.get("hallucinated_summary") or item.get("summary") or ""
            reference = item.get("right_summary") or ""
            prompt = f"Summarize the following document:\n\n{document}"
            task_type = TaskType.SUMMARY
        elif self.subtask == "dialogue":
            history = item.get("dialogue_history") or item.get("context") or ""
            knowledge = item.get("knowledge") or ""
            response = item.get("hallucinated_response") or item.get("response") or ""
            reference = item.get("right_response") or ""
            parts = [f"Knowledge: {knowledge}"] if knowledge else []
            parts.append(f"Dialogue:\n{history}")
            prompt = "\n\n".join(parts)
            task_type = TaskType.DIALOGUE
        else:
            prompt = str(item.get("input", item.get("prompt", "")))
            response = str(item.get("output", item.get("response", "")))
            reference = str(item.get("reference", ""))
            task_type = TaskType.OTHER
        
        return Sample(
            id=str(item.get("id", idx)),
            prompt=prompt,
            response=response,
            reference=reference,
            label=label,
            task_type=task_type,
            metadata={"subtask": self.subtask}
        )
    
    def _parse_label(self, item: Dict[str, Any]) -> int:
        for field in ["hallucination", "label", "is_hallucinated"]:
            if field in item:
                val = item[field]
                if isinstance(val, bool):
                    return 1 if val else 0
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    return 1 if val.lower() in ("yes", "true", "1") else 0
        return 0


@DATASETS.register("halueval_qa")
class HaluEvalQADataset(HaluEvalDataset):
    def __init__(self, path: Union[str, Path], config: Optional[DatasetConfig] = None):
        super().__init__(path, config, subtask="qa")


@DATASETS.register("halueval_sum", aliases=["halueval_summarization"])
class HaluEvalSumDataset(HaluEvalDataset):
    def __init__(self, path: Union[str, Path], config: Optional[DatasetConfig] = None):
        super().__init__(path, config, subtask="summarization")


@DATASETS.register("halueval_dial", aliases=["halueval_dialogue"])
class HaluEvalDialogueDataset(HaluEvalDataset):
    def __init__(self, path: Union[str, Path], config: Optional[DatasetConfig] = None):
        super().__init__(path, config, subtask="dialogue")
