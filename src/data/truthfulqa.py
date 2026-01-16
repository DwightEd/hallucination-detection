"""TruthfulQA dataset parser."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Union
import json
import logging

from src.core import Sample, TaskType, SplitType, DatasetConfig, DatasetError, DATASETS
from .base import BaseDataset

logger = logging.getLogger(__name__)


@DATASETS.register("truthfulqa", aliases=["truthful_qa", "TruthfulQA"])
class TruthfulQADataset(BaseDataset):
    """TruthfulQA dataset for truthfulness evaluation."""
    
    def __init__(self, path: Optional[Union[str, Path]] = None, config: Optional[DatasetConfig] = None,
                 subset: str = "generation", split: str = "validation", use_huggingface: bool = True):
        self.subset = subset
        self.split_name = split
        self.use_huggingface = use_huggingface
        self._hf_dataset = None
        
        if path is None and use_huggingface:
            self.path = Path("huggingface://truthful_qa")
            self.config = config or DatasetConfig(name="truthfulqa")
        else:
            self.path = Path(path) if path else Path(".")
            self.config = config or DatasetConfig(path=str(self.path))
            if path:
                self._validate()
    
    def _validate(self) -> None:
        if self.use_huggingface and self.path.as_posix().startswith("huggingface://"):
            return
        super()._validate()
    
    def _load_huggingface(self):
        if self._hf_dataset is not None:
            return self._hf_dataset
        try:
            from datasets import load_dataset
            logger.info(f"Loading TruthfulQA from HuggingFace (subset={self.subset})")
            self._hf_dataset = load_dataset("truthful_qa", self.subset, split=self.split_name, trust_remote_code=True)
            logger.info(f"Loaded {len(self._hf_dataset)} samples")
            return self._hf_dataset
        except ImportError:
            raise DatasetError("datasets library required", details={"install": "pip install datasets"})
        except Exception as e:
            raise DatasetError(f"Failed to load TruthfulQA: {e}")
    
    def __iter__(self) -> Iterator[Sample]:
        if self.use_huggingface:
            dataset = self._load_huggingface()
            for idx, item in enumerate(dataset):
                yield self._parse_hf_item(item, idx)
        else:
            yield from self._iter_local()
    
    def _iter_local(self) -> Iterator[Sample]:
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            with open(self.path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        yield self._parse_local_item(json.loads(line), idx)
        elif suffix == ".json":
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("data", data.get("questions", [data]))
            for idx, item in enumerate(data):
                yield self._parse_local_item(item, idx)
    
    def _parse_hf_item(self, item: Dict[str, Any], idx: int) -> Sample:
        best_answer = item.get("best_answer", "")
        correct_answers = item.get("correct_answers", [])
        if not best_answer and correct_answers:
            best_answer = correct_answers[0]
        return Sample(
            id=str(idx),
            prompt=item.get("question", ""),
            response="",
            reference=best_answer,
            label=None,
            task_type=TaskType.QA,
            split=SplitType.VALIDATION if self.split_name == "validation" else SplitType.TEST,
            metadata={"category": item.get("category", ""), "correct_answers": correct_answers, "incorrect_answers": item.get("incorrect_answers", [])}
        )
    
    def _parse_local_item(self, item: Dict[str, Any], idx: int) -> Sample:
        best_answer = item.get("best_answer") or item.get("answer") or item.get("reference") or ""
        correct_answers = item.get("correct_answers", [])
        if not best_answer and correct_answers:
            best_answer = correct_answers[0]
        label = item.get("label")
        if label is not None:
            label = int(label)
        return Sample(
            id=str(item.get("id", idx)),
            prompt=item.get("question", item.get("prompt", "")),
            response=item.get("response", item.get("model_answer", "")),
            reference=best_answer,
            label=label,
            task_type=TaskType.QA,
            metadata={"category": item.get("category", ""), "correct_answers": correct_answers}
        )
    
    def __len__(self) -> int:
        if self.use_huggingface:
            return len(self._load_huggingface())
        return super().__len__()


def download_truthfulqa(output_dir: Union[str, Path] = "./data/truthfulqa") -> Path:
    """Download TruthfulQA and save locally."""
    from datasets import load_dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    output_file = output_dir / "truthfulqa.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return output_file
