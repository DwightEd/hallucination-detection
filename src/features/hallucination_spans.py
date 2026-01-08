"""Utilities for converting character-level hallucination spans to token-level labels.

Following RAGTruth format for consistency across datasets.
This module provides functions to:
1. Convert character-level spans to token-level spans
2. Generate token-level hallucination labels (0/1 for each token)
3. Work with both annotated datasets (RAGTruth) and LLM-judged datasets
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class HallucinationSpanInfo:
    """Information about a hallucination span at both character and token levels.
    
    Attributes:
        char_start: Start character position in response (inclusive)
        char_end: End character position in response (exclusive)
        token_start: Start token position in full input (inclusive)
        token_end: End token position in full input (exclusive)
        text: The hallucinated text
        label_type: Type of hallucination
    """
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text: str = ""
    label_type: str = ""


def calculate_hallucination_token_spans(
    labels: List[Dict[str, Any]],
    prompt_text: str,
    response_text: str,
    tokenizer,
    end_inclusive: bool = False,
) -> List[List[int]]:
    """Convert character-level hallucination spans to token-level spans.
    
    This function maps character-level annotation (as in RAGTruth) to
    token-level positions that can be used during feature extraction.
    
    Args:
        labels: List of dicts with 'start' and 'end' (char-level relative to response)
        prompt_text: The prompt text (used to calculate prefix token length)
        response_text: The response text that contains hallucinations
        tokenizer: HuggingFace tokenizer
        end_inclusive: If True, label['end'] is inclusive; if False, exclusive
    
    Returns:
        List of [token_start_in_input, token_end_in_input_exclusive] pairs
    """
    if not labels:
        return []
    
    # Tokenize response alone to get character-to-token offsets
    response_encoding = tokenizer(
        response_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = response_encoding["offset_mapping"]
    
    # Get prefix length (tokenize prompt without special tokens for alignment)
    prompt_encoding = tokenizer(prompt_text, add_special_tokens=False)
    prefix_len = len(prompt_encoding["input_ids"])
    
    spans = []
    for item in labels:
        char_start = item.get("start", 0)
        char_end = item.get("end", 0)
        
        # Normalize to exclusive end
        if end_inclusive:
            char_end = char_end + 1
        
        # Find token indices that cover [char_start, char_end)
        token_start = None
        token_end = None
        
        for idx, (s, e) in enumerate(offsets):
            # Token covers chars [s, e) (e exclusive)
            if token_start is None and s <= char_start < e:
                token_start = idx
            # Find token_end
            if token_start is not None and s <= char_end <= e:
                token_end = idx + 1
                break
        
        # Fallback: if char_start maps but char_end extends beyond
        if token_start is not None and token_end is None:
            for j in range(token_start, len(offsets)):
                s, e = offsets[j]
                if e >= char_end:
                    token_end = j + 1
                    break
            if token_end is None:
                token_end = len(offsets)
        
        if token_start is None:
            continue
        
        # Convert to input-level token indices by adding prefix length
        spans.append([prefix_len + token_start, prefix_len + token_end])
    
    return spans


def get_token_hallucination_labels(
    seq_len: int,
    hallucination_spans: List[List[int]],
) -> List[int]:
    """Generate token-level 0/1 hallucination labels.
    
    Args:
        seq_len: Total sequence length (number of tokens in input)
        hallucination_spans: List of [start, end_exclusive] token indices
    
    Returns:
        List of length seq_len with 0 (not hallucinated) or 1 (hallucinated)
    """
    labels = [0] * seq_len
    for span in hallucination_spans:
        if len(span) >= 2:
            start, end = span[0], span[1]
            start = max(0, start)
            end = min(seq_len, end)
            for i in range(start, end):
                labels[i] = 1
    return labels


def extract_hallucination_info_from_sample(
    sample_metadata: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Extract hallucination span information from sample metadata.
    
    Works with RAGTruth format where spans are stored in metadata.
    
    Args:
        sample_metadata: Sample.metadata dict
    
    Returns:
        Tuple of (labels list, has_spans bool)
    """
    spans = sample_metadata.get("hallucination_spans", [])
    
    labels = []
    for span in spans:
        if isinstance(span, dict):
            labels.append({
                "start": span.get("start", 0),
                "end": span.get("end", 0),
                "text": span.get("text", ""),
                "label_type": span.get("type", span.get("label_type", "")),
            })
    
    return labels, len(labels) > 0


def calculate_hallucination_labels_for_input(
    prompt_text: str,
    response_text: str,
    hallucination_spans: List[Dict[str, Any]],
    tokenizer,
    add_special_tokens: bool = True,
) -> Tuple[List[int], List[List[int]], int]:
    """Calculate token-level hallucination labels for a full input sequence.
    
    This is the main function to call during feature extraction.
    
    Args:
        prompt_text: The prompt/question text
        response_text: The model response text
        hallucination_spans: List of span dicts with 'start', 'end'
        tokenizer: HuggingFace tokenizer
        add_special_tokens: Whether to add special tokens during encoding
    
    Returns:
        Tuple of:
        - hallucination_labels: List of 0/1 for each token in full input
        - token_spans: List of [start, end] token indices for each span
        - seq_len: Total sequence length
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=add_special_tokens)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    
    prompt_len = len(prompt_ids)
    seq_len = prompt_len + len(response_ids)
    
    token_spans = calculate_hallucination_token_spans(
        labels=hallucination_spans,
        prompt_text=prompt_text,
        response_text=response_text,
        tokenizer=tokenizer,
        end_inclusive=False,
    )
    
    hallucination_labels = get_token_hallucination_labels(seq_len, token_spans)
    
    return hallucination_labels, token_spans, seq_len


def needs_llm_judge_for_spans(sample_metadata: Dict[str, Any]) -> bool:
    """Check if a sample needs LLM judge to generate hallucination spans.
    
    Returns True if the sample has no span annotations.
    
    Args:
        sample_metadata: Sample.metadata dict
    
    Returns:
        True if LLM judge is needed to annotate spans
    """
    spans = sample_metadata.get("hallucination_spans", [])
    return len(spans) == 0
