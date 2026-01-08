"""Predefined prompt templates for different tasks and datasets."""
from __future__ import annotations
from typing import Optional

from .base import PromptTemplate


# ==============================================================================
# Predefined Templates
# ==============================================================================

# Basic QA prompt
QA_PROMPT = PromptTemplate(
    name="qa",
    template="Question: {question}\nAnswer:",
)

# QA with context
QA_WITH_CONTEXT_PROMPT = PromptTemplate(
    name="qa_with_context",
    template="Context: {context}\n\nQuestion: {question}\nAnswer:",
)

# RAGTruth (passthrough - uses original prompt)
RAGTRUTH_PROMPT = PromptTemplate(
    name="ragtruth",
    template="{prompt}",
)

# TruthfulQA
TRUTHFULQA_PROMPT = PromptTemplate(
    name="truthfulqa",
    template="Q: {question}\nA:",
)

# HaluEval QA
HALUEVAL_QA_PROMPT = PromptTemplate(
    name="halueval_qa",
    template="Question: {question}\nAnswer:",
)

# HaluEval with knowledge
HALUEVAL_QA_WITH_KNOWLEDGE_PROMPT = PromptTemplate(
    name="halueval_qa_knowledge",
    template="Knowledge: {knowledge}\n\nQuestion: {question}\nAnswer:",
)

# Summarization prompt
SUMMARIZATION_PROMPT = PromptTemplate(
    name="summarization",
    template="Summarize the following document:\n\n{document}\n\nSummary:",
)

# Data to text prompt
DATA2TEXT_PROMPT = PromptTemplate(
    name="data2text",
    template="Describe the following data:\n\n{data}\n\nDescription:",
)

# Chat-style prompt with system message
CHAT_QA_PROMPT = PromptTemplate(
    name="chat_qa",
    template="{question}",
    system_message="You are a helpful assistant. Answer questions accurately and concisely.",
)


# ==============================================================================
# Template Registry
# ==============================================================================

_TEMPLATES = {
    "qa": QA_PROMPT,
    "qa_with_context": QA_WITH_CONTEXT_PROMPT,
    "ragtruth": RAGTRUTH_PROMPT,
    "truthfulqa": TRUTHFULQA_PROMPT,
    "halueval_qa": HALUEVAL_QA_PROMPT,
    "halueval_qa_knowledge": HALUEVAL_QA_WITH_KNOWLEDGE_PROMPT,
    "summarization": SUMMARIZATION_PROMPT,
    "data2text": DATA2TEXT_PROMPT,
    "chat_qa": CHAT_QA_PROMPT,
}


def get_prompt_template(name: str) -> Optional[PromptTemplate]:
    """Get prompt template by name.
    
    Args:
        name: Template name (case-insensitive)
        
    Returns:
        PromptTemplate or None if not found
    """
    return _TEMPLATES.get(name.lower())


def register_template(name: str, template: PromptTemplate) -> None:
    """Register a new prompt template.
    
    Args:
        name: Template name
        template: PromptTemplate instance
    """
    _TEMPLATES[name.lower()] = template


def list_templates() -> list:
    """List all available template names.
    
    Returns:
        List of template names
    """
    return list(_TEMPLATES.keys())
