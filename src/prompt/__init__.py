"""Prompt templates and utilities for hallucination detection."""

from .base import (
    PromptConfig,
    QaPromptConfig,
    RAGTruthPromptConfig,
    PromptTemplate,
    ChatMessage,
)
from .templates import (
    QA_PROMPT,
    QA_WITH_CONTEXT_PROMPT,
    RAGTRUTH_PROMPT,
    TRUTHFULQA_PROMPT,
    HALUEVAL_QA_PROMPT,
    get_prompt_template,
    register_template,
)

__all__ = [
    # Configuration classes
    "PromptConfig",
    "QaPromptConfig",
    "RAGTruthPromptConfig",
    # Base classes
    "PromptTemplate",
    "ChatMessage",
    # Predefined templates
    "QA_PROMPT",
    "QA_WITH_CONTEXT_PROMPT",
    "RAGTRUTH_PROMPT",
    "TRUTHFULQA_PROMPT",
    "HALUEVAL_QA_PROMPT",
    # Functions
    "get_prompt_template",
    "register_template",
]
