"""Base classes for prompt handling.

This module provides prompt-related data structures for building
prompts from samples and templates.

Note: Configuration classes (PromptConfig, QaPromptConfig, RAGTruthPromptConfig)
are imported from src.core.config to avoid duplication.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Import configuration classes from core (single source of truth)
from src.core.config import (
    PromptConfig,
    QaPromptConfig,
    RAGTruthPromptConfig,
)


@dataclass
class ChatMessage:
    """Single chat message for conversation-style prompts.
    
    Attributes:
        role: Message role - "system", "user", or "assistant"
        content: The message content
    """
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for API calls."""
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ChatMessage":
        """Create from dictionary."""
        return cls(role=data["role"], content=data["content"])
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"ChatMessage(role={self.role!r}, content={content_preview!r})"


@dataclass
class PromptTemplate:
    """Template for generating prompts.
    
    Supports simple string formatting and conversion to chat messages.
    
    Attributes:
        name: Template identifier
        template: Format string with placeholders (e.g., "{question}")
        system_message: Optional system message for chat format
        few_shot_examples: Optional list of few-shot examples
    
    Example:
        template = PromptTemplate(
            name="qa",
            template="Question: {question}\\nAnswer:",
            system_message="You are a helpful assistant.",
        )
        
        prompt = template.format(question="What is 2+2?")
        # "Question: What is 2+2?\\nAnswer:"
        
        messages = template.to_messages(question="What is 2+2?")
        # [ChatMessage(role="system", ...), ChatMessage(role="user", ...)]
    """
    name: str
    template: str
    system_message: Optional[str] = None
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    
    def format(self, **kwargs) -> str:
        """Format template with given arguments.
        
        Args:
            **kwargs: Values to substitute into template placeholders
            
        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)
    
    def to_messages(self, **kwargs) -> List[ChatMessage]:
        """Convert to chat messages format.
        
        Creates a list of ChatMessage objects suitable for chat-based APIs.
        
        Args:
            **kwargs: Values to substitute into template placeholders
            
        Returns:
            List of ChatMessage objects
        """
        messages = []
        
        # Add system message if present
        if self.system_message:
            messages.append(ChatMessage(role="system", content=self.system_message))
        
        # Add few-shot examples
        for example in self.few_shot_examples:
            if "question" in example:
                messages.append(ChatMessage(role="user", content=example["question"]))
            if "answer" in example:
                messages.append(ChatMessage(role="assistant", content=example["answer"]))
        
        # Add current user message
        user_content = self.format(**kwargs)
        messages.append(ChatMessage(role="user", content=user_content))
        
        return messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "template": self.template,
            "system_message": self.system_message,
            "few_shot_examples": self.few_shot_examples,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "default"),
            template=data.get("template", "{question}"),
            system_message=data.get("system_message"),
            few_shot_examples=data.get("few_shot_examples", []),
        )
    
    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, template={self.template[:50]!r}...)"


# Re-export configuration classes for backward compatibility
__all__ = [
    "PromptConfig",
    "QaPromptConfig",
    "RAGTruthPromptConfig",
    "ChatMessage",
    "PromptTemplate",
]
