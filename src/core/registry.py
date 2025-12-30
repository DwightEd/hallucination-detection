"""Component registry pattern for extensible architecture.

Usage:
    @DATASETS.register("my_dataset")
    class MyDataset(BaseDataset):
        ...
    
    dataset = DATASETS.create("my_dataset", path="./data")
"""
from __future__ import annotations
from typing import TypeVar, Generic, Dict, Type, Any, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Registry(Generic[T]):
    """Simple component registry."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(self, name: str, aliases: Optional[List[str]] = None) -> Callable:
        """Decorator to register a class."""
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                logger.warning(f"{self.name}: Overwriting '{name}'")
            self._registry[name] = cls
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name
            return cls
        return decorator
    
    def get(self, name: str) -> Type[T]:
        """Get registered class by name."""
        resolved = self._aliases.get(name, name)
        if resolved not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"{self.name}: '{name}' not found. Available: [{available}]")
        return self._registry[resolved]
    
    def create(self, name: str, **kwargs: Any) -> T:
        """Create instance by name."""
        cls = self.get(name)
        return cls(**kwargs)
    
    def contains(self, name: str) -> bool:
        """Check if name is registered."""
        resolved = self._aliases.get(name, name)
        return resolved in self._registry
    
    def list(self) -> List[str]:
        """List all registered names."""
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        return self.contains(name)
    
    def __len__(self) -> int:
        return len(self._registry)


# Global Registries
DATASETS = Registry("datasets")
METHODS = Registry("methods")
MODELS = Registry("models")
EXTRACTORS = Registry("extractors")
LLM_APIS = Registry("llm_apis")


def list_available() -> Dict[str, List[str]]:
    """List all available components."""
    return {
        "datasets": DATASETS.list(),
        "methods": METHODS.list(),
        "models": MODELS.list(),
        "extractors": EXTRACTORS.list(),
        "llm_apis": LLM_APIS.list(),
    }
