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
    """Simple component registry.
    
    Allows registration and retrieval of classes by name.
    Supports aliases for alternative names.
    """
    
    def __init__(self, name: str):
        """Initialize registry.
        
        Args:
            name: Registry name (for error messages)
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(self, name: str, aliases: Optional[List[str]] = None) -> Callable:
        """Decorator to register a class.
        
        Args:
            name: Primary name for the class
            aliases: Optional list of alternative names
            
        Returns:
            Decorator function
        """
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
        """Get registered class by name.
        
        Args:
            name: Class name or alias
            
        Returns:
            The registered class
            
        Raises:
            KeyError: If name not found
        """
        resolved = self._aliases.get(name, name)
        if resolved not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"{self.name}: '{name}' not found. Available: [{available}]")
        return self._registry[resolved]
    
    def create(self, name: str, **kwargs: Any) -> T:
        """Create instance by name.
        
        Args:
            name: Class name or alias
            **kwargs: Arguments to pass to constructor
            
        Returns:
            Instance of the registered class
        """
        cls = self.get(name)
        return cls(**kwargs)
    
    def contains(self, name: str) -> bool:
        """Check if name is registered.
        
        Args:
            name: Class name or alias
            
        Returns:
            True if registered
        """
        resolved = self._aliases.get(name, name)
        return resolved in self._registry
    
    def list(self) -> List[str]:
        """List all registered names.
        
        Returns:
            List of primary names
        """
        return list(self._registry.keys())
    
    def list_with_aliases(self) -> Dict[str, List[str]]:
        """List all names with their aliases.
        
        Returns:
            Dict mapping primary names to list of aliases
        """
        result = {name: [] for name in self._registry.keys()}
        for alias, name in self._aliases.items():
            if name in result:
                result[name].append(alias)
        return result
    
    def __contains__(self, name: str) -> bool:
        return self.contains(name)
    
    def __len__(self) -> int:
        return len(self._registry)
    
    def __repr__(self) -> str:
        return f"Registry({self.name}, {len(self)} items)"


# Global Registries
DATASETS = Registry("datasets")
METHODS = Registry("methods")
MODELS = Registry("models")
EXTRACTORS = Registry("extractors")
LLM_APIS = Registry("llm_apis")


def list_available() -> Dict[str, List[str]]:
    """List all available components.
    
    Returns:
        Dict mapping registry name to list of registered items
    """
    return {
        "datasets": DATASETS.list(),
        "methods": METHODS.list(),
        "models": MODELS.list(),
        "extractors": EXTRACTORS.list(),
        "llm_apis": LLM_APIS.list(),
    }
