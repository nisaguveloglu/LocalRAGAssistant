"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Any, Sequence


class LLMProvider(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    def chat(self, messages: Sequence[dict[str, Any]]) -> str:
        """Sends chat completion messages to the language model."""
        pass