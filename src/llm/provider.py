from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Base interface for all LLM providers.
    """

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """
        Send a chat request to the model.

        Args:
            messages: OpenAI-compatible message list.

        Returns:
            Model response as string.
        """
        pass