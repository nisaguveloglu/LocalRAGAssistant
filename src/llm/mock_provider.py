"""Mock LLM provider implementation for offline testing."""

from typing import Any, Sequence

from llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock provider for unit testing and offline development."""

    def chat(self, messages: Sequence[dict[str, Any]]) -> str:
        response: list[str] = []
        for message in messages:
            role = str(message.get("role", "")).upper()
            response.append(f"\n===== {role} =====\n")
            response.append(str(message.get("content", "")))
        return "\n".join(response)