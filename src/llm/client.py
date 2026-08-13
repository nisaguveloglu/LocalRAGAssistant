"""OpenAI-compatible client wrapper for local LLM inference."""

from typing import Any, Iterable, Sequence, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from config import CHAT_MODEL, OLLAMA_API_KEY, OLLAMA_BASE_URL
from llm.provider import LLMProvider
from utils.logger import logger


class LLMClient(LLMProvider):
    """Client wrapper for local LLM interaction via Ollama API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        api_key: str = OLLAMA_API_KEY,
        model: str = CHAT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Generates a response for a plain text prompt string."""
        return self.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Answer only using the provided context whenever possible."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: Sequence[ChatCompletionMessageParam] | Sequence[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Sends chat messages to the LLM backend."""
        try:
            formatted_messages = cast(
                Iterable[ChatCompletionMessageParam],
                messages,
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response.choices:
                logger.warning("LLM response returned no choices.")
                return ""

            content = response.choices[0].message.content
            if content is None:
                logger.warning("LLM response content was None.")
                return ""

            return content.strip()
        except Exception as err:
            logger.error("LLM inference failed: %s", err)
            raise RuntimeError(
                f"Failed to communicate with local LLM service at {self.base_url}. "
                "Ensure Ollama is running and the model is pulled."
            ) from err