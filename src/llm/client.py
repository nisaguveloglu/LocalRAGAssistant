from openai import OpenAI

from config import (
    CHAT_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)


class LLMClient:
    """
    Local LLM client using Ollama's OpenAI-compatible API.
    """

    def __init__(self) -> None:

        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
        )

        self.model = CHAT_MODEL

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a response from a plain text prompt.
        """

        response = self.client.chat.completions.create(
            model=self.model,
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

        return response.choices[0].message.content.strip()

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Chat using an existing OpenAI-format message list.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()