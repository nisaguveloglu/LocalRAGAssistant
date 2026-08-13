"""Prompt builder constructing system and user messages for RAG generation."""

from openai.types.chat import ChatCompletionMessageParam

from domain.search_result import SearchResult


class PromptBuilder:
    """Constructs structured chat messages for the RAG LLM."""

    def __init__(self) -> None:
        self.system_prompt = (
            "You are an AI assistant that answers questions using only the "
            "provided context.\n\n"
            "Rules:\n"
            "1. Use only the retrieved context.\n"
            "2. If the answer is not contained in the context, clearly state "
            "\"I don't have enough information in the provided documents.\"\n"
            "3. Do not make up information.\n"
            "4. Answer clearly and concisely.\n"
            "5. If multiple context sections are relevant, combine them into "
            "a single coherent answer."
        )

    def build(
        self,
        query: str,
        search_results: list[SearchResult],
    ) -> list[ChatCompletionMessageParam]:
        """Constructs an OpenAI-compatible message list containing system and user prompts."""
        context = self._build_context(search_results)

        user_prompt = (
            f"Context:\n"
            f"{context}\n\n"
            f"Question:\n"
            f"{query}\n\n"
            f"Answer:"
        )

        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]


    def _build_context(
        self,
        search_results: list[SearchResult],
    ) -> str:
        """Formats search results into a structured context string with section headers."""
        if not search_results:
            return "No relevant context found."

        sections: list[str] = []

        for index, result in enumerate(search_results, start=1):
            chunk = result.chunk
            header = f"[Source {index}] {chunk.filename} (Chunk {chunk.chunk_index})"
            if chunk.section_title:
                header += f" - Section: {chunk.section_title}"

            sections.append(f"{header}\n{chunk.content}")

        return "\n\n".join(sections)