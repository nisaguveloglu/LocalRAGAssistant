from llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """
    Mock provider used during development.

    Instead of calling a real LLM,
    it simply returns the prompt that would
    normally be sent to the model.
    """

    def chat(self, messages: list[dict]) -> str:

        response = []

        for message in messages:

            role = message["role"].upper()

            response.append(f"\n===== {role} =====\n")

            response.append(message["content"])

        return "\n".join(response)