"""Retrieval-Augmented Generation (RAG) pipeline orchestrator."""

from config import TOP_K
from ingestion.embedder import SentenceTransformerEmbedder
from llm.client import LLMClient
from llm.prompt import PromptBuilder
from retrieval.retriever import Retriever
from utils.logger import logger


class RAGPipeline:
    """RAG pipeline combining vector retrieval with LLM answer generation."""

    def __init__(
        self,
        top_k: int = TOP_K,
    ) -> None:
        self.top_k = top_k
        self.embedder = SentenceTransformerEmbedder()
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.llm = LLMClient()

    def ask(self, question: str) -> str:
        """Executes full RAG workflow for a user question."""
        logger.debug("Processing user query: %s", question)

        query_embedding = self.embedder.embed(question)

        search_results = self.retriever.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        logger.debug("Retrieved %d context chunks for query.", len(search_results))

        messages = self.prompt_builder.build(
            query=question,
            search_results=search_results,
        )

        return self.llm.chat(messages)