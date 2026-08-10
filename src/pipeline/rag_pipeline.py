from ingestion.embedder import SentenceTransformerEmbedder
from llm.client import LLMClient
from llm.prompt import PromptBuilder
from retrieval.retriever import Retriever


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline.

    İş Akışı
    --------
    User Question
            ↓
    Query Embedding
            ↓
    Retriever
            ↓
    Top-K SearchResult
            ↓
    Prompt Builder
            ↓
    LLM
            ↓
    Answer
    """

    def __init__(
        self,
        top_k: int = 5,
    ) -> None:

        self.top_k = top_k

        self.embedder = SentenceTransformerEmbedder()

        self.retriever = Retriever()

        self.prompt_builder = PromptBuilder()

        self.llm = LLMClient()

    def ask(self, question: str) -> str:
        """
        Kullanıcının sorusunu cevaplar.

        Parameters
        ----------
        question : str
            Kullanıcının sorusu.

        Returns
        -------
        str
            Model cevabı.
        """

        query_embedding = self.embedder.embed(question)

        search_results = self.retriever.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        messages = self.prompt_builder.build(
            query=question,
            search_results=search_results,
        )

        answer = self.llm.chat(messages)

        return answer