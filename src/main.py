"""Command-line interface entry point for Local RAG Assistant."""

import argparse
import sys

from config import TOP_K
from pipeline.ingest_pipeline import IngestionPipeline
from pipeline.rag_pipeline import RAGPipeline
from utils.logger import logger


def main() -> None:
    """Main CLI entry point for document ingestion and chat interaction."""
    parser = argparse.ArgumentParser(
        prog="LocalRAGAssistant",
        description="Local RAG Assistant using Sentence Transformers, SQLite, and Ollama.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "ingest",
        help="Read documents, create chunks, generate embeddings, and store in database.",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive RAG chat session.",
    )
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of retrieved context chunks.",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        pipeline = IngestionPipeline()
        pipeline.run()
        return

    if args.command == "chat":
        pipeline = RAGPipeline(top_k=args.top_k)

        print("=" * 60)
        print("Local RAG Assistant Interactive Chat")
        print("Type 'exit' or 'quit' to end session.")
        print("=" * 60)

        while True:
            try:
                question = input("\nYou > ").strip()
                if not question:
                    continue

                if question.lower() in {"exit", "quit", "q"}:
                    print("\nGoodbye!")
                    break

                answer = pipeline.ask(question)
                print(f"\nAssistant > {answer}")
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                sys.exit(0)
            except Exception as err:
                logger.error("Error generating answer: %s", err)
                print(f"\nError: {err}")


if __name__ == "__main__":
    main()