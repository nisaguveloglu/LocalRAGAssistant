import argparse

from pipeline.ingest_pipeline import IngestionPipeline
from pipeline.rag_pipeline import RAGPipeline


def main() -> None:
    
    #Local RAG Assistant uygulamasının giriş noktası.
    

    parser = argparse.ArgumentParser(
        prog="LocalRAGAssistant",
        description="Local RAG Assistant using Foundry Local",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------------
    # ingest
    # --------------------------------------------------------------

    subparsers.add_parser(
        "ingest",
        help="Read documents, create chunks, generate embeddings and store them.",
    )

    # --------------------------------------------------------------
    # chat
    # --------------------------------------------------------------

    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive RAG chat session.",
    )

    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # INGEST
    # --------------------------------------------------------------

    if args.command == "ingest":

        pipeline = IngestionPipeline()

        pipeline.run()

        return

    # --------------------------------------------------------------
    # CHAT
    # --------------------------------------------------------------

    if args.command == "chat":

        pipeline = RAGPipeline(
            top_k=args.top_k,
        )

        print("=" * 60)
        print("Local RAG Assistant")
        print("Type 'exit' to quit.")
        print("=" * 60)

        while True:

            question = input("\nYou > ").strip()

            if not question:
                continue

            if question.lower() in {
                "exit",
                "quit",
                "q",
            }:
                break

            answer = pipeline.ask(question)

            print(f"\nAssistant > {answer}")


if __name__ == "__main__":
    main()