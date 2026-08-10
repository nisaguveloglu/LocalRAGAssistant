# Local RAG Assistant

A local Retrieval-Augmented Generation (RAG) application developed with Python for question answering over PDF documents.

The system loads PDF documents, splits their text into chunks, generates embeddings, stores the processed data in SQLite, retrieves relevant content for a user query, and generates an answer using a locally running LLM through Ollama.

## Technologies

- Python
- Sentence Transformers
- SQLite
- PyPDF
- Ollama
- Phi-3 Mini

## Architecture

PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
SQLite Database
      ↓
Semantic Retrieval
      ↓
Prompt Builder
      ↓
Ollama / Phi-3 Mini
      ↓
Answer

## Project Structure

LocalRAGAssistant/
├── database/
├── documents/
├── src/
│   ├── domain/
│   ├── ingestion/
│   ├── llm/
│   ├── pipeline/
│   ├── retrieval/
│   ├── storage/
│   └── utils/
├── .gitignore
├── README.md
└── requirements.txt

## Installation

Clone the repository:
git clone <YOUR_REPOSITORY_URL>
cd LocalRAGAssistant

Create a virtual environment:
python3 -m venv .venv

Activate the virtual environment:
source .venv/bin/activate

Install the required dependencies:
pip install -r requirements.txt

## Setup
Install Ollama and start the local LLM service:
ollama serve

In another terminal, download the required model:
ollama pull phi3:mini

Place the PDF documents you want to work with inside the documents/ directory.

## Usage
First, process the documents:

python src/main.py ingest

The ingestion pipeline:

loads the PDF documents,
extracts their text,
splits the text into chunks,
generates embeddings,
stores the chunks and embeddings in SQLite.

Then start the RAG assistant:

python src/main.py chat