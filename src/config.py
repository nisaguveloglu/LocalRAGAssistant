from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = PROJECT_ROOT / "documents"

DATABASE_DIR = PROJECT_ROOT / "database"

DATABASE_PATH = DATABASE_DIR / "rag.db"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

OLLAMA_API_KEY = "ollama"

CHAT_MODEL = "phi3:mini"


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100


# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = 5