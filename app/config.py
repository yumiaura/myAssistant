"""Configuration constants for the application."""

import os
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX = os.getenv("FAISS_INDEX", "index.faiss")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "documents.pkl")
HISTORY_FILE = os.getenv("HISTORY_FILE", "history.txt")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", "PROMPT.j2")
RAW_FOLDER = os.getenv("RAW_FOLDER", "raw")

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")

# Model configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
