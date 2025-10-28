"""FAISS vector store implementation."""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import FAISS_INDEX, DOCUMENTS_PATH, EMBEDDING_MODEL_NAME


def _load_model():
    """Load the sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _read_text_files(folder_path: Path) -> List[str]:
    """Read all text files from folder and return their content."""
    documents = []
    for file_path in folder_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                documents.append(content)
    return documents


def _create_embeddings(documents: List[str]) -> np.ndarray:
    """Create embeddings for documents."""
    model = _load_model()
    embeddings = model.encode(documents)
    return embeddings.astype("float32")


def build_index(raw_folder: Path):
    """Build FAISS index from raw folder."""
    documents = _read_text_files(raw_folder)
    if not documents:
        raise ValueError("No text files found in raw folder")
    
    print(f"Found {len(documents)} text files in {raw_folder}")
    
    embeddings = _create_embeddings(documents)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save index and documents
    faiss.write_index(index, FAISS_INDEX)
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)
    
    print(f"Index built with {len(documents)} documents")
    print(f"FAISS index saved to: {FAISS_INDEX}")
    print(f"Documents saved to: {DOCUMENTS_PATH}")


def search_similar(query: str, top_k: int = 3) -> List[str]:
    """Search for similar documents."""
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(DOCUMENTS_PATH):
        raise FileNotFoundError("FAISS index not found. Run build first")
    
    # Load index and documents
    index = faiss.read_index(FAISS_INDEX)
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)
    
    # Create query embedding
    model = _load_model()
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    # Return similar documents
    similar_docs = []
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):
            similar_docs.append(documents[idx])
    
    return similar_docs
