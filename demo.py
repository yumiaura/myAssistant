#!/usr/bin/env python3
"""Main script for FAISS vector search with Ollama integration."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from app.vector_store import build_index, search_similar
from app.ollama_client import query_ollama
from app.history import get_recent_history, save_interaction
from app.template import render_prompt
from app.config import FAISS_INDEX, RAW_FOLDER


def build_command(args):
    """Build FAISS index from raw folder."""
    raw_path = Path(RAW_FOLDER)
    if not raw_path.exists():
        print("Error: raw folder not found")
        sys.exit(1)
    
    print(f"Building FAISS index from {raw_path}")
    build_index(raw_path)
    print("Index built successfully")


def query_command(args):
    """Query the FAISS index and Ollama."""
    if not Path(FAISS_INDEX).exists():
        print("Error: FAISS index not found. Run 'build' first")
        sys.exit(1)
    
    query_text = args.text
    print(f"Searching for: {query_text}")
    
    # Search similar documents
    similar_docs = search_similar(query_text, top_k=3)
    
    # Get recent history
    history = get_recent_history(10)
    
    # Prepare context for template
    context = {
        "current_date": datetime.now().isoformat(),
        "user_query": query_text,
        "similar_documents": similar_docs,
        "history": history
    }
    
    # Render prompt template
    prompt = render_prompt(context)
    
    # Query Ollama
    response = query_ollama(prompt)
    
    # Save interaction to history
    save_interaction(query_text, response)
    
    print(f"Response: {response}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FAISS vector search with Ollama")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build FAISS index from raw folder")
    build_parser.set_defaults(func=build_command)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("text", help="Query text")
    query_parser.set_defaults(func=query_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
