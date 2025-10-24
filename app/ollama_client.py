"""Ollama client for querying language models."""

import requests
from app.config import OLLAMA_URL, OLLAMA_MODEL


def query_ollama(prompt: str) -> str:
    """Query Ollama with the given prompt."""
    url = OLLAMA_URL
    model = OLLAMA_MODEL
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(f"{url}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "No response received")
    
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama: {e}"
