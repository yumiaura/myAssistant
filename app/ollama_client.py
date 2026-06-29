"""Ollama client for querying language models."""

import re

import requests
from app.config import OLLAMA_URL, OLLAMA_MODEL

# Reasoning models such as qwen3 may wrap chain-of-thought in <think>...</think>
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model response."""
    return THINK_BLOCK_RE.sub("", text).strip()


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
        return strip_reasoning(result.get("response", "No response received"))
    
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama: {e}"
