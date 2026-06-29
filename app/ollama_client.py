"""Ollama client for querying language models."""

import json
import re

import requests
from app.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

# Reasoning models such as qwen3 may wrap chain-of-thought in <think>...</think>
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model response."""
    return THINK_BLOCK_RE.sub("", text).strip()


def query_ollama(prompt: str) -> str:
    """Query Ollama, streaming the answer to stdout as it is generated.

    Returns the full response text with reasoning blocks removed.
    """
    url = OLLAMA_URL
    model = OLLAMA_MODEL

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    chunks = []
    try:
        with requests.post(
            f"{url}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT, stream=True
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                token = json.loads(line).get("response", "")
                if token:
                    print(token, end="", flush=True)
                    chunks.append(token)
            print()
    except requests.exceptions.RequestException as e:
        message = f"Error querying Ollama: {e}"
        print(message)
        return message

    return strip_reasoning("".join(chunks))
