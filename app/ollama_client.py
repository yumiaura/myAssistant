"""LLM client backed by pydantic-ai, talking to Ollama's OpenAI-compatible API."""

import asyncio
import re

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from app.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

# Reasoning models such as qwen3 may wrap chain-of-thought in <think>...</think>
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

AGENT_CACHE = None


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model response."""
    return THINK_BLOCK_RE.sub("", text).strip()


def build_agent() -> Agent:
    """Construct (once) the pydantic-ai agent pointing at the Ollama server."""
    global AGENT_CACHE
    if AGENT_CACHE is None:
        model = OllamaModel(
            OLLAMA_MODEL,
            provider=OllamaProvider(base_url=f"{OLLAMA_URL}/v1"),
        )
        AGENT_CACHE = Agent(model, model_settings=ModelSettings(timeout=OLLAMA_TIMEOUT))
    return AGENT_CACHE


async def stream_response(prompt: str) -> str:
    """Stream the answer to stdout as it is generated and return the full text."""
    agent = build_agent()
    chunks = []
    async with agent.run_stream(prompt) as result:
        async for delta in result.stream_text(delta=True):
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return strip_reasoning("".join(chunks))


def query_ollama(prompt: str) -> str:
    """Query the model via pydantic-ai, streaming output, returning the full text."""
    try:
        return asyncio.run(stream_response(prompt))
    except Exception as error:
        message = f"Error querying Ollama: {error}"
        print(message)
        return message
