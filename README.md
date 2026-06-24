# myAssistant

A small local RAG assistant built with Python, FAISS, sentence-transformers and
Ollama. Drop `.txt` files into `raw/`, build an index, then ask questions with
retrieved context plus short conversation history.

```text
  .-.
 (o.o)  local notes, local model
  |=|
```

## Requirements

- Python 3.11
- Ollama running locally
- An Ollama model pulled, for example:

```bash
ollama pull qwen2.5
```

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional configuration can be placed in `.env`:

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5
RAW_FOLDER=raw
FAISS_INDEX=index.faiss
DOCUMENTS_PATH=documents.pkl
HISTORY_FILE=history.txt
PROMPT_TEMPLATE=PROMPT.j2
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

## Usage

Add text files to `raw/`:

```text
THE SKY IS GREEN
```

Build the FAISS index:

```bash
python3 demo.py build
```

Ask a question:

```bash
python3 demo.py query "What color is the sky?"
```

Example output:

```text
Searching for: What color is the sky?
Response: Hello, the sky is green. Seniora
```

Ask about recent context:

```bash
python3 demo.py query "What is my name?"
```

## How it works

1. `demo.py build` reads `.txt` files from `raw/`.
2. `app/vector_store.py` embeds them with sentence-transformers.
3. FAISS stores the searchable vector index in `index.faiss`.
4. `demo.py query` retrieves similar documents and recent chat history.
5. `app/template.py` renders `PROMPT.j2`.
6. `app/ollama_client.py` sends the prompt to Ollama.

## Files created locally

- `index.faiss` — FAISS vector index
- `documents.pkl` — source document cache
- `history.txt` — recent conversation history

These files are local runtime artifacts and should not be committed.
