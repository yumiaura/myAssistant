# myAssistant

# Installation
Required: python3.11
```
pip3 install -r requirements.txt
```

# Prerequisites
[Ollama](https://ollama.com) must be running locally with the default model pulled:
```
ollama pull qwen3:8b
```

# Configuration
Settings are read from environment variables (a `.env` file is loaded automatically).
Copy the template and adjust as needed:
```
cp env_template.txt .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:8b` | Model used for generation |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer for embeddings |
| `SIMILARITY_THRESHOLD` | `0.25` | Min cosine score to keep a retrieved document |
| `TOP_K` | `3` | Number of documents retrieved per query |
| `RAW_FOLDER` | `raw` | Folder with source `.txt` files |
| `FAISS_INDEX` | `index.faiss` | Path to the saved FAISS index |
| `DOCUMENTS_PATH` | `documents.pkl` | Path to the pickled documents |
| `HISTORY_FILE` | `history.txt` | Conversation history file |
| `PROMPT_TEMPLATE` | `PROMPT.j2` | Jinja2 prompt template |

# Usage:
# Example RAG
```
(venv) yumi@ubuntu:~/Documents/myassistant$ cat raw/sky_green.txt
THE SKY IS GREEN
```
# Build FAISS INDEX
```
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py build
Building FAISS index from raw
Found 1 text files in raw
Index built with 1 documents
FAISS index saved to: index.faiss
Documents saved to: documents.pkl
Index built successfully
```
# Check LLM
```
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py query "Hello my name is Yumi"
Searching for: Hello my name is Yumi
Response: Hello Yumi, how can I assist you today? Seniora
```
# Check RAG
```
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py query "What color is the sky?"
Searching for: What color is the sky?
Response: Hello, the sky is green. Seniora
```
# Check Context
```
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py query "What is my name?"
Searching for: What is my name?
Response: Hello, your name is Yumi. Seniora
```
