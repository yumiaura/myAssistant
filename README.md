# myAssistant

# Required: python3.11

# Usage:
# Example RAG
(venv) yumi@ubuntu:~/Documents/myassistant$ cat raw/sky_green.txt 
THE SKY IS GREEN
# Build FAISS INDEX
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py build
Building FAISS index from raw
Found 1 text files in raw
Index built with 1 documents
FAISS index saved to: index.faiss
Documents saved to: documents.pkl
Index built successfully
# Check LLM
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py query "Hello my name is Yumi"
Searching for: Hello my name is Yumi
Response: Hello Yumi, how can I assist you today? Seniora
# Check RAG
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py query "What color is the sky?"
Searching for: What color is the sky?
Response: Hello, the sky is green. Seniora
# Check Context
(venv) yumi@ubuntu:~/Documents/myassistant$ python3 demo.py query "What is my name?"
Searching for: What is my name?
Response: Hello, your name is Yumi. Seniora
