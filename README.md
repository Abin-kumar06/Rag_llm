RAG Helpers and Usecase Apps

This workspace contains four usecases. Each `usecase*/app.py` now:

- Loads the local RAG text file in the same folder.
- Splits the document into overlapping character chunks using `rag_utils.load_and_chunk`.
- Uses `rag_utils.SimpleTfidfRAG` for retrieval (TF-IDF + cosine similarity).
- Calls `ollama.chat` if the `ollama` package/runtime is available; otherwise falls back to a simple heuristic reply.

Quick start (create a venv and install deps):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a usecase app:

```powershell
python usecase3\app.py
```

Notes:
- If you don't have `ollama` or a local model, the apps will still run and return a fallback answer synthesized from retrieved context.
- Chunk size and overlap are set per usecase; adjust in the respective `app.py` if needed.# Rag_llm