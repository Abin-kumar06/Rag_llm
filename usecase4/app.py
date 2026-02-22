from dataclasses import dataclass
from typing import List
import numpy as np
try:
    import ollama
except Exception:
    ollama = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    id: str
    text: str
    source: str = ""


def load_text_file(filepath, chunk_size: int = 700, overlap: int = 150):
    import os

    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        filepath = os.path.join(os.path.dirname(__file__), filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # normalize whitespace
    content = "\n\n".join([p.strip() for p in content.split("\n\n") if p.strip()])
    if not content:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    idx = 0
    while idx < len(content):
        chunk = content[idx: idx + chunk_size].strip()
        if len(chunk) > 20:
            chunks.append(Document(id=f"text_chunk_{i}", text=chunk, source=filepath))
            i += 1
        idx += step

    return chunks


class SimpleTfidfRAG:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def retrieve(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = np.argsort(sims)[::-1][:top_k]
        return [(self.docs[i], float(sims[i])) for i in idx if sims[i] > 0]


PROMPT_TEMPLATE = """
You are a legal compliance assistant. Provide precise answers and, when possible, quote or reference the relevant clause from the context.

Use ONLY the context below. If the information is not present, say "I don't know" and recommend next steps (e.g., consult full contracts or legal counsel).

Context:
{context}

Question:
{question}
"""


def ask_model(context: str, question: str, model_name: str = "gemma3:1b") -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    if ollama is None:
        import re
        q_words = re.findall(r"\w+", question.lower())
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', context) if s.strip()]
        scored = []
        for s in sentences:
            score = sum(1 for w in q_words if w in s.lower())
            scored.append((score, s))
        scored.sort(reverse=True)
        top = [s for sc, s in scored[:3] if sc > 0]
        if top:
            return "Fallback (no model): " + " \n".join(top)
        return "Fallback (no model): I don't know. Consult the full contract or legal counsel."

    resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


def main():
    print("=== RAG + Legal Compliance ===")

    docs = load_text_file("legal_compliance_audit.txt", chunk_size=420, overlap=110)
    rag = SimpleTfidfRAG(docs)

    while True:
        query = input("\nAsk Legal Question (type exit to quit): ")
        if query.lower() == "exit":
            break

        results = rag.retrieve(query, top_k=4)

        if not results:
            print("No matching context found.")
            continue

        context = "\n\n".join([doc.text for doc, _ in results])

        print("\n--- Retrieved Context ---")
        print(context[:900])

        print("\n--- Answer ---")
        answer = ask_model(context, query, model_name="gemma3:1b")
        print(answer)


if __name__ == "__main__":
    main()
