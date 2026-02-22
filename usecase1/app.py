from dataclasses import dataclass
from typing import List
import numpy as np
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    id: str
    text: str
    source: str = ""


def load_text_file(filepath, chunk_size: int = 500, overlap: int = 100):
    import os

    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        # try relative to this script
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


PROMPT = f"""
You are an internal HR assistant. Answer succinctly using ONLY the context provided below. If the answer is not present, say "I don't know" and advise which internal resource or person to contact.

Context:
{{context}}

Question:
{{question}}
"""


def ask_phi3(context, question):
    prompt = PROMPT.format(context=context, question=question)

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def main():
    print("=== RAG + Phi3 Employee KB ===")

    docs = load_text_file("employee_knowledge_base.txt", chunk_size=350, overlap=80)
    rag = SimpleTfidfRAG(docs)

    while True:
        query = input("\nAsk Question (type exit to quit): ")
        if query.lower() == "exit":
            break

        results = rag.retrieve(query)

        if not results:
            print("No matching context found.")
            continue

        context = "\n\n".join([doc.text for doc, _ in results])

        print("\n--- Retrieved Context ---")
        print(context[:500])

        print("\n--- Phi3 Answer ---")
        answer = ask_phi3(context, query)
        print(answer)


if __name__ == "__main__":
    main()
