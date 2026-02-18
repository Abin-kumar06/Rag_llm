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

def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 20]

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            id=f"text_chunk_{i}",
            text=chunk,
            source="employee_knowledge_base.txt"
        ))
    return docs

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

def ask_phi3(context, question):
    prompt = f"""
You are a temple knowledge assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def main():
    print("=== RAG + Phi3 Temple Search ===")

    docs = load_text_file("Templesofondipudur.txt")
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
