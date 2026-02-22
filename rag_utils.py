from dataclasses import dataclass
from typing import List
import re


@dataclass
class Document:
    id: str
    text: str
    source: str = ""


def load_and_chunk(filepath: str, chunk_size: int = 500, overlap: int = 100) -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize whitespace
    content = re.sub(r"\s+", " ", content).strip()

    if not content:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[Document] = []
    idx = 0
    i = 0
    while idx < len(content):
        chunk = content[idx: idx + chunk_size].strip()
        if len(chunk) > 40:
            chunks.append(Document(id=f"chunk_{i}", text=chunk, source=filepath))
            i += 1
        idx += step

    return chunks


from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleTfidfRAG:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        texts = [d.text for d in docs]
        if texts:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.doc_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.vectorizer = None
            self.doc_matrix = None

    def retrieve(self, query: str, top_k: int = 3):
        if not self.vectorizer or not self.doc_matrix:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idx:
            if sims[i] > 0:
                results.append((self.docs[i], float(sims[i])))
        return results
