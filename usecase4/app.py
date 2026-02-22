from rag_utils import load_and_chunk, SimpleTfidfRAG
import os


PROMPT_TEMPLATE = """
You are a legal compliance assistant. Provide precise answers and, when possible, quote or reference the relevant clause from the context.

Use ONLY the context below. If the information is not present, say "I don't know" and recommend next steps (e.g., consult full contracts or legal counsel).

Context:
{context}

Question:
{question}
"""


def ask_model(context: str, question: str, model_name: str = "gemma3:1b") -> str:
    try:
        import ollama
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]
    except Exception:
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
            return "Fallback answer (no model): " + " \n".join(top)
        return "Fallback answer (no model): I don't know. Consult the full contract or legal counsel."


def main():
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, "legal_compliance_audit.txt")
    docs = load_and_chunk(filepath, chunk_size=700, overlap=150)
    rag = SimpleTfidfRAG(docs)

    print("=== Legal Compliance Assistant ===")
    if not docs:
        print("No documents found in", filepath)
        return

    while True:
        query = input("\nAsk Legal Question (type exit to quit): ")
        if query.lower() == "exit":
            break

        results = rag.retrieve(query, top_k=4)
        if not results:
            print("No matching context found.")
            continue

        context = "\n\n".join([f"[source: {r[0].source}] {r[0].text}" for r in results])

        print("\n--- Retrieved Context (truncated) ---")
        print(context[:900])

        print("\n--- Answer ---")
        answer = ask_model(context, query, model_name="gemma3:1b")
        print(answer)


if __name__ == "__main__":
    main()
