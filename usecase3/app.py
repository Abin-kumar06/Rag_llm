from rag_utils import load_and_chunk, SimpleTfidfRAG
import os


PROMPT_TEMPLATE = """
You are a customer support assistant whose job is to provide concise, actionable troubleshooting steps and clear escalation instructions.

Answer using ONLY the context below. If the context does not contain the answer, say "I don't know" and provide safe next steps (diagnostics to run or what information to collect).

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
        return "Fallback answer (no model): I don't know. Collect logs, browser/version info, and steps to reproduce."


def main():
    base = os.path.dirname(__file__)
    filepath = os.path.join(base, "customer_support_ticket_rag.txt")
    docs = load_and_chunk(filepath, chunk_size=500, overlap=120)
    rag = SimpleTfidfRAG(docs)

    print("=== Customer Support Knowledge Assistant ===")
    if not docs:
        print("No documents found in", filepath)
        return

    while True:
        query = input("\nAsk Support Question (type exit to quit): ")
        if query.lower() == "exit":
            break

        results = rag.retrieve(query, top_k=3)
        if not results:
            print("No matching context found.")
            continue

        context = "\n\n".join([f"[source: {r[0].source}] {r[0].text}" for r in results])

        print("\n--- Retrieved Context (truncated) ---")
        print(context[:800])

        print("\n--- Answer ---")
        answer = ask_model(context, query, model_name="gemma3:1b")
        print(answer)


if __name__ == "__main__":
    main()
