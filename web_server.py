from flask import Flask, render_template, request, jsonify
import os
import sys

# Import RAG logic from usecases
sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

# Load all usecase modules dynamically
USECASES = {
    "usecase1": {
        "name": "Employee Knowledge Base (HR)",
        "folder": "usecase1",
        "description": "Query HR policies, leave policies, work arrangements, and company guidelines.",
        "model": "phi3",
    },
    "usecase2": {
        "name": "Technical Documentation Support",
        "folder": "usecase2",
        "description": "Get answers about API endpoints, authentication, versioning, and error handling.",
        "model": "gemma3:1b",
    },
    "usecase3": {
        "name": "Customer Support Tickets",
        "folder": "usecase3",
        "description": "Find resolutions to common customer issues and troubleshooting steps.",
        "model": "gemma3:1b",
    },
    "usecase4": {
        "name": "Legal & Compliance Audit",
        "folder": "usecase4",
        "description": "Query contract clauses, termination terms, force majeure, and compliance obligations.",
        "model": "gemma3:1b",
    },
}


def load_rag_module(usecase_key):
    """Dynamically load RAG components from a usecase folder."""
    folder = USECASES[usecase_key]["folder"]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), folder))
    
    # Import the app module from the usecase folder
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"{usecase_key}_app",
        os.path.join(os.path.dirname(__file__), folder, "app.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.route("/")
def index():
    """Serve the main UI with all usecases."""
    return render_template("index.html", usecases=USECASES)


@app.route("/api/usecases")
def get_usecases():
    """Return list of available usecases."""
    return jsonify(USECASES)


@app.route("/api/query", methods=["POST"])
def query():
    """Process a query for a selected usecase."""
    data = request.json
    usecase_key = data.get("usecase")
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if usecase_key not in USECASES:
        return jsonify({"error": f"Invalid usecase: {usecase_key}"}), 400

    try:
        # Load the RAG module for this usecase
        uc_module = load_rag_module(usecase_key)

        # Load documents from the text file in the usecase folder
        folder = USECASES[usecase_key]["folder"]
        doc_filename = {
            "usecase1": "employee_knowledge_base.txt",
            "usecase2": "technical_documentation_support.txt",
            "usecase3": "customer_support_ticket_rag.txt",
            "usecase4": "legal_compliance_audit.txt",
        }[usecase_key]

        doc_path = os.path.join(os.path.dirname(__file__), folder, doc_filename)
        
        # Get chunk parameters per usecase (smaller chunks = more precise)
        chunk_params = {
            "usecase1": {"chunk_size": 350, "overlap": 80},
            "usecase2": {"chunk_size": 400, "overlap": 100},
            "usecase3": {"chunk_size": 350, "overlap": 80},
            "usecase4": {"chunk_size": 420, "overlap": 110},
        }
        
        params = chunk_params[usecase_key]
        docs = uc_module.load_text_file(doc_path, **params)

        if not docs:
            return jsonify({"error": "No documents loaded. Check file path."}), 500

        # Build RAG index
        rag = uc_module.SimpleTfidfRAG(docs)

        # Retrieve relevant chunks
        results = rag.retrieve(question, top_k=4 if usecase_key in ["usecase2", "usecase4"] else 3)

        if not results:
            return jsonify({
                "answer": "No matching context found.",
                "context": "",
                "usecase": usecase_key
            })

        # Combine context with scores
        context_parts = [f"[Relevance: {score:.2%}]\n{doc.text}" for doc, score in results]
        context = "\n\n".join(context_parts)

        # Get answer from model (or fallback)
        answer = None
        try:
            import ollama
            
            # Get the appropriate function and call model
            if usecase_key == "usecase1":
                answer = uc_module.ask_phi3(context, question)
            else:
                answer = uc_module.ask_model(context, question, USECASES[usecase_key]["model"])
            
            # If answer is empty, fallback
            if not answer or not answer.strip():
                raise ValueError("Model returned empty response")
                
        except Exception as e:
            print(f"[ERROR] Model fail for {usecase_key}: {str(e)}")
            # Fallback: simple keyword matching
            import re
            q_words = re.findall(r"\w+", question.lower())
            sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', context) if s.strip()]
            scored = []
            for s in sentences:
                score = sum(1 for w in q_words if w in s.lower())
                scored.append((score, s))
            scored.sort(reverse=True)
            top_sents = [s for sc, s in scored[:3] if sc > 0]
            if top_sents:
                answer = "Fallback answer (no model): " + " ".join(top_sents)
            else:
                answer = "Fallback answer (no model): I don't know."

        # Format results with scores for display
        context_with_scores = "\n\n".join([
            f"<strong>Match Score: {score:.1%}</strong>\n{doc.text}"
            for doc, score in results
        ])

        return jsonify({
            "answer": answer,
            "context": context_with_scores[:1500],
            "scores": [float(score) for _, score in results],
            "usecase": usecase_key
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
