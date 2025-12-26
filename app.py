from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

app = Flask(__name__)

# ----------------------------
# Gemini Setup
# ----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------
# Load CSV
# ----------------------------
df = pd.read_csv("qa_data (1).csv")
docs = [f"Q: {q}\nA: {a}" for q, a in zip(df.question, df.answer)]

# ----------------------------
# Embeddings (SMALL MODEL)
# ----------------------------
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
doc_embeddings = embedder.encode(docs)

# ----------------------------
# Simple RAG (Cosine Similarity)
# ----------------------------
def rag(query):
    q_emb = embedder.encode([query])[0]

    scores = np.dot(doc_embeddings, q_emb) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_emb)
    )

    best_idx = scores.argmax()
    context = docs[best_idx]

    prompt = f"""
Answer ONLY from the context below.
If not found, say: No relevant Q&A found.

Context:
{context}

Question: {query}
"""
    return model.generate_content(prompt).text.strip()

# ----------------------------
# Flask Route
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]
        answer = rag(query)
    return render_template("index.html", answer=answer)

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run()
