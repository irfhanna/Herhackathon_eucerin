#!/usr/bin/env python3
"""
Improved query_rag_ollama.py

New features:
- Performs standard RAG retrieval for Eucerin products.
- Also performs "Product Improvement Reasoning" using external insights.

Usage:
  python query_rag_ollama.py --index eucerin_faiss.index --meta eucerin_metadata.jsonl --query "Which Eucerin products contain urea for very dry skin?" --top_k 5 --insights "Dry skin barrier damaged, low ceramides, needs moisture-locking agents"
"""

import json
import argparse
import faiss
import numpy as np
from ollama import Client

EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "llama2:7b"    # more capable than llama2
client = Client()


# -----------------------------
# Load Metadata
# -----------------------------
def load_metadata(meta_path):
    md = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            md[int(obj["id"])] = obj
    return md


# -----------------------------
# Embedding
# -----------------------------
def embed_query(text):
    resp = client.embed(model=EMBED_MODEL, input=text)
    emb = np.array(resp.embeddings[0]).astype("float32")
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb[0]


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(index_path, meta_path, query, top_k=5):
    index = faiss.read_index(index_path)
    metadata = load_metadata(meta_path)
    q_emb = embed_query(query)
    D, I = index.search(np.expand_dims(q_emb, axis=0), top_k)

    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        m = metadata.get(int(idx))
        if m:
            hits.append({
                "id": idx,
                "score": float(score),
                "product_id": m.get("product_id"),
                "product_name": m.get("product_name"),
                "text": m.get("text"),
                "source": m.get("source"),
                "meta": m.get("meta")
            })
    return hits


# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(question, hits, insights=None):
    ctxs = []
    for h in hits:
        txt = h["text"]
        src = h.get("source", "")
        ctxs.append(
            f"Source: {src}\nProduct: {h.get('product_name')}\nExcerpt: {txt}\n"
        )

    context_block = "\n\n---\n\n".join(ctxs)

    improvement_block = f"\nAdditional Insights:\n{insights}\n" if insights else ""

    prompt = f"""
You are a skincare marketing specialist RAG system specializing in Eucerin products.

FIRST TASK — Product Retrieval:
Use ONLY the context to answer the user's question.
If the answer cannot be found in the context, say "I don't know".
List matching products with citations.

SECOND TASK — Product Improvement Reasoning:
Using the extracted product ingredients AND the external insights provided,
determine:
1. What these products do well.
2. What could be improved.
3. How the product could be optimized.

THIRD TASK — Marketing Suggestions:
Based on the above reasoning, suggest marketing angles or product improvements
that would better address the user's concern.



Context:
{context_block}

{improvement_block}

User Question:
{question}

Return the answer in this format:

### Answer
<normal RAG answer>

### Product Improvement Suggestions
<improvement suggestions based on insights + product data>

### Marketing Suggestions
<marketing suggestions based on insights + product data>

### Sources
<list URLs or metadata sources>
"""
    return prompt


# -----------------------------
# LLM call
# -----------------------------
def call_chat_model(prompt):
    resp = client.generate(
        model=CHAT_MODEL,
        prompt=prompt
    )
    return resp.response


# -----------------------------
# Main
# -----------------------------
def main(args):
    hits = retrieve(args.index, args.meta, args.query, top_k=args.top_k)
    if not hits:
        print("No relevant documents found.")
        return

    prompt = build_prompt(args.query, hits, insights=args.insights)

    print("=== Prompt Sent to Model ===")
    print(prompt[:1500] + ("\n\n...TRUNCATED...\n" if len(prompt) > 1500 else "\n"))

    print("=== Retrieving answer from model ===")
    answer = call_chat_model(prompt)

    print("\n=== Final Answer ===\n")
    print(answer)

    print("\n=== Retrieved chunks ===")
    for h in hits:
        print(f"- id:{h['id']} score:{h['score']:.4f} product:{h['product_name']} source:{h['source']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--insights", type=str, required=False, default=None,
                   help="Insights from graph analysis module (optional)")
    args = p.parse_args()
    main(args)
