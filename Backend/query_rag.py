#!/usr/bin/env python3
"""
query_rag_ollama.py
Usage:
  python query_rag_ollama.py --index eucerin_faiss.index --meta eucerin_metadata.jsonl --query "Which Eucerin products contain urea for very dry skin?" --top_k 5
"""

import json
import argparse
import faiss
import numpy as np
from ollama import Client

# Config
EMBED_MODEL = "mxbai-embed-large"      # Ollama embedding model (must be pulled)
CHAT_MODEL = "llama2:7b"          # Ollama chat/completion model
client = Client()

def load_metadata(meta_path):
    md = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            md[int(obj["id"])] = obj
    return md

def embed_query(text):
    resp = client.embed(model=EMBED_MODEL, input=text)
    emb = np.array(resp.embeddings[0]).astype("float32")
    emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb[0]


def retrieve(index_path, meta_path, query, top_k=5):
    index = faiss.read_index(index_path)
    metadata = load_metadata(meta_path)
    q_emb = embed_query(query)
    D, I = index.search(np.expand_dims(q_emb, axis=0), top_k)
    ids = I[0].tolist()
    scores = D[0].tolist()
    hits = []
    for idx, score in zip(ids, scores):
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

def build_prompt(question, hits):
    ctxs = []
    for h in hits:
        txt = h["text"]
        src = h.get("source","")
        ctxs.append(f"Source: {src}\nProduct: {h.get('product_name')}\nExcerpt: {txt}\n")
    context_block = "\n\n---\n\n".join(ctxs)
    prompt = f"""You are a helpful assistant that answers questions about Eucerin products using the provided context. Use the context to cite product pages and ingredients. If answer cannot be found in the context, say you don't know and suggest checking the product pages.

Context:
{context_block}

Question: {question}

Answer concisely and list the source URLs you used at the end under 'Sources:'.
"""
    return prompt

def call_chat_model(prompt):
    resp = client.generate(
        model=CHAT_MODEL,
        prompt=prompt
        
    )
    return resp.response


def main(args):
    hits = retrieve(args.index, args.meta, args.query, top_k=args.top_k)
    if not hits:
        print("No relevant documents found.")
        return
    prompt = build_prompt(args.query, hits)
    print("=== Prompt sent to model ===")
    print(prompt[:2000] + ("\n\n...TRUNCATED...\n" if len(prompt) > 2000 else "\n"))
    print("=== Retrieving answer from model ===")
    answer = call_chat_model(prompt)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Retrieved chunks (top_k) ===")
    for h in hits:
        print(f"- id:{h['id']} score:{h['score']:.4f} product:{h['product_name']} source:{h['source']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    main(args)
