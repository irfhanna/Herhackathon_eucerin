#!/usr/bin/env python3
"""
build_faiss_index_ollama.py
Usage:
  python build_faiss_index_ollama.py --input eucerin_products_full.jsonl --index_out eucerin_faiss_ollama.index --meta_out eucerin_metadata_ollama.jsonl
"""

import json
import argparse
from tqdm import tqdm
import faiss
import time
import numpy as np

# Ollama SDK
from ollama import Client

# Config
EMBED_MODEL = "mxbai-embed-large"  # adjust based on Ollama's available models
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
BATCH_SIZE = 32

EMBED_DIM = 1024  # adjust if your Ollama embedding model has different dim

client = Client()

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks

def get_embeddings_batch_ollama(texts):
    """
    Returns a list of embeddings from Ollama
    """
    embeddings = []
    for text in texts:
        resp = client.embed(model=EMBED_MODEL, input=text)
        embeddings.append(resp.embeddings[0])
    return embeddings

def main(args):
    docs = []
    next_id = 1

    print("Reading and chunking documents...")
    for product in read_jsonl(args.input):
        product_id = product.get("id") or product.get("product_name") or f"prod_{next_id}"
        product_name = product.get("product_name","")
        source = product.get("source","")
        text = product.get("text","")
        short = product.get("short_description","")
        chunks = chunk_text(text if text else short)
        if not chunks:
            continue
        for c in chunks:
            docs.append({
                "id": next_id,
                "product_id": product_id,
                "product_name": product_name,
                "text": c,
                "source": source,
                "meta": {
                    "key_ingredients": product.get("key_ingredients", []),
                    "problems_solved": product.get("problems_solved", []),
                    "category": product.get("category",""),
                }
            })
            next_id += 1

    print(f"Created {len(docs)} chunks from input.")

    if len(docs) == 0:
        print("No documents/chunks to index. Exiting.")
        return

    print("Creating embeddings in batches (Ollama)...")
    all_embeddings = [None] * len(docs)
    i = 0
    while i < len(docs):
        batch = docs[i:i+BATCH_SIZE]
        texts = [d["text"] for d in batch]
        try:
            emb = get_embeddings_batch_ollama(texts)
        except Exception as e:
            print("Embedding error:", e)
            print("Sleeping 5s and retrying...")
            time.sleep(5)
            continue
        for j, v in enumerate(emb):
            all_embeddings[i + j] = v
        i += BATCH_SIZE
        print(f"Embedded {min(i, len(docs))}/{len(docs)}")

    print("Building FAISS index...")
    xb = np.array(all_embeddings).astype("float32")
    faiss.normalize_L2(xb)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIDMap(index)
    ids = np.array([d["id"] for d in docs]).astype("int64")
    index.add_with_ids(xb, ids)
    print(f"FAISS index contains {index.ntotal} vectors")

    print("Saving FAISS index to", args.index_out)
    faiss.write_index(index, args.index_out)

    print("Writing metadata to", args.meta_out)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        for d in docs:
            out = {
                "id": d["id"],
                "product_id": d["product_id"],
                "product_name": d["product_name"],
                "text": d["text"],
                "source": d["source"],
                "meta": d["meta"]
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to eucerin_products_full.jsonl")
    p.add_argument("--index_out", default="eucerin_faiss_ollama.index", help="Path to save faiss index")
    p.add_argument("--meta_out", default="eucerin_metadata_ollama.jsonl", help="Path to save chunk metadata (jsonl)")
    args = p.parse_args()
    main(args)
