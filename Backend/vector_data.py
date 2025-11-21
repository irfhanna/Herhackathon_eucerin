# Pseudocode: uses openai python client
from openai import OpenAI
client = OpenAI(api_key="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")
import json

def embed_text(txt):
    resp = client.embeddings.create(model="text-embedding-3-small", input=txt)
    return resp.data[0].embedding

with open("data.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        text = doc["text"]
        # optionally chunk text here
        embedding = embed_text(text)
        # upsert to vector DB (example for Pinecone / Chroma below)
