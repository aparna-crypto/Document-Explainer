import os
from functools import lru_cache
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL ="all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        chunks, start = [], 0
        while start < len(text):
             chunks.append(text[start:start+size])
             start+=size-overlap
        return chunks

def ingest_pdf(path):
    path = Path(path)
    reader = PdfReader(str(path))
    docs, metas, ids = [], [], []

    for page_num, page in enumerate(reader.pages, start=1):
         text = (page.extract_text() or "").strip()
         for i, chunk in enumerate(chunk_text(text)):
              docs.append(chunk)
              metas.append({"source": path.name, "page": page_num})
              ids.append(f"{path.name}-p{page_num}-c{i}")
    if not docs:
         return 0


    embeddings = get_embedder().encode(docs).tolist()
    get_collection().upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(docs)


def retrieve(question, k=5):
     q_emb = get_embedder().encode([question]).tolist()
     res = get_collection().query(query_embeddings=q_emb, n_results=k)
     return [
          {"text": d, "source": m["source"], "page": m["page"], "distance":dist}
          for d,m,dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
     ]

def ask(question,k=5):
    hits = retrieve(question, k)
    if not hits:
        return "No documents indexed yet,", []

    context = "\n\n".join(
        f"[{i}] ({h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )

    prompt = (
        "Answer the question using ONLY the numbered sources below. "
        "Cite sources inline like [1] or [2]. "
        "If the sources don't contain the answer, say you couldn't find it.\n\n"
        f"SOURCES:\n{context}\n\nQUESTION: {question}"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response['message']['content'], hits
