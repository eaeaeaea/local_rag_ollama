#!/usr/bin/env python3
# Core: config, IO, embeddings/chat, FAISS index, retrieval.

import os, json, math, csv
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

# ---------- Optional deps (fail fast with clear message) ----------
try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit(f"FAISS not installed. pip install: pip install faiss-cpu | {e}")

try:
    from pypdf import PdfReader  # type: ignore
except Exception as e:
    raise SystemExit(f"pypdf not installed. pip install: pip install pypdf | {e}")

# ----------------------- Config -----------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL",   "llama3.1:8b")
EMB_MODEL   = os.environ.get("EMBED_MODEL", "nomic-embed-text")

BASE_DIR      = os.environ.get("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(BASE_DIR, "artifacts"))
INDEX_PATH    = os.path.join(ARTIFACTS_DIR, "faiss.index")
META_PATH     = os.path.join(ARTIFACTS_DIR, "meta.json")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

ALLOWED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".csv"}

TOP_K_DEFAULT   = int(os.environ.get("TOP_K", "5"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LLM_NUM_CTX     = int(os.environ.get("LLM_NUM_CTX", "8192"))

SYSTEM_PROMPT_RAG = (
    "You answer strictly from the provided context. "
    "If the context is insufficient, say you don't have enough information. "
    "Cite sources inline like [filename#chunkN]. Be concise."
)
SYSTEM_PROMPT_BARE = "You are a helpful assistant. Answer concisely. If unsure, say so."

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ----------------------- Embedding & Chat -----------------------
def _l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def ollama_embed(text: str, model: str = EMB_MODEL):
    url = f"{OLLAMA_HOST}/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Embeddings error: {r.text}")
    data = r.json()
    emb = data.get("embedding")
    if not emb:
        raise RuntimeError("Embeddings response missing 'embedding'")
    return emb

def ollama_chat(system_prompt: str, user_prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "options": {"temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Chat error: {r.text}")
    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()

# ----------------------- Document Loading -----------------------
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return ""

def read_csv_as_rows(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=1):
            # Join all key-value pairs dynamically
            row_text = "; ".join(f"{k}:{v}" for k, v in r.items())
            src_id = f"{path}#row{i}"
            rows.append((src_id, row_text))
    return rows

def chunk_text(text: str, chunk_size: int, overlap: int):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def load_documents_from_dir(data_dir: str):
    docs = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            p = os.path.join(root, fn)
            ext = os.path.splitext(p)[1].lower()
            if ext not in ALLOWED_EXTS:
                continue
            if ext == ".pdf":
                text = read_pdf(p).strip()
                if text:
                    docs.append((p, text))
            elif ext == ".csv":
                docs.extend(read_csv_as_rows(p))
            else:
                text = read_txt(p).strip()
                if text:
                    docs.append((p, text))
    return docs

# ----------------------- Index Build / Load -----------------------
_index = None
_meta: List[Dict[str, Any]] = []

def index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)

def index_loaded() -> bool:
    return _index is not None and bool(_meta)

def meta_len() -> int:
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return len(json.load(f))
    except Exception:
        return 0

def save_index(index, meta):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index_meta():
    global _index, _meta
    if not index_exists():
        raise RuntimeError("Index not found. Build it first from the UI.")
    _index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        _meta = json.load(f)

def ensure_loaded():
    if not index_loaded():
        load_index_meta()

def build_index_from_data(data_dir: str, chunk_size: int, overlap: int, embed_model: str):
    docs = load_documents_from_dir(data_dir)
    if not docs:
        raise RuntimeError(f"No supported documents found under {data_dir}")

    chunks = []
    for src, text in docs:
        if src.endswith(".csv") or "#row" in src:
            chunks.append({"source": src, "chunk_id": 0, "text": text})
        else:
            parts = chunk_text(text, chunk_size, overlap)
            for i, part in enumerate(parts):
                chunks.append({"source": src, "chunk_id": i, "text": part})

    vectors = []
    for c in chunks:
        v = ollama_embed(c["text"], model=embed_model)
        v = _l2_normalize(v)
        vectors.append(v)

    xb = np.array(vectors, dtype="float32")
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
    index.add(xb)

    save_index(index, chunks)
    load_index_meta()  # refresh in-memory
    return {"docs": len(docs), "chunks": len(chunks), "dim": dim}

# ----------------------- Retrieval -----------------------
def retrieve(query: str, top_k: int):
    ensure_loaded()
    qv = ollama_embed(query)
    qv = _l2_normalize(qv)
    xq = np.array([qv], dtype="float32")
    scores, idxs = _index.search(xq, top_k)
    hits = []
    for i, s in zip(idxs[0], scores[0]):
        if i == -1:
            continue
        item = dict(_meta[i])
        item["score"] = float(s)
        hits.append(item)
    return hits

def build_context(hits):
    blocks = []
    for h in hits:
        src = h.get("source", "unknown")
        cidx = h.get("chunk_id", 0)
        text = h.get("text", "")
        header = f"[{os.path.basename(str(src))}#chunk{cidx}] (score={h.get('score',0):.3f})"
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)

def list_data_files():
    files = []
    for root, _, fns in os.walk(DATA_DIR):
        for fn in fns:
            p = os.path.join(root, fn)
            files.append({"path": os.path.relpath(p, DATA_DIR), "bytes": os.path.getsize(p)})
    return files
