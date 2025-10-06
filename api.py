#!/usr/bin/env python3
"""
FastAPI app (routes). Static UI served from ./static/index.html
"""

import os, time, shutil
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import core  # local module

app = FastAPI(title="Local RAG Comparator UI + Index Builder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve ./static under /static
app.mount("/static", StaticFiles(directory=core.STATIC_DIR), name="static")

# ----------------------- UI -----------------------
@app.get("/")
def index_page():
    html_path = os.path.join(core.STATIC_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h3>UI not found</h3><p>Expected: static/index.html next to api.py.</p>", status_code=404)
    return FileResponse(html_path)

# ----------------------- API -----------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "index_loaded": core.index_loaded(),
        "index_exists": core.index_exists(),
        "meta_len": core.meta_len(),
        "paths": {"data": core.DATA_DIR, "index": core.INDEX_PATH, "meta": core.META_PATH},
        "models": {"llm": core.LLM_MODEL, "embed": core.EMB_MODEL},
    }

@app.get("/api/list")
def list_files():
    files = core.list_data_files()
    return {"count": len(files), "files": files}

@app.delete("/api/data")
def clear_data():
    if os.path.isdir(core.DATA_DIR):
        shutil.rmtree(core.DATA_DIR)
    os.makedirs(core.DATA_DIR, exist_ok=True)
    if os.path.isdir(core.ARTIFACTS_DIR):
        shutil.rmtree(core.ARTIFACTS_DIR)
    os.makedirs(core.ARTIFACTS_DIR, exist_ok=True)
    return {"cleared": True}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for uf in files:
        name = os.path.basename(uf.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if not name or ext not in core.ALLOWED_EXTS:
            continue
        dest = os.path.join(core.DATA_DIR, name)
        content = await uf.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved.append({"name": name, "bytes": len(content)})
    if not saved:
        raise HTTPException(status_code=400, detail="No files saved (empty selection or unsupported extensions)")
    return {"saved": saved, "data_dir": core.DATA_DIR}

@app.post("/api/build")
def api_build(chunk_size: int = Form(1200), overlap: int = Form(200), embed_model: str = Form(core.EMB_MODEL)):
    try:
        t0 = time.time()
        stats = core.build_index_from_data(core.DATA_DIR, chunk_size, overlap, embed_model)
        t1 = time.time()
        return {"ok": True, "stats": stats, "ms": int((t1 - t0) * 1000)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def api_query(payload: Dict[str, Any]):
    try:
        question = (payload.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="'question' is required")
        top_k = int(payload.get("top_k") or core.TOP_K_DEFAULT)

        t0 = time.time()
        hits = core.retrieve(question, top_k)
        ctx = core.build_context(hits)
        t1 = time.time()

        rag_user = f"Question:\n{question}\n\nContext:\n{ctx}"
        rag_ans = core.ollama_chat(core.SYSTEM_PROMPT_RAG, rag_user)
        t2 = time.time()

        bare_ans = core.ollama_chat(core.SYSTEM_PROMPT_BARE, question)
        t3 = time.time()

        return {
            "rag": {"answer": rag_ans, "chunks": hits},
            "llm": {"answer": bare_ans},
            "latency_ms": {
                "retrieval": int((t1 - t0) * 1000),
                "rag": int((t2 - t1) * 1000),
                "llm": int((t3 - t2) * 1000),
                "total": int((t3 - t0) * 1000),
            },
            "used": {"model": core.LLM_MODEL, "embed": core.EMB_MODEL, "top_k": top_k},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
