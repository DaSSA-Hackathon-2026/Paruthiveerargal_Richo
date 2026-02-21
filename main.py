"""
FastAPI Backend for Multi-Agent RAG System
"""

import os
import re
import shutil
from contextlib import asynccontextmanager
from typing import Optional

import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agents import MultiAgentOrchestrator, _query_cache_entries, _query_cache_lock, _query_cache_clear, _query_cache_list, SEMANTIC_SIMILARITY_THRESHOLD
from config import PDF_FOLDER
from rag_system import RAGSystem

# ─── App Lifespan ──────────────────────────────────────────────────────────────

rag: Optional[RAGSystem] = None
orchestrator: Optional[MultiAgentOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, orchestrator
    print("[Startup] Initializing RAG system...")
    rag = RAGSystem()
    orchestrator = MultiAgentOrchestrator(rag)
    print("[Startup] System ready.")
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Multi-Agent RAG API",
    description="RAG system with Query Analysis, Retrieval, Generation & Validation agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class IngestResponse(BaseModel):
    status: str
    files: list
    total_chunks: int
    message: str


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Multi-Agent RAG API is running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "rag_ready": rag is not None}


@app.get("/stats")
async def get_stats():
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag.get_index_stats()


# ─── Cache Routes ───────────────────────────────────────────────────────────────

@app.get("/cache/stats")
async def cache_stats():
    """
    Returns how many questions are currently cached and lists them.
    Shows the similarity threshold used for semantic matching.
    """
    entries = _query_cache_list()
    return {
        "cached_questions": len(entries),
        "max_size": 128,
        "similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
        "questions": [e["query"] for e in entries],
    }


@app.delete("/cache/clear")
async def cache_clear():
    """
    Clears the entire query answer cache.
    Call this after re-ingesting new documents so stale answers are removed.
    """
    count = _query_cache_clear()
    return {
        "status": "cleared",
        "entries_removed": count,
        "message": f"Removed {count} cached answers. All future queries will be freshly generated.",
    }


@app.delete("/cache/clear/{query}")
async def cache_clear_one(query: str):
    """
    Clears the cache for a single specific question by exact query text match.
    """
    with _query_cache_lock:
        before = len(_query_cache_entries)
        _query_cache_entries[:] = [e for e in _query_cache_entries if e["query"].lower().strip() != query.lower().strip()]
        after = len(_query_cache_entries)
    removed = before - after
    return {
        "status": "cleared" if removed > 0 else "not_found",
        "query": query,
        "entries_removed": removed,
    }


# ─── Ingest & Upload ────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_pdfs():
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    result = rag.ingest_folder()
    # Clear query cache after re-ingesting so stale answers don't persist
    _query_cache_clear()
    return IngestResponse(
        status=result["status"],
        files=result["files"],
        total_chunks=result["total_chunks"],
        message=f"Ingested {len(result['files'])} files with {result['total_chunks']} chunks. Query cache cleared.",
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    os.makedirs(PDF_FOLDER, exist_ok=True)
    save_path = os.path.join(PDF_FOLDER, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = rag.extract_text_from_pdf(save_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    chunks = rag.chunk_text(text, file.filename)
    rag.embed_and_upsert(chunks)

    # Clear cache — new document may change existing answers
    _query_cache_clear()

    return {
        "status": "success",
        "file": file.filename,
        "chunks": len(chunks),
        "message": f"Uploaded and ingested '{file.filename}' with {len(chunks)} chunks. Query cache cleared.",
    }


@app.post("/query")
async def query(request: QueryRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = orchestrator.run(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_query(request: QueryRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator.query_agent.analyze(request.query)


@app.post("/retrieve")
async def retrieve(request: QueryRequest):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator.retrieval_agent.retrieve([request.query])


# ─── PDF Serving ───────────────────────────────────────────────────────────────

def _find_pdf(filename: str) -> str:
    """Search for a PDF in PDF_FOLDER and its subdirectories."""
    safe_name = os.path.basename(filename)
    search_roots = [PDF_FOLDER, "./pdfs", "."]
    for root in search_roots:
        direct = os.path.join(root, safe_name)
        if os.path.exists(direct):
            return direct
        for dirpath, _, files in os.walk(root):
            if safe_name in files:
                return os.path.join(dirpath, safe_name)
    return None


@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    safe_name = os.path.basename(filename)
    pdf_path = _find_pdf(safe_name)
    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{safe_name}' not found. Check that PDF_FOLDER in config.py points to the correct directory (currently: {PDF_FOLDER})"
        )
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={safe_name}"},
    )


@app.get("/pdf/{filename}/highlight")
async def get_highlight_info(filename: str, text: str, chunk_idx: int = 0):
    safe_name = os.path.basename(filename)
    pdf_path = _find_pdf(safe_name)
    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"PDF '{safe_name}' not found. Check PDF_FOLDER in config.py (currently: {PDF_FOLDER})"
        )

    search_snippet = text[:200].strip()
    search_clean   = re.sub(r'\s+', ' ', search_snippet).lower()

    found_page  = 1
    found_rects = []
    page_height = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text  = page.extract_text() or ""
                page_clean = re.sub(r'\s+', ' ', page_text).lower()

                if search_clean[:80] in page_clean:
                    found_page  = page_num
                    page_height = float(page.height)
                    words       = page.extract_words()

                    # First 12 meaningful words from snippet (len > 2)
                    snippet_words = {
                        w.lower() for w in search_snippet.split()
                        if len(w) > 2
                    }
                    snippet_words = set(list(snippet_words)[:12])

                    rects = []
                    for word in words:
                        cleaned = word["text"].lower().strip('.,;:()[]"\''')')
                        if cleaned in snippet_words:
                            rects.append({
                                "x1":   round(word["x0"],     2),
                                "y1":   round(word["top"],    2),
                                "x2":   round(word["x1"],     2),
                                "y2":   round(word["bottom"], 2),
                                "page": page_num,
                            })
                        if len(rects) >= 40:
                            break

                    found_rects = rects
                    break

    except Exception as e:
        print(f"[Highlight] Error: {e}")

    return {
        "filename":    safe_name,
        "page":        found_page,
        "page_height": page_height,
        "rects":       found_rects,
        "search_text": search_snippet[:120],
        "pdf_url":     f"/pdf/{safe_name}",
    }


@app.get("/pdfs")
async def list_pdfs():
    os.makedirs(PDF_FOLDER, exist_ok=True)
    files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    return {"files": files, "folder": PDF_FOLDER}