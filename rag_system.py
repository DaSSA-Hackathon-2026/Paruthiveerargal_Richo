"""
RAG System: PDF Ingestion + Pinecone Vector Store
─────────────────────────────────────────────────
Embeddings  : HuggingFace SentenceTransformers (all-MiniLM-L6-v2)
Reranking   : CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
Vector Store: Pinecone
"""

import os
import glob
import hashlib
import re
from typing import List, Dict, Any, Optional

import PyPDF2
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    PDF_FOLDER,
    HF_MODEL_NAME,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)

# CrossEncoder model for reranking — scores (query, passage) pairs directly
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to fetch before reranking (fetch more, keep best)
RETRIEVAL_CANDIDATES = TOP_K_RESULTS * 4


class RAGSystem:
    def __init__(self):
        print(f"[RAG] Loading bi-encoder: {HF_MODEL_NAME}")
        self.embedder = SentenceTransformer(HF_MODEL_NAME)

        print(f"[RAG] Loading cross-encoder reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

        print(f"[RAG] Connecting to Pinecone...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        print("[RAG] RAG System ready.")

    def _ensure_index(self):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            print(f"[RAG] Creating index '{PINECONE_INDEX_NAME}'...")
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
            )
        else:
            print(f"[RAG] Index '{PINECONE_INDEX_NAME}' already exists.")

    # ─── PDF Processing ───────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"[RAG] Error reading {pdf_path}: {e}")
        return text

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and remove junk characters."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x20-\x7E]', '', text)
        return text.strip()

    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Sliding window chunking with overlap.
        Each chunk has context from its neighbors to reduce boundary issues.
        """
        text = self._clean_text(text)
        words = text.split()
        chunks = []
        i = 0
        chunk_idx = 0

        while i < len(words):
            chunk_words = words[i : i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            chunk_id = hashlib.md5(f"{source}-{chunk_idx}".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "chunk_idx": chunk_idx,
            })
            i += CHUNK_SIZE - CHUNK_OVERLAP
            chunk_idx += 1

        return chunks

    def embed_and_upsert(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,   # cosine similarity works better normalized
        ).tolist()

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_idx": chunk["chunk_idx"],
                },
            })

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size])
        print(f"[RAG] Upserted {len(vectors)} vectors.")

    def ingest_folder(self, folder: str = PDF_FOLDER) -> Dict[str, Any]:
        os.makedirs(folder, exist_ok=True)
        pdf_files = glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True)
        pdf_files += glob.glob(os.path.join(folder, "*.pdf"))
        pdf_files = list(set(pdf_files))

        if not pdf_files:
            return {"status": "no_pdfs", "files": [], "total_chunks": 0}

        total_chunks = 0
        ingested_files = []

        for pdf_path in pdf_files:
            print(f"[RAG] Processing: {pdf_path}")
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"[RAG] No text extracted from {pdf_path}, skipping.")
                continue
            filename = os.path.basename(pdf_path)
            chunks = self.chunk_text(text, filename)
            self.embed_and_upsert(chunks)
            total_chunks += len(chunks)
            ingested_files.append({"file": filename, "chunks": len(chunks)})

        return {"status": "success", "files": ingested_files, "total_chunks": total_chunks}

    # ─── Retrieval Pipeline ───────────────────────────────────────────────────

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Stage 1: Fast approximate nearest-neighbor search via Pinecone."""
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        retrieved = []
        for match in results.matches:
            text = match.metadata.get("text", "") or match.metadata.get("content", "")
            source = match.metadata.get("source", "") or match.metadata.get("source_pdf", "unknown")
            if not text.strip():
                continue
            retrieved.append({
                "text": text,
                "source": source,
                "chunk_idx": match.metadata.get("chunk_idx", 0),
                "vector_score": round(match.score, 4),
                "rerank_score": None,
                "score": round(match.score, 4),
            })
        return retrieved

    def _rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Stage 2: CrossEncoder reranking.
        Scores each (query, passage) pair directly — much more accurate than vector similarity alone.
        """
        if not chunks:
            return chunks

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)
            # Combine vector + rerank score (weighted: 30% vector, 70% rerank)
            chunk["score"] = round(0.3 * chunk["vector_score"] + 0.7 * float(score), 4)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove near-duplicate chunks based on text overlap."""
        seen = []
        unique = []
        for chunk in chunks:
            text_key = chunk["text"][:120].lower().strip()
            if not any(text_key == s for s in seen):
                seen.append(text_key)
                unique.append(chunk)
        return unique

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline:
        1. Vector search (fetch candidates = top_k * 4)
        2. Deduplicate
        3. CrossEncoder rerank
        4. Return top_k best results
        """
        # Stage 1: Fetch more candidates than needed
        candidates = self._vector_search(query, top_k=RETRIEVAL_CANDIDATES)
        print(f"[RAG] Vector search returned {len(candidates)} candidates")

        # Stage 2: Deduplicate
        candidates = self._deduplicate(candidates)
        print(f"[RAG] After dedup: {len(candidates)} candidates")

        if not candidates:
            return []

        # Stage 3: Rerank with CrossEncoder
        if rerank and len(candidates) > 0:
            results = self._rerank(query, candidates, top_k=top_k)
            print(f"[RAG] After reranking: top {len(results)} chunks selected")
        else:
            results = sorted(candidates, key=lambda x: x["vector_score"], reverse=True)[:top_k]

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "index_name": PINECONE_INDEX_NAME,
            "dimension": EMBEDDING_DIM,
            "bi_encoder": HF_MODEL_NAME,
            "cross_encoder": RERANKER_MODEL,
        }