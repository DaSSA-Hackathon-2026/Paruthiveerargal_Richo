"""
ingest.py — Parse PDFs and store chunks in Pinecone.

Uses HuggingFace SentenceTransformers for embeddings, Gemini for generation.

Run ONCE (or when PDFs change):
    python ingest.py
    python ingest.py --folder ./my_pdfs
    python ingest.py --chunk-size 600
"""

import os
import re
import json
import hashlib
import logging
import warnings
import argparse
from pathlib import Path

import pdfplumber
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Silence pdfminer font noise ───────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*FontBBox.*")
warnings.filterwarnings("ignore", message=".*Cannot.*font.*")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PDF_FOLDER       = os.getenv("PDF_FOLDER",        "./pdfs")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",    "500"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY",  "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX",    "rag-anything")
PINECONE_NS      = os.getenv("PINECONE_NAMESPACE","")
PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD",    "aws")
PINECONE_REGION  = os.getenv("PINECONE_REGION",   "us-east-1")

# HuggingFace SentenceTransformer — free, local, no API key needed
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _make_id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', '', text)
    return text.strip()

def _table_to_markdown(table: list[list]) -> str:
    if not table or not table[0]:
        return ""
    rows = [[str(c or "").strip() for c in row] for row in table]
    header = rows[0]
    sep    = ["---"] * len(header)
    lines  = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep)    + " |",
    ]
    for row in rows[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")
    return "\n".join(lines)


# ─── EMBEDDING ────────────────────────────────────────────────────────────────

def _get_embedder() -> tuple:
    """
    Returns (embed_fn, vector_dim) using HuggingFace SentenceTransformers.
    Runs locally — no API key required.
    """
    log.info(f"Loading SentenceTransformer: {HF_MODEL_NAME} (dim={EMBEDDING_DIM})")
    model = SentenceTransformer(HF_MODEL_NAME)

    def embed_hf(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False).tolist()

    return embed_hf, EMBEDDING_DIM


# ─── PINECONE ─────────────────────────────────────────────────────────────────

def _init_pinecone(vector_dim: int):
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is not set.\n"
            "Get a free key at: https://app.pinecone.io"
        )
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = {idx.name: idx for idx in pc.list_indexes()}

    if PINECONE_INDEX in existing_indexes:
        existing_dim = existing_indexes[PINECONE_INDEX].dimension
        if existing_dim != vector_dim:
            log.warning(
                f"Index '{PINECONE_INDEX}' has dimension {existing_dim} "
                f"but embedder produces {vector_dim}. Deleting and recreating..."
            )
            pc.delete_index(PINECONE_INDEX)
            log.info("Old index deleted.")
        else:
            log.info(f"Using existing index '{PINECONE_INDEX}' (dim={existing_dim}).")
            if PINECONE_NS:
                log.info(f"Namespace: '{PINECONE_NS}'")
            return pc.Index(PINECONE_INDEX)

    log.info(f"Creating Pinecone index '{PINECONE_INDEX}' (dim={vector_dim})...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=vector_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    log.info("Index created.")
    return pc.Index(PINECONE_INDEX)


# ─── PDF PARSING ──────────────────────────────────────────────────────────────

def _parse_pdf(pdf_path: Path, chunk_size: int) -> tuple[list[dict], list[str]]:
    """
    Parse a single PDF into chunks.
    Each chunk: { id, type, title, content, metadata }
    """
    filename = pdf_path.name
    chunks:  list[dict] = []
    errors:  list[str]  = []

    log.info(f"▶  {filename}")

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_texts:  dict[int, str]  = {}
            page_tables: dict[int, list] = {}

            for page_num, page in enumerate(pdf.pages, start=1):

                # ── Tables ────────────────────────────────────────────────────
                try:
                    page_tables[page_num] = page.extract_tables() or []
                except Exception as e:
                    errors.append(f"{filename} p.{page_num} table: {e}")
                    page_tables[page_num] = []

                # ── Text (mask table regions) ──────────────────────────────────
                try:
                    table_objs = page.find_tables() or []
                    if table_objs:
                        filtered = page
                        for t in table_objs:
                            bb = t.bbox
                            filtered = filtered.filter(
                                lambda obj, b=bb: not (
                                    obj.get("x0",     0) >= b[0] and
                                    obj.get("x1",     0) <= b[2] and
                                    obj.get("top",    0) >= b[1] and
                                    obj.get("bottom", 0) <= b[3]
                                )
                            )
                        raw_text = filtered.extract_text() or ""
                    else:
                        raw_text = page.extract_text() or ""
                    page_texts[page_num] = _clean(raw_text)
                except Exception as e:
                    errors.append(f"{filename} p.{page_num} text: {e}")
                    page_texts[page_num] = ""

            # ── Build word list with page tracking ────────────────────────────
            full_words: list[str] = []
            word_pages: list[int] = []
            for pn in sorted(page_texts):
                words = page_texts[pn].split()
                full_words.extend(words)
                word_pages.extend([pn] * len(words))

            # ── Text chunks ───────────────────────────────────────────────────
            if full_words:
                n      = len(full_words)
                splits = [(0, n)] if n <= chunk_size else [(0, n // 2), (n // 2, n)]

                for part_num, (s, e) in enumerate(splits, start=1):
                    seg_words = full_words[s:e]
                    seg_pages = sorted(set(word_pages[s:e]))
                    p_label   = (
                        f"p.{seg_pages[0]}"
                        if len(seg_pages) == 1
                        else f"p.{seg_pages[0]}–{seg_pages[-1]}"
                    )
                    total_parts = len(splits)
                    title = (
                        f"{filename} — Full Text"
                        if total_parts == 1
                        else f"{filename} — Part {part_num} of 2 ({p_label})"
                    )
                    chunks.append({
                        "id":      _make_id(filename, "text", str(part_num)),
                        "type":    "text",
                        "title":   title,
                        "content": " ".join(seg_words),
                        "metadata": {
                            "source_pdf":   filename,
                            "source_pages": seg_pages,
                            "part":         part_num if total_parts > 1 else None,
                            "type":         "text",
                            "title":        title,
                        },
                    })

                log.info(
                    f"   [text]  {len(splits)} chunk(s) | {n} words | "
                    f"pages {word_pages[0]}–{word_pages[-1]}"
                )

            # ── Table chunks ──────────────────────────────────────────────────
            t_count = 0
            for pn in sorted(page_tables):
                for raw_table in page_tables[pn]:
                    if not raw_table or len(raw_table) < 2:
                        continue
                    header_row = [str(c or "").strip() for c in raw_table[0]]
                    data_rows  = [
                        [str(c or "").strip() for c in row]
                        for row in raw_table[1:]
                        if any(c for c in row)
                    ]
                    if not header_row or not data_rows:
                        continue
                    t_count += 1
                    title   = f"{filename} — Table {t_count} (p.{pn})"
                    content = _table_to_markdown(raw_table)
                    chunks.append({
                        "id":      _make_id(filename, "table", str(pn), str(t_count)),
                        "type":    "table",
                        "title":   title,
                        "content": content,
                        "metadata": {
                            "source_pdf":    filename,
                            "source_pages":  [pn],
                            "part":          None,
                            "type":          "table",
                            "title":         title,
                            "table_headers": json.dumps(header_row),
                            "table_rows":    json.dumps(data_rows),
                        },
                    })

            if t_count:
                log.info(f"   [table] {t_count} chunk(s)")

    except Exception as e:
        errors.append(f"Cannot open {filename}: {e}")

    return chunks, errors


# ─── UPSERT ───────────────────────────────────────────────────────────────────

def _upsert_chunks(index, chunks: list[dict], embed_fn, ns: str = "") -> None:
    """Embed and upsert in batches of 50."""
    BATCH = 50

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i: i + BATCH]
        texts = [c["content"][:8000] for c in batch]

        try:
            vectors = embed_fn(texts)
        except Exception as e:
            log.error(f"Embedding batch failed: {e}")
            raise

        records = []
        for chunk, vector in zip(batch, vectors):
            meta = {k: v for k, v in chunk["metadata"].items() if v is not None}
            meta["content"] = chunk["content"][:20000]
            # Pinecone requires list values to be list-of-strings
            if "source_pages" in meta:
                meta["source_pages"] = [str(p) for p in meta["source_pages"]]
            records.append({
                "id":       chunk["id"],
                "values":   vector,
                "metadata": meta,
            })

        kwargs = {"vectors": records}
        if ns:
            kwargs["namespace"] = ns
        index.upsert(**kwargs)
        log.info(f"   Upserted batch {i // BATCH + 1} ({len(records)} vectors)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def ingest_folder(folder: str, chunk_size: int) -> None:
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDFs found in '{folder}'.")
        return

    embed_fn, vector_dim = _get_embedder()
    index = _init_pinecone(vector_dim)

    all_errors:  list[str] = []
    total_text  = 0
    total_table = 0

    for pdf_path in pdf_files:
        chunks, errors = _parse_pdf(pdf_path, chunk_size)
        all_errors.extend(errors)

        if chunks:
            _upsert_chunks(index, chunks, embed_fn, ns=PINECONE_NS)

        total_text  += sum(1 for c in chunks if c["type"] == "text")
        total_table += sum(1 for c in chunks if c["type"] == "table")

    log.info("─" * 60)
    log.info(f"✓ Done — {len(pdf_files)} PDFs | {total_text} text | {total_table} table chunks")
    log.info(f"  Index     : {PINECONE_INDEX}")
    if PINECONE_NS:
        log.info(f"  Namespace : {PINECONE_NS}")
    if all_errors:
        log.warning(f"  Errors: {len(all_errors)}")
        for e in all_errors:
            log.warning(f"    {e}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone using SentenceTransformers.")
    parser.add_argument("--folder",     default=PDF_FOLDER, help="Folder containing PDFs")
    parser.add_argument("--chunk-size", default=CHUNK_SIZE, type=int,
                        help="Words per text chunk before splitting into 2")
    args = parser.parse_args()

    ingest_folder(folder=args.folder, chunk_size=args.chunk_size)