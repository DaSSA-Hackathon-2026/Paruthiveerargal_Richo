# 🔍 Multi-Agent RAG System

A document question-answering system that combines a **5-agent pipeline**, **Pinecone vector search**, **HuggingFace embeddings**, **CrossEncoder reranking**, and **Gemini 2.5 Flash** for generation — with a dark-themed web UI featuring an inline PDF viewer.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  MultiAgentOrchestrator              │
│                                                     │
│  1. SafetyGuardAgent   → Block harmful queries      │
│  2. QueryAnalysisAgent → Decompose & refine query   │
│  3. RetrievalAgent     → Multi-query vector search  │
│                          + CrossEncoder reranking   │
│  4. GenerationAgent    → Gemini 2.5 Flash answer    │
│  5. ValidationAgent    → Hallucination detection    │
└─────────────────────────────────────────────────────┘
    │
    ▼
JSON Response → Frontend UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI (Python) |
| **Vector Store** | Pinecone (Serverless) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| **Reranker** | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| **LLM** | Google Gemini 2.5 Flash |
| **PDF Parsing** | `pdfplumber` (text + tables), `PyPDF2` |
| **Frontend** | Vanilla HTML/CSS/JS + PDF.js |

---

## Prerequisites

- Python 3.10+
- Node.js (optional, for docx tooling)
- A [Pinecone](https://app.pinecone.io) account (free tier works)
- A [Google AI Studio](https://aistudio.google.com) API key for Gemini

---

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pdfplumber PyPDF2 pinecone-client \
            sentence-transformers google-genai python-dotenv
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-anything
PINECONE_NAMESPACE=
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key

# Paths & tuning
PDF_FOLDER=./pdfs
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5

# HuggingFace model (local, no API key needed)
HF_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

---

## Usage

### 1. Add PDFs

Place your PDF files in the `./pdfs` folder (or the path set in `PDF_FOLDER`).

### 2. Ingest PDFs

```bash
# Ingest all PDFs in ./pdfs
python ingest.py

# Custom folder or chunk size
python ingest.py --folder ./my_docs --chunk-size 600
```

This extracts text and tables, embeds them locally, and upserts vectors to Pinecone.

### 3. Start the API server

```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the UI

Open `index.html` in your browser (no build step needed). Make sure the API is running at `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | RAG system status |
| `GET` | `/stats` | Index stats (vector count, model info) |
| `POST` | `/ingest` | Ingest all PDFs from `PDF_FOLDER` |
| `POST` | `/upload` | Upload and immediately ingest a single PDF |
| `POST` | `/query` | Run full multi-agent pipeline |
| `POST` | `/analyze` | Run query analysis agent only |
| `POST` | `/retrieve` | Run retrieval agent only |
| `GET` | `/pdfs` | List available PDFs |
| `GET` | `/pdf/{filename}` | Serve a PDF file |
| `GET` | `/pdf/{filename}/highlight` | Get page + highlight rects for a text snippet |

### Query request/response

```json
// POST /query
{ "query": "What are the main findings?" }

// Response
{
  "query": "What are the main findings?",
  "blocked": false,
  "needs_clarification": false,
  "analysis": { "search_strategy": "broad", "sub_queries": [...] },
  "retrieval": { "sources": [...], "total_chunks": 5, "top_chunks": [...] },
  "generation": { "answer": "...", "confidence": "high" },
  "validation": { "final_verdict": "approved", "hallucination_detected": false, ... },
  "final_answer": "...",
  "elapsed_ms": 1840
}
```

---

## How the Agents Work

### 1. SafetyGuardAgent
Blocks prompt injections, jailbreak attempts, harmful content, and tool-call requests before any processing occurs.

### 2. QueryAnalysisAgent
Uses Gemini to refine vague queries, decompose complex ones into sub-queries, and select a search strategy (`broad`, `specific`, or `multi-hop`). Requests clarification only when truly necessary.

### 3. RetrievalAgent
Runs each sub-query independently against Pinecone, merges the candidate pool, deduplicates, then applies CrossEncoder reranking across the whole pool for the highest-quality final chunks.

### 4. GenerationAgent
Feeds token-truncated, reranked context to Gemini 2.5 Flash with strict grounding instructions. Skips generation entirely if retrieval returns nothing.

### 5. ValidationAgent
Scores the generated answer for hallucination, completeness (0–1), and relevance (0–1). Returns a `final_verdict` of `approved`, `needs_revision`, or `rejected`, along with a corrected answer if needed.

---

## PDF Viewer

Clicking a source pill in the UI opens an inline PDF viewer (powered by PDF.js) that:
- Jumps directly to the page containing the retrieved chunk
- Highlights matched words with a green overlay
- Supports page-by-page navigation

---

## Performance Features

- **Response caching** — identical Gemini prompts are served from an in-memory cache (MD5-keyed, max 256 entries)
- **Exponential backoff** — auto-retries on Gemini 429/503 errors (up to 4 attempts)
- **Token-aware truncation** — context is trimmed to 12,000 chars before generation to avoid exceeding limits
- **Multi-query retrieval** — sub-queries are searched in parallel and merged before reranking

---

## Project Structure

```
.
├── main.py          # FastAPI app, routes, PDF serving
├── agents.py        # All 5 agents + orchestrator
├── rag_system.py    # Embeddings, chunking, Pinecone, reranking
├── ingest.py        # Standalone CLI ingestion script
├── config.py        # Centralised config (reads from .env)
├── index.html       # Frontend UI
├── pdfs/            # Drop your PDFs here
└── .env             # API keys and settings (not committed)
```

---

## Troubleshooting

**`PDF 'xyz.pdf' not found`** — Check that `PDF_FOLDER` in your `.env` matches where your PDFs are stored.

**`PINECONE_API_KEY is not set`** — Make sure your `.env` is in the project root and `python-dotenv` is installed.

**Dimension mismatch on Pinecone index** — If you change `HF_MODEL_NAME`, the existing index will be deleted and recreated automatically.

**UI shows "Connection error"** — Confirm the FastAPI server is running at `http://localhost:8000` and CORS is not blocked by your browser.

---

## License

MIT