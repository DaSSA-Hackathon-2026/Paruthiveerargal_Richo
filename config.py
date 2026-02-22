import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX", "rag-anything")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE   = os.getenv("PINECONE_NAMESPACE", "")

HF_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

PDF_FOLDER = os.getenv("D:\HackVerse\RPD-en-US\RPD-en-US", "./pdfs")


HF_TOKEN = os.getenv("HF_TOKEN", "hf_hxxxxxxxxxxxxxxxxxxsKl")



HF_MODEL = os.getenv("HF_MODEL", "moonshotai/Kimi-K2-Instruct-0905:groq")
