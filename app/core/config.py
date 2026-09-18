import os

# --- Retrieval / embeddings ---
PKL_MIN_PATH = os.environ.get("PKL_MIN_PATH", "resources/embeddings_meta_min.pkl")
FAISS_PATH = os.environ.get("FAISS_PATH", "resources/faiss_index_ip.bin")
SCOPUS_CSV = os.environ.get("SCOPUS_CSV", "resources/scopusdata.csv")
SCOPUS_SEP = os.environ.get("SCOPUS_SEP", "|")

# --- Language detection (fastText) ---
LID_MODEL_PATH = os.environ.get("LID_MODEL_PATH", "resources/lid.176.ftz")

# --- Re-ranking (cross-encoder) ---
CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
CE_BATCH_SIZE = int(os.environ.get("CE_BATCH_SIZE", "64"))
W_CE = float(os.environ.get("W_CE", "0.7"))
W_DENSE = float(os.environ.get("W_DENSE", "0.3"))
TEXT_COLS = ["title", "abstract", "chunk_text", "summary", "authkeywords", "keywords"]

# --- RAG context / prompt ---
TOP_CONTEXT = int(os.environ.get("RAG_TOP_CONTEXT", "6"))
MAX_INPUT_CHARS = int(os.environ.get("RAG_MAX_INPUT_CHARS", "7000"))
MAX_CHUNK_CHARS = int(os.environ.get("RAG_MAX_CHUNK_CHARS", "900"))
DO_TRIM_ABSTRACT = os.environ.get("RAG_TRIM_ABSTRACT", "1") == "1"

# --- LLM (Ollama) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.2"))
MAX_NEW_TOKENS = int(os.environ.get("RAG_MAX_NEW_TOKENS", "768"))
HTTP_TIMEOUT_SECS = int(os.environ.get("RAG_HTTP_TIMEOUT_SECS", "300"))

# --- API ---
DEFAULT_TOPK = int(os.environ.get("API_TOPK", "100"))
