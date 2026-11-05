# --- Etapa builder: toolchain de compilación (fasttext necesita swig/cmake/gcc) ---
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        swig \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r /app/requirements.txt

# --- Etapa final: solo runtime, sin compiladores ---
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# Run as appuser from here on, so the model cache below belongs to it.
USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

# Bake the Hugging Face models into the image. They used to be downloaded (559 MB)
# the first time they were needed after every container creation, which made
# startup depend on internet access and on the HF servers. The embedding model is
# the one named in resources/embeddings_meta_min.pkl; if that artifact is
# regenerated with another model, change it here too.
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small', device='cpu'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')"

# 2) Código y artefactos (RAG)
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser resources/ /app/resources/

# 3) Variables de entorno (ajusta si aplica)
ENV PKL_MIN_PATH=/app/resources/embeddings_meta_min.pkl \
    FAISS_PATH=/app/resources/faiss_index_ip.bin \
    SCOPUS_CSV=/app/resources/scopusdata.csv \
    SCOPUS_SEP="|" \
    LID_MODEL_PATH=/app/resources/lid.176.ftz \
    RAG_TEMPERATURE=0.2 \
    RAG_MAX_NEW_TOKENS=768 \
    RAG_TOP_CONTEXT=6 \
    RAG_MAX_INPUT_CHARS=7000 \
    RAG_MAX_CHUNK_CHARS=900 \
    RAG_TRIM_ABSTRACT=1 \
    API_TOPK=100 \
    RAG_HTTP_TIMEOUT_SECS=300 \
    CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    OLLAMA_HOST=http://ollama:11434 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OLLAMA_MODEL=gemma3:4b \
    PORT=8181

# Puerto FastAPI
EXPOSE 8181

# /health loads the embedding model, the FAISS index and the catalogue on its
# first call, hence the long start period.
HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8181/health', timeout=10)"

# Uvicorn como server (más estándar que __main__ en contenedores)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8181"]
