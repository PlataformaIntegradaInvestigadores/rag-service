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

WORKDIR /app

COPY --from=builder /install /usr/local

# 2) Código y artefactos (RAG)
COPY app/ /app/app/
COPY resources/ /app/resources/

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
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=gemma3:4b \
    PORT=8181

# Puerto FastAPI
EXPOSE 8181

# Uvicorn como server (más estándar que __main__ en contenedores)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8181"]
