# Centinela — rag-service

Servicio de búsqueda semántica y generación aumentada (RAG) sobre el corpus de publicaciones científicas de Centinela. Recibe una pregunta en lenguaje natural, recupera los papers más relevantes vía FAISS + reranking, y genera una respuesta citada usando un LLM local (Ollama).

Parte del org multi-repo `PlataformaIntegradaInvestigadores`. Se expone a través de `gateway-service` en la ruta `/api/rag/` (upstream `rag-service:8181`); no se llama directo desde el frontend.

## Stack

- FastAPI 0.119 + Uvicorn
- FAISS (`faiss-cpu`) — índice vectorial de similitud (`resources/faiss_index_ip.bin`)
- Sentence-Transformers — embeddings + cross-encoder para reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- fastText — detección de idioma de la consulta (`resources/lid.176.ftz`)
- Ollama (HTTP, servicio aparte) — generación con `gemma3:4b`
- pandas / numpy — manejo del corpus (`resources/scopusdata.csv`, `resources/embeddings_meta_min.pkl`)

## Estructura del proyecto

```
app/
├── main.py                     # FastAPI app, CORS, registro de routers
├── core/
│   ├── config.py                # lectura de variables de entorno (rutas, params de RAG/Ollama)
│   └── utils.py                 # helpers cross-cutting (_safe_str)
├── schemas/
│   └── rag.py                   # modelos Pydantic de request/response
├── services/
│   ├── language.py               # detección de idioma (fastText)
│   ├── retrieval.py               # carga de índice FAISS + embeddings + búsqueda
│   ├── rerank.py                  # cross-encoder reranking
│   ├── context.py                  # construcción de contexto/prompt, autores
│   ├── ollama_client.py            # cliente HTTP hacia Ollama
│   └── refs.py                     # formateo de referencias citadas
└── api/v1/endpoints/
    ├── health.py                    # GET /health
    └── ask.py                        # POST /ask

resources/                       # artefactos de datos (índice FAISS, embeddings, corpus, modelo fastText)
tests/                           # pytest, mocks de los loaders pesados (FAISS/fastText/sentence-transformers)
```

## Requisitos previos

- Docker + Docker Compose
- Un contenedor Ollama corriendo el modelo `gemma3:4b` (ver más abajo)

## Levantar en local

### Con Docker (recomendado)

```bash
docker compose up -d
docker exec -it ollama bash
ollama pull gemma3:4b
```

El servicio queda disponible en `http://127.0.0.1:8181`.

### Sin Docker (desarrollo)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8181 --reload
```

Requiere que las variables de entorno (ver abajo) apunten a los artefactos en `resources/` y a un Ollama accesible.

## Variables de entorno

Ver `.env.example`. Variables clave:

| Variable | Descripción |
|---|---|
| `SCOPUS_CSV` | Ruta al corpus (`resources/scopusdata.csv` en Docker) |
| `PKL_MIN_PATH` | Ruta a los embeddings pre-calculados (`resources/embeddings_meta_min.pkl`) |
| `FAISS_PATH` | Ruta al índice FAISS (`resources/faiss_index_ip.bin`) |
| `LID_MODEL_PATH` | Ruta al modelo fastText de detección de idioma (`resources/lid.176.ftz`) |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | Endpoint y modelo del servicio Ollama |
| `RAG_TOP_CONTEXT`, `RAG_MAX_INPUT_CHARS`, `RAG_MAX_CHUNK_CHARS`, `RAG_TRIM_ABSTRACT` | Tuning del contexto pasado al LLM |
| `RAG_TEMPERATURE`, `RAG_MAX_NEW_TOKENS`, `RAG_HTTP_TIMEOUT_SECS` | Tuning de la generación |
| `API_TOPK` | Top-K de resultados devueltos por `/ask` |
| `PORT` | Puerto de escucha de Uvicorn |

## Documentación (Swagger)

Schema OpenAPI (autogenerado por FastAPI): `GET /openapi.json`. UI local propia en `/docs`. Centralizado también en el hub del `gateway-service`: `/api/docs/v1/rag`. Las rutas ya viven en la raíz, así que basta declarar `servers: [{"url": "/api/rag"}]` en el constructor de `FastAPI(...)` — no hace falta recortar prefijo. `/health` queda fuera del schema (`include_in_schema=False`).

## Tests

```bash
pytest tests/ -v --cov=app --cov-report=term
```

Cobertura mínima exigida en CI: **90%** (`--cov-fail-under=90` en `.github/workflows/ci.yml`). Los loaders pesados (FAISS, fastText, sentence-transformers, Ollama HTTP) se mockean vía caches a nivel de módulo en `services/`.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`): control de calidad (`ruff` + `black`) → tests unitarios (pytest, cobertura ≥90%) → build de imagen Docker → deploy automático a staging (`develop` branch, runner self-hosted `ticcd`) con healthcheck y rollback automático.

## Convenciones

- Branches: `feature/*` → `develop`, `hotfix/*` → `main`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/), inglés, con el *por qué* en el cuerpo.
