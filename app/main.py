import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import ask, health

app = FastAPI(
    title="Centinela RAG API",
    version="1.0.0",
    servers=[{"url": "/api/rag", "description": "Gateway"}],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ask.router)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8181")))
