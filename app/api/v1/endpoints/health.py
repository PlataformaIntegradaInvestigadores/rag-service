import logging
import socket

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.language import _get_lid_model
from app.services.retrieval import load_faiss, load_pkl_and_model, load_scopus_csv

router = APIRouter()
logger = logging.getLogger(__name__)


def _local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "unknown"


def _check(name, fn):
    try:
        fn()
        return {"name": name, "status": "ok"}
    except Exception:
        logger.exception("Health check failed: %s", name)
        return {"name": name, "status": "error"}


@router.get("/health", include_in_schema=False)
def health():
    services = [
        _check("modelo-embeddings", load_pkl_and_model),
        _check("indice-faiss", load_faiss),
        _check("catalogo-scopus", load_scopus_csv),
        _check("modelo-idioma", _get_lid_model),
    ]
    ok = all(s["status"] == "ok" for s in services)

    payload = {
        "server_name": "rag-service",
        "ip_address": _local_ip(),
        "global_status": "Online" if ok else "Offline",
        "groups": [
            {
                "group_name": "RAG / Búsqueda Semántica con IA",
                "group_status": "Operativo" if ok else "Caído",
                "services": services,
            }
        ],
    }
    if ok:
        return payload
    return JSONResponse(status_code=503, content=payload)
