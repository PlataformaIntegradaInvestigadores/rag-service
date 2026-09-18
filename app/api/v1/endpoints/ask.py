import time

from fastapi import APIRouter, HTTPException

from app.core.config import DEFAULT_TOPK, MAX_CHUNK_CHARS, MAX_INPUT_CHARS, TOP_CONTEXT
from app.schemas.rag import AskRequest, AskResponse, Block, Timing
from app.services.context import build_context_blocks, compose_prompt
from app.services.language import detect_lang_any
from app.services.ollama_client import generate_with_ollama_http
from app.services.refs import render_used_refs_report
from app.services.rerank import rerank_with_cross_encoder
from app.services.retrieval import search_full_scopus

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query vacía.")

    # --- Detección de idioma (sin fallback) ---
    try:
        detected_iso, conf = detect_lang_any(q)
        print(f"[INFO] Idioma detectado: {detected_iso} (conf={conf:.3f})")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Fallo detectando idioma: {e}"
        ) from e

    topk = payload.topk or DEFAULT_TOPK
    t0 = time.time()

    # 1) Recuperación (denso FAISS + unión CSV)
    try:
        df_topk = search_full_scopus(q, topk=topk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en búsqueda: {e}") from e
    t1 = time.time()

    # 2) Re-ranking (CE + denso)
    try:
        reranked = rerank_with_cross_encoder(
            query_text=q,
            df_topk=df_topk,
            text_cols=None,
            score_dense_col="score",
            fuse_with_dense=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en re-ranking: {e}") from e
    t2 = time.time()

    # 3) Bloques de contexto
    try:
        blocks_raw = build_context_blocks(
            reranked, top_k=TOP_CONTEXT, max_chunk_chars=MAX_CHUNK_CHARS
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Fallo construyendo bloques: {e}"
        ) from e

    # Convertir dicts → modelos Block
    blocks = [Block(**b) for b in blocks_raw]

    # 4) Prompt universal y 5) Generación forzada al idioma detectado
    prompt = compose_prompt(
        q, blocks_raw, max_chars=MAX_INPUT_CHARS, target_iso=detected_iso
    )
    try:
        answer_text = generate_with_ollama_http(prompt, target_iso=detected_iso)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Fallo generando respuesta LLM: {e}"
        ) from e
    t3 = time.time()

    # 6) Auditoría
    try:
        used_refs_report = render_used_refs_report(answer_text, blocks_raw)
    except Exception as e:
        used_refs_report = f"(No se pudo auditar las citas) Detalle: {e}"

    timing_ms_dict = {
        "search": int((t1 - t0) * 1000),
        "rerank": int((t2 - t1) * 1000),
        "generate": int((t3 - t2) * 1000),
        "total": int((t3 - t0) * 1000),
    }
    timing = Timing(**timing_ms_dict)

    return AskResponse(
        query=q,
        answer_text=answer_text,
        blocks=blocks,
        used_refs_report=used_refs_report,
        timing_ms=timing,
    )
