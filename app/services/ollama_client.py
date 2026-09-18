import re

import requests

from app.core.config import (
    HTTP_TIMEOUT_SECS,
    MAX_NEW_TOKENS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TEMPERATURE,
)
from app.core.utils import _safe_str


def generate_with_ollama_http(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    base_url: str = OLLAMA_HOST,
    timeout: int = HTTP_TIMEOUT_SECS,
    target_iso: str | None = None,
) -> str:
    """
    Sin fallback: target_iso (p.ej. 'es','en','fr') es obligatorio.
    """
    if not target_iso or not re.fullmatch(r"[a-z]{2}", target_iso):
        raise RuntimeError(
            "target_iso requerido y debe ser un código ISO-639-1 de dos letras "
            "(ej. 'es','en')."
        )

    system_msg = (
        "You are an academic assistant. Write the ENTIRE answer in the language "
        f"whose ISO 639-1 code is '{target_iso}'. "
        "Do not mix languages. Be concise, evidence-based, and follow APA 7th "
        "in-text citations using [n]. "
        "Use ONLY the provided excerpts."
    )

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_new_tokens),
            "stop": ["\nFuentes", "\nFUENTES", "\nReferences", "\nREFERENCIAS"],
        },
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = _safe_str(data.get("message", {}).get("content", ""))
    if not content.strip():
        raise RuntimeError("La respuesta de Ollama está vacía.")
    return content
