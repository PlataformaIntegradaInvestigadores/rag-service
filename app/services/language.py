import re

import fasttext

from app.core.config import LID_MODEL_PATH
from app.core.utils import _safe_str

_lid_model_cache = None


def _get_lid_model():
    global _lid_model_cache
    if _lid_model_cache is None:
        _lid_model_cache = fasttext.load_model(LID_MODEL_PATH)
        print(f"[INFO] fastText LID cargado desde: {LID_MODEL_PATH}")
    return _lid_model_cache


def detect_lang_any(text: str) -> tuple[str, float]:
    """
    Devuelve (iso639_1, confidence) p.ej., ('es', 0.98).
    Sin fallback: si el código no es válido, lanza excepción.
    """
    s = _safe_str(text).strip()
    if not s:
        raise ValueError("Texto vacío para detección de idioma.")
    model = _get_lid_model()
    labels, probs = model.predict(s.replace("\n", " ")[:4000], k=1)
    lab = _safe_str(labels[0]).replace("__label__", "").lower()
    conf = float(probs[0])

    # Normaliza a ISO-639-1 de 2 letras
    iso = lab.split("-")[0]
    if len(iso) > 2:
        iso = iso[:2]
    if not re.fullmatch(r"[a-z]{2}", iso):
        raise ValueError(f"Etiqueta de idioma inválida devuelta por fastText: {lab}")
    return iso, conf
