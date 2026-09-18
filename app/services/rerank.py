from collections.abc import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from app.core.config import CE_BATCH_SIZE, CROSS_ENCODER_MODEL, TEXT_COLS, W_CE, W_DENSE


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-12)


def _first_nonempty(row: pd.Series, cols: Sequence[str]) -> str:
    for c in cols:
        if c in row and isinstance(row[c], str) and row[c].strip():
            return row[c]
    parts = []
    for c in row.index:
        name = c.lower()
        if any(
            tok in name
            for tok in (
                "title",
                "abstract",
                "summary",
                "keywords",
                "chunk",
                "desc",
                "text",
            )
        ):
            v = row[c]
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
    return " ".join(parts)[:4096]


_ce_cache: CrossEncoder | None = None


def get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL) -> CrossEncoder:
    global _ce_cache
    if _ce_cache is None:
        _ce_cache = CrossEncoder(model_name, device="cpu")
        print(f"[INFO] Cross-Encoder cargado: {model_name}")
    return _ce_cache


def _build_pairs(
    query_text: str, df_topk: pd.DataFrame, text_cols: list[str] | None
) -> tuple[list[tuple[str, str]], list[int]]:
    cols = text_cols or TEXT_COLS
    pairs, idx_map = [], []
    for i, row in df_topk.iterrows():
        txt = _first_nonempty(row, cols)
        pairs.append((query_text, txt if isinstance(txt, str) else ""))
        idx_map.append(i)
    return pairs, idx_map


def rerank_with_cross_encoder(
    query_text: str,
    df_topk: pd.DataFrame,
    text_cols: list[str] | None = None,
    score_dense_col: str = "score",
    fuse_with_dense: bool = True,
    batch_size: int = CE_BATCH_SIZE,
    model_name: str = CROSS_ENCODER_MODEL,
) -> pd.DataFrame:
    if df_topk is None or len(df_topk) == 0:
        raise ValueError("df_topk vacío.")
    ce = get_cross_encoder(model_name)
    pairs, idx_map = _build_pairs(query_text, df_topk, text_cols)

    scores_ce = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        s = ce.predict(batch)
        scores_ce.append(np.asarray(s, dtype=np.float32))
    scores_ce = (
        np.concatenate(scores_ce, axis=0)
        if scores_ce
        else np.zeros(len(df_topk), dtype=np.float32)
    )

    out = df_topk.copy()
    out.loc[idx_map, "score_ce"] = scores_ce
    ce_norm = _minmax(out["score_ce"].values)

    if fuse_with_dense and score_dense_col in out.columns:
        dense_norm = _minmax(out[score_dense_col].values)
        out["score_dense_norm"] = dense_norm
        out["score_final"] = W_CE * ce_norm + W_DENSE * dense_norm
    else:
        out["score_final"] = ce_norm

    out = out.sort_values("score_final", ascending=False).reset_index(drop=True)
    return out
