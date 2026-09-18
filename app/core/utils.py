def _safe_str(x) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN
        return ""
    try:
        return str(x)
    except Exception:
        return ""
