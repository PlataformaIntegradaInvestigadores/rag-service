import re

import pandas as pd

from app.core.config import (
    DO_TRIM_ABSTRACT,
    MAX_CHUNK_CHARS,
    MAX_INPUT_CHARS,
    TOP_CONTEXT,
)
from app.core.utils import _safe_str
from app.services.rerank import _first_nonempty


def _shorten(txt, lim: int) -> str:
    s = _safe_str(txt)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[: lim - 3] + "...") if len(s) > lim else s


def _split_authors(raw: str) -> list[str]:
    s = _safe_str(raw)
    if not s.strip():
        return []
    s = s.replace("|", ";").replace(" and ", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def _last_name(name: str) -> str:
    n = _safe_str(name).strip()
    if not n:
        return ""
    if "," in n:
        return n.split(",")[0].strip()
    tokens = n.split()
    return tokens[-1].strip() if tokens else n


def _format_authors_for_mention(raw: str, max_names: int = 2) -> str | None:
    authors = _split_authors(raw)
    if not authors:
        return None
    last_names = [_last_name(a) for a in authors if _last_name(a)]
    if not last_names:
        return None
    if len(last_names) == 1:
        return last_names[0]
    if len(last_names) == 2:
        return f"{last_names[0]} y {last_names[1]}"
    return f"{last_names[0]} et al."


def _extract_year(row: pd.Series) -> str | None:
    for c in ["year", "publication_year", "cover_date", "date"]:
        val = _safe_str(row.get(c))
        m = re.search(r"(19|20)\d{2}", val)
        if m:
            return m.group(0)
    return None


def build_context_blocks(
    df_reranked: pd.DataFrame,
    top_k: int = TOP_CONTEXT,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    if df_reranked is None or len(df_reranked) == 0:
        raise ValueError("df_reranked está vacío.")
    cols_title = [c for c in ["title", "chunk_title"] if c in df_reranked.columns] or [
        "title"
    ]
    cols_abs = [
        c for c in ["abstract", "chunk_text", "summary"] if c in df_reranked.columns
    ] or ["abstract"]
    blocks = []
    for i in range(min(top_k, len(df_reranked))):
        row = df_reranked.iloc[i]
        title = _first_nonempty(row, cols_title) or "Sin título"
        body = _first_nonempty(row, cols_abs)
        if DO_TRIM_ABSTRACT:
            body = _shorten(body, max_chunk_chars)
        authors_raw = _safe_str(row.get("authors", ""))
        year = _extract_year(row)
        doi_raw = _safe_str(row.get("doi", ""))
        blocks.append(
            {
                "cite_id": str(row.get("scopus_id") or row.get("vec_id") or f"row{i}"),
                "title": title,
                "text": body,
                "authors_mention": _format_authors_for_mention(authors_raw),
                "authors_raw": authors_raw,
                "year": year,
                "doi_raw": doi_raw,
            }
        )
    return blocks


def _doi_url(doi_raw: str) -> str:
    doi = _safe_str(doi_raw).strip()
    if not doi:
        return "s/d"
    return doi if doi.lower().startswith("http") else f"https://doi.org/{doi}"


def _authors_cite_line(raw_authors: str) -> str:
    names = _split_authors(_safe_str(raw_authors))
    if not names:
        return "Autor(es) no disponibles"
    return "; ".join([_safe_str(n).strip() for n in names if _safe_str(n).strip()])


def render_fuentes_from_blocks(blocks: list[dict]) -> str:
    lines = []
    for i, b in enumerate(blocks, start=1):
        autores = _authors_cite_line(b.get("authors_raw", ""))
        titulo = _safe_str(b.get("title", "Sin título"))
        doiurl = _doi_url(b.get("doi_raw", ""))
        lines.append(f'[{i}] {autores}; "{titulo}"; {doiurl}')
    return "\n".join(lines)


def compose_prompt(
    query: str,
    blocks: list[dict],
    max_chars: int = MAX_INPUT_CHARS,
    target_iso: str = "es",
) -> str:
    """
    Universal compose_prompt (English instructions) that forces the model
    to answer entirely in the language specified by target_iso (ISO 639-1).
    It keeps your [n] citation discipline and RAG constraints.
    """
    if not re.fullmatch(r"[a-z]{2}", _safe_str(target_iso).lower()):
        target_iso = "es"
    target_iso = target_iso.lower()

    header = (
        "You are an academic assistant for Retrieval-Augmented Generation (RAG).\n"
        "Write the ENTIRE answer in the language whose ISO 639-1 code is "
        f"'{target_iso}'. "
        "Do not mix languages. Ignore the language of the user message if it "
        "differs from this code.\n\n"
        "Style & constraints:\n"
        "- Be clear, concise, and evidence-based.\n"
        "- Follow APA 7th in-text style.\n"
        "- Use ONLY the provided excerpts. Do NOT invent or add external facts.\n"
        "- Every paragraph must include an explicit author mention written in the "
        "target language "
        "(e.g., 'según <autor>' or 'according to <author>') followed by the "
        "bracketed citation [n].\n"
        "- If no author is available, use 'according to source [n]' (translated "
        "into the target language if applicable).\n"
        "- Do NOT print any 'References' section at the end.\n\n"
        f"Question:\n{query}\n\n"
        "Source excerpts (with author/year if available):\n"
    )

    parts = []
    for i, b in enumerate(blocks, start=1):
        author_mention = b.get("authors_mention") or f"source [{i}]"
        year = f" ({b['year']})" if b.get("year") else ""
        head = (
            f"[{i}] { _safe_str(b.get('title','Untitled')) } — Authors: "
            f"{author_mention}{year}"
        )
        parts.append(f"{head}\n{_safe_str(b.get('text',''))}\n")

    footer = (
        "\nWriting checklist (must obey):\n"
        f"- Use the numeric citations [1..{len(blocks)}] exactly as provided.\n"
        "- Keep paragraphs tightly focused; avoid generic boilerplate.\n"
        "- Maintain consistent tone; no bullet points unless strictly necessary.\n"
        "- Do not include any section titled 'References' or similar.\n"
    )

    fuentes_prompt = render_fuentes_from_blocks(blocks)
    prompt = (
        header
        + "\n".join(parts)
        + footer
        + "\n(Do NOT print the list below)\nSources guide:\n"
        + fuentes_prompt
    )
    return prompt[:max_chars]
