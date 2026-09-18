import re

from app.core.utils import _safe_str
from app.services.context import _authors_cite_line, _doi_url


def extract_used_refs(answer_text: str, n_max: int) -> list[int]:
    nums = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", _safe_str(answer_text))]
    seen, used = set(), []
    for n in nums:
        if 1 <= n <= n_max and n not in seen:
            used.append(n)
            seen.add(n)
    return used


def render_used_refs_report(answer_text: str, blocks: list[dict]) -> str:
    used = extract_used_refs(answer_text, len(blocks))
    if not used:
        return "No se detectaron citas [n] en el texto."
    lines = ["Citas usadas en el texto:"]
    for n in used:
        b = blocks[n - 1]
        lines.append(
            f" - [{n}] {_authors_cite_line(b.get('authors_raw',''))}; "
            f"\"{_safe_str(b.get('title','Sin título'))}\"; "
            f"{_doi_url(b.get('doi_raw',''))}"
        )
    return "\n".join(lines)
