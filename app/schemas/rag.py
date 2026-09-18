from pydantic import BaseModel


class Block(BaseModel):
    cite_id: str
    title: str
    text: str
    authors_mention: str | None = None
    authors_raw: str | None = None
    year: str | None = None
    doi_raw: str | None = None


class Timing(BaseModel):
    search: int
    rerank: int
    generate: int
    total: int


class AskRequest(BaseModel):
    query: str
    topk: int | None = None


class AskResponse(BaseModel):
    query: str
    answer_text: str
    blocks: list[Block]
    used_refs_report: str
    timing_ms: Timing
