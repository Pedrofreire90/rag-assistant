"""Busca híbrida: busca lexical (full-text do PostgreSQL) + fusão RRF com a busca vetorial."""
import os
import re

from langchain_core.documents import Document
from sqlalchemy import text

CFG = os.getenv("FTS_CONFIG", "portuguese")  # idioma do dicionário de stemming/stopwords
assert re.fullmatch(r"[a-z_]+", CFG), "FTS_CONFIG inválido"
RRF_K = int(os.getenv("RRF_K", "60"))
_ready = False


def key(doc: Document) -> tuple:
    """Identifica um trecho independentemente de qual busca o encontrou."""
    return (doc.metadata.get("source"), doc.metadata.get("location"), doc.page_content)


def _ensure_index(engine):
    global _ready
    if not _ready:
        with engine.begin() as c:
            c.execute(text(f"CREATE INDEX IF NOT EXISTS idx_chunks_fts_{CFG} "
                           f"ON langchain_pg_embedding USING GIN (to_tsvector('{CFG}', document))"))
        _ready = True


def _or_query(question: str) -> str:
    terms = dict.fromkeys(t for t in re.findall(r"\w+", question.lower()) if len(t) > 1)
    return " | ".join(terms)  # OR entre termos; stopwords são descartadas pelo dicionário


def lexical_search(engine, collection: str, question: str, limit: int) -> list[Document]:
    q = _or_query(question)
    if not q:
        return []
    _ensure_index(engine)
    sql = text(f"""
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :n AND to_tsvector('{CFG}', e.document) @@ to_tsquery('{CFG}', :q)
        ORDER BY ts_rank_cd(to_tsvector('{CFG}', e.document), to_tsquery('{CFG}', :q)) DESC
        LIMIT :k""")
    with engine.connect() as c:
        rows = c.execute(sql, {"n": collection, "q": q, "k": limit}).all()
    return [Document(page_content=r.document, metadata=r.cmetadata) for r in rows]


def fuse(vector_hits: list, lexical_docs: list, limit: int) -> list:
    """RRF: soma 1/(RRF_K + posição) em cada lista. -> [(Document, distancia|None, origem, score_rrf)]"""
    pool: dict = {}
    for rank, (d, dist) in enumerate(vector_hits, start=1):
        e = pool.setdefault(key(d), {"doc": d, "dist": dist, "o": set(), "rrf": 0.0})
        e["o"].add("vetorial"); e["rrf"] += 1 / (RRF_K + rank)
    for rank, d in enumerate(lexical_docs, start=1):
        e = pool.setdefault(key(d), {"doc": d, "dist": None, "o": set(), "rrf": 0.0})
        e["o"].add("lexical"); e["rrf"] += 1 / (RRF_K + rank)
    best = sorted(pool.values(), key=lambda e: e["rrf"], reverse=True)[:limit]
    return [(e["doc"], e["dist"], "vetorial+lexical" if len(e["o"]) == 2 else next(iter(e["o"])), e["rrf"])
            for e in best]
