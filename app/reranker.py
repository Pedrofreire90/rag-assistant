"""Reranking com cross-encoder (FlashRank): reordena os candidatos da busca vetorial."""
import os
from functools import lru_cache

MODEL = os.getenv("RERANK_MODEL", "ms-marco-MultiBERT-L-12")  # multilíngue
CACHE = os.getenv("RERANK_CACHE", "/models")


@lru_cache
def _ranker():
    from flashrank import Ranker
    return Ranker(model_name=MODEL, cache_dir=CACHE)


def rerank(query: str, hits: list, top_n: int) -> list:
    """hits: [(Document, distancia)] -> [(Document, distancia, score_rerank)] com os top_n melhores."""
    from flashrank import RerankRequest
    passages = [{"id": i, "text": d.page_content} for i, (d, _) in enumerate(hits)]
    ranked = _ranker().rerank(RerankRequest(query=query, passages=passages))
    return [(hits[r["id"]][0], hits[r["id"]][1], float(r["score"])) for r in ranked[:top_n]]
