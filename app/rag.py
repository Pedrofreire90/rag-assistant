"""Vector store (pgvector), recuperação e geração com citações."""
import logging
import os
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from . import hybrid, reranker

log = logging.getLogger(__name__)
COLLECTION = "docs"
RERANK_DEFAULT = os.getenv("RERANK", "true").lower() == "true"
HYBRID_DEFAULT = os.getenv("HYBRID", "true").lower() == "true"
CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))  # quantos trechos a busca vetorial entrega ao reranker

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente corporativo. Responda SOMENTE com base nos trechos numerados abaixo. "
     "Cite as fontes no texto usando o número entre colchetes, ex.: [1] ou [2][3]. "
     "Se os trechos não contêm a resposta, diga que não encontrou essa informação nos documentos. "
     "Responda no idioma da pergunta.\n\nTRECHOS:\n{context}"),
    ("human", "{question}"),
])


@lru_cache
def engine():
    return create_engine(os.environ["DATABASE_URL"])


@lru_cache
def store() -> PGVector:
    emb = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return PGVector(embeddings=emb, collection_name=COLLECTION,
                    connection=os.environ["DATABASE_URL"], use_jsonb=True)


@lru_cache
def llm():
    model = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    if os.getenv("LLM_PROVIDER", "anthropic") == "openai":
        return ChatOpenAI(model=model, temperature=0)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, temperature=0, max_tokens=1024)


def delete_source(source: str):
    with engine().begin() as c:
        c.execute(text("""DELETE FROM langchain_pg_embedding e USING langchain_pg_collection c
                          WHERE c.uuid = e.collection_id AND c.name = :n AND e.cmetadata->>'source' = :s"""),
                  {"n": COLLECTION, "s": source})


def add_chunks(source: str, chunks) -> int:
    delete_source(source)  # reenvio do mesmo arquivo substitui o anterior
    store().add_documents(chunks)
    return len(chunks)


def list_sources():
    store()  # garante criação das tabelas
    with engine().connect() as c:
        rows = c.execute(text("""SELECT e.cmetadata->>'source' AS source, count(*) AS chunks
            FROM langchain_pg_embedding e JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = :n GROUP BY 1 ORDER BY 1"""), {"n": COLLECTION})
        return [{"source": r.source, "chunks": r.chunks} for r in rows]


def ask(question: str, k: int = 5, rerank: bool | None = None, hybrid_search: bool | None = None) -> dict:
    use_rerank = RERANK_DEFAULT if rerank is None else rerank
    use_hybrid = HYBRID_DEFAULT if hybrid_search is None else hybrid_search
    n = max(k, CANDIDATES) if (use_rerank or use_hybrid) else k

    vec = store().similarity_search_with_score(question, k=n)  # score = distância cosseno
    cands = [(d, s, "vetorial", None) for d, s in vec]
    hybrid_ok = False
    if use_hybrid:
        try:
            cands = hybrid.fuse(vec, hybrid.lexical_search(engine(), COLLECTION, question, n), n)
            hybrid_ok = True
        except Exception:  # sem a parte lexical, degrada para a busca vetorial pura
            log.exception("Busca lexical falhou; usando só a busca vetorial")
    if not cands:
        return {"answer": "Nenhum documento indexado ainda.", "sources": [], "mode": ""}

    info = {hybrid.key(d): (o, rrf) for d, _, o, rrf in cands}
    pairs = [(d, s) for d, s, _, _ in cands]
    ranked = [(d, s, None) for d, s in pairs[:k]]
    if use_rerank:
        try:
            ranked = reranker.rerank(question, pairs, k)
        except Exception:  # sem o modelo, mantém a ordem anterior
            log.exception("Reranking falhou; usando ordem da busca anterior")
    reranked = any(rr is not None for _, _, rr in ranked)

    context = "\n\n".join(f"[{i}] ({d.metadata['source']}, {d.metadata['location']})\n{d.page_content}"
                          for i, (d, _, _) in enumerate(ranked, start=1))
    answer = (PROMPT | llm()).invoke({"context": context, "question": question}).content

    sources = []
    for i, (d, s, rr) in enumerate(ranked, start=1):
        origin, rrf = info[hybrid.key(d)]
        if rr is not None:
            rel, kind = rr, "reranker"
        elif rrf is not None:
            rel, kind = rrf, "rrf"
        else:
            rel, kind = 1 - s, "cosseno"
        sources.append({"id": i, "source": d.metadata["source"], "location": d.metadata["location"],
                        "origin": origin, "score_type": kind, "relevance": round(rel, 4),
                        "vector_relevance": None if s is None else round(1 - s, 3),
                        "reranked": rr is not None, "snippet": d.page_content[:350]})
    mode = ("híbrida (vetorial + lexical)" if hybrid_ok else "vetorial") + (" + reranking" if reranked else "")
    return {"answer": answer, "sources": sources, "mode": mode}
