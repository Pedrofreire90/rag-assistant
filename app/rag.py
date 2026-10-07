"""Vector store (pgvector), recuperação e geração com citações."""
import logging
import os
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from . import reranker

log = logging.getLogger(__name__)
COLLECTION = "docs"
RERANK_DEFAULT = os.getenv("RERANK", "true").lower() == "true"
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


def ask(question: str, k: int = 5, rerank: bool | None = None) -> dict:
    use_rerank = RERANK_DEFAULT if rerank is None else rerank
    # Com reranking, busca-se mais candidatos e o cross-encoder escolhe os k melhores.
    hits = store().similarity_search_with_score(question, k=max(k, CANDIDATES) if use_rerank else k)
    if not hits:
        return {"answer": "Nenhum documento indexado ainda.", "sources": []}

    ranked = [(d, s, None) for d, s in hits[:k]]
    if use_rerank:
        try:
            ranked = reranker.rerank(question, hits, k)
        except Exception:  # sem o modelo, degrada para a busca vetorial pura
            log.exception("Reranking falhou; usando ordem da busca vetorial")

    context = "\n\n".join(f"[{i}] ({d.metadata['source']}, {d.metadata['location']})\n{d.page_content}"
                          for i, (d, _, _) in enumerate(ranked, start=1))
    answer = (PROMPT | llm()).invoke({"context": context, "question": question}).content
    sources = [{"id": i, "source": d.metadata["source"], "location": d.metadata["location"],
                "relevance": round(rr if rr is not None else 1 - s, 3),
                "vector_relevance": round(1 - s, 3), "reranked": rr is not None,
                "snippet": d.page_content[:350]}
               for i, (d, s, rr) in enumerate(ranked, start=1)]
    return {"answer": answer, "sources": sources}
