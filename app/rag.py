"""Vector store (pgvector), recuperação e geração com citações."""
import os
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

COLLECTION = "docs"

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


def ask(question: str, k: int = 5) -> dict:
    hits = store().similarity_search_with_score(question, k=k)  # score = distância cosseno
    if not hits:
        return {"answer": "Nenhum documento indexado ainda.", "sources": []}
    context = "\n\n".join(f"[{i}] ({d.metadata['source']}, {d.metadata['location']})\n{d.page_content}"
                          for i, (d, _) in enumerate(hits, start=1))
    answer = (PROMPT | llm()).invoke({"context": context, "question": question}).content
    sources = [{"id": i, "source": d.metadata["source"], "location": d.metadata["location"],
                "relevance": round(1 - s, 3), "snippet": d.page_content[:350]}
               for i, (d, s) in enumerate(hits, start=1)]
    return {"answer": answer, "sources": sources}
