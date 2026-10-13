"""Vector store (pgvector), recuperação e geração com citações."""
import logging
import os
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
     "Use o histórico da conversa apenas para entender a pergunta; os fatos devem vir dos trechos. "
     "Cite as fontes no texto usando o número entre colchetes, ex.: [1] ou [2][3]. "
     "Se os trechos não contêm a resposta, diga que não encontrou essa informação nos documentos. "
     "Responda no idioma da pergunta.\n\nTRECHOS:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Reescreva a última pergunta do usuário como uma pergunta completa e independente, usando o histórico "
     "para resolver referências (ex.: 'e o segundo?', 'explique melhor', 'por quê?'). Mantenha o idioma. "
     "Se a pergunta já for independente, devolva-a sem alterações. Responda APENAS com a pergunta."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
MAX_HISTORY = 6  # mensagens (3 trocas) consideradas



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


def retrieve(question: str, k: int = 5, rerank: bool | None = None, hybrid_search: bool | None = None):
    """Recuperação sem LLM. -> (ranked [(Document, dist|None, score_rerank|None)], info, hybrid_ok, reranked)"""
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
        return [], {}, False, False

    info = {hybrid.key(d): (o, rrf) for d, _, o, rrf in cands}
    pairs = [(d, s) for d, s, _, _ in cands]
    ranked = [(d, s, None) for d, s in pairs[:k]]
    if use_rerank:
        try:
            ranked = reranker.rerank(question, pairs, k)
        except Exception:  # sem o modelo, mantém a ordem anterior
            log.exception("Reranking falhou; usando ordem da busca anterior")
    return ranked, info, hybrid_ok, any(rr is not None for _, _, rr in ranked)


def _history_messages(history: list | None) -> list:
    return [HumanMessage(content=(t.get("content") or "")[:1500]) if t.get("role") == "user"
            else AIMessage(content=(t.get("content") or "")[:1500])
            for t in (history or [])[-MAX_HISTORY:]]


def condense_question(question: str, history: list | None) -> str:
    """Transforma uma pergunta de acompanhamento em uma pergunta independente (usada na busca)."""
    msgs = _history_messages(history)
    if not msgs:
        return question
    try:
        out = (CONDENSE_PROMPT | llm()).invoke({"history": msgs, "question": question}).content.strip()
        return out if 0 < len(out) <= 500 else question
    except Exception:
        log.exception("Reescrita da pergunta falhou; usando a pergunta original")
        return question


def ask(question: str, k: int = 5, rerank: bool | None = None, hybrid_search: bool | None = None,
        history: list | None = None) -> dict:
    standalone = condense_question(question, history)
    ranked, info, hybrid_ok, reranked = retrieve(standalone, k, rerank, hybrid_search)
    if not ranked:
        return {"answer": "Nenhum documento indexado ainda.", "sources": [], "mode": ""}

    context = "\n\n".join(f"[{i}] ({d.metadata['source']}, {d.metadata['location']})\n{d.page_content}"
                          for i, (d, _, _) in enumerate(ranked, start=1))
    answer = (PROMPT | llm()).invoke({"context": context, "history": _history_messages(history),
                                      "question": question}).content

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
    return {"answer": answer, "sources": sources, "mode": mode, "standalone_question": standalone}


def debug_retrieval(question: str, contains: str | None = None, k: int = 5) -> dict:
    """Mostra o ranking de cada etapa (vetorial, lexical, fusão, reranker) sem chamar o LLM.
    Se `contains` for dado, informa em que posição aparecem os trechos que contêm esse texto."""
    n = max(k, CANDIDATES)
    vec = store().similarity_search_with_score(question, k=n)
    lex = hybrid.lexical_search(engine(), COLLECTION, question, n)
    fused = hybrid.fuse(vec, lex, n)
    rer = reranker.rerank(question, [(d, s) for d, s, _, _ in fused], n) if fused else []

    def view(docs):
        return [{"posicao": i, "fonte": d.metadata.get("source"), "local": d.metadata.get("location"),
                 "contem_termo": (contains.lower() in d.page_content.lower()) if contains else None,
                 "inicio": d.page_content[:110].replace("\n", " ")} for i, d in enumerate(docs, start=1)]

    etapas = {"1_vetorial": view([d for d, _ in vec]), "2_lexical": view(lex),
              "3_fusao_rrf": view([d for d, *_ in fused]), "4_reranker": view([d for d, _, _ in rer])}
    achados = ({nome: [x["posicao"] for x in v if x["contem_termo"]] for nome, v in etapas.items()}
               if contains else None)
    return {"consulta_lexical": hybrid._or_query(question), "posicao_dos_trechos_com_o_termo": achados,
            "etapas": etapas}
