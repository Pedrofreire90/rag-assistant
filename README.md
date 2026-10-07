# Assistente Inteligente com RAG

Pergunte em linguagem natural sobre PDFs, DOCX e planilhas; as respostas citam trechos e páginas/abas de origem.

## Rodar
```bash
cp .env.example .env      # preencha as chaves
docker compose up --build
# abra http://localhost:8000  (API docs: /docs)
```

## Fluxo
1. **Upload** (`POST /documents`): extrai texto (pypdf, python-docx, pandas) por página/aba.
2. **Chunking**: `RecursiveCharacterTextSplitter` (1000 chars, overlap 150), com metadados `source` e `location`.
3. **Embeddings**: OpenAI `text-embedding-3-small`, gravados no PostgreSQL via `pgvector` (LangChain `PGVector`).
4. **Busca semântica + reranking** (`POST /ask`): a busca vetorial recupera 20 candidatos por cosseno e um cross-encoder multilíngue (FlashRank) reordena e mantém os 5 melhores. Use `"rerank": false` no corpo da requisição para comparar.
5. **Geração**: trechos numerados vão ao LLM (Claude ou OpenAI), que responde só com base neles e cita `[n]`.
6. **Interface**: a UI mostra as fontes realmente citadas, com trecho e relevância.

## Estrutura
```
app/main.py    endpoints FastAPI
app/ingest.py  parsing + chunking
app/rag.py     pgvector, retrieval, prompt, LLM
app/static/    interface web
```

## Próximos passos
Reranking, busca híbrida (BM25 + vetor), OCR para PDFs escaneados, autenticação e coleções por usuário, avaliação com RAGAS.
