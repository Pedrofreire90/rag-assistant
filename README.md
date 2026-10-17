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
4. **Busca híbrida**: a busca vetorial (cosseno) e a busca lexical (full-text do PostgreSQL, índice GIN) rodam em paralelo e são combinadas por RRF. Use `"hybrid": false` para desligar.
5. **Reranking** (`POST /ask`): a busca vetorial recupera 20 candidatos por cosseno e um cross-encoder multilíngue (FlashRank) reordena e mantém os 5 melhores. Use `"rerank": false` no corpo da requisição para comparar.
6. **Geração**: trechos numerados vão ao LLM (Claude ou OpenAI), que responde só com base neles e cita `[n]`.
7. **Interface**: a UI mostra as fontes realmente citadas, com trecho e relevância.

## Estrutura
```
app/main.py    endpoints FastAPI
app/ingest.py  parsing + chunking
app/rag.py     pgvector, retrieval, prompt, LLM
app/static/    interface web
```

## Próximos passos
Reranking, busca híbrida (BM25 + vetor), OCR para PDFs escaneados, autenticação e coleções por usuário, avaliação com RAGAS.

## Avaliação

Mede em que posição o trecho correto aparece, em cada modo (vetorial, híbrida, com reranking, híbrida + reranking).

1. Copie `eval/questions.example.json` para `eval/questions.json` e escreva suas perguntas.
   - `question`: a pergunta.
   - `expect_contains`: textos que o trecho correto contém (sem acento/maiúscula; todos precisam estar no mesmo trecho).
   - `source` (opcional): nome do arquivo esperado. `answer_contains` (opcional): palavras que a resposta final deve ter.
2. Rode:
```bash
docker compose exec app python -m app.evaluate eval/questions.json
docker compose exec app python -m app.evaluate eval/questions.json --modes vetorial,hibrida+rerank --answers
```
Saída: Hit@1/3/5, MRR e tempo médio por modo, as perguntas que falharam e um JSON em `eval/results/`.
`--answers` chama o LLM (custa tokens) e confere `answer_contains`.

## Conversa com memória

A interface envia as últimas 6 mensagens junto com cada pergunta (`history` em `POST /ask`). Em perguntas de acompanhamento ("e o segundo?"), o LLM primeiro reescreve a pergunta de forma independente, e essa versão é usada na busca. A resposta final também recebe o histórico. A pergunta reescrita aparece na interface como "Pergunta interpretada". O botão **Nova conversa** limpa o histórico.

## Avaliação com RAGAS

Complementa a avaliação de recuperação (`app.evaluate`): em vez de só ver se o trecho certo apareceu, avalia a **resposta final** com um LLM "juiz".

| Métrica | Pergunta que responde |
|---|---|
| `fidelidade` | A resposta se apoia nos trechos recuperados, ou inventou algo? |
| `relevancia` | A resposta trata do que foi perguntado? |
| `precisao_ctx` | Os trechos úteis vêm primeiro, com pouco lixo? |
| `cobertura_ctx` | Os trechos trazem tudo que a resposta de referência precisa? |

```bash
cp eval/ragas_questions.example.json eval/ragas_questions.json   # escreva pergunta + resposta de referência
docker compose exec app python -m app.ragas_eval eval/ragas_questions.json --limit 3   # teste barato
docker compose exec app python -m app.ragas_eval eval/ragas_questions.json             # vetorial x híbrida+rerank
docker compose exec app python -m app.ragas_eval eval/ragas_questions.json --modes vetorial,hibrida,hibrida+rerank
```

Notas: cada pergunta consome ~10 chamadas do juiz por modo (custo). Configure `JUDGE_MODEL` com um modelo mais forte que o gerador. Notas do juiz variam um pouco entre execuções, então compare diferenças grandes, não de centésimos. O resultado detalhado (respostas, trechos e notas) fica em `eval/results/ragas-*.json`.
`app/ragas_compat.py` contorna um import quebrado do ragas 0.4.x com o langchain-community atual; pode ser removido quando o ragas corrigir.
