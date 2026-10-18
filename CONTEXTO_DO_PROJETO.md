# Contexto do projeto: Assistente Inteligente com RAG

> Documento para retomar o desenvolvimento (inclusive numa conversa nova com uma IA).
> Mantenha-o atualizado: ao terminar uma melhoria, ajuste as seções "Estado atual" e "Pendências".

## 1. Objetivo

Assistente que responde perguntas sobre documentos (PDF, DOCX, planilhas) usando RAG, com citação das fontes e interface web simples. Projeto de portfólio; o dono está aprendendo, então **explique o porquê das mudanças e dê comandos completos**.

Stack: FastAPI, LangChain, PostgreSQL + pgvector, Docker, Claude (respostas) e OpenAI (embeddings).

## 2. Como rodar

```bash
cp .env.example .env      # preencha OPENAI_API_KEY e ANTHROPIC_API_KEY
docker compose up --build -d
# http://localhost:8000   (documentação da API: /docs)
docker compose logs -f app
```

Depois de mudar o `.env`: `docker compose down` e `docker compose up -d`. Depois de mudar o código ou o `requirements.txt`: `docker compose up --build -d`.

## 3. Estrutura

```
Dockerfile, docker-compose.yml, requirements.txt, .env.example, README.md
app/main.py       endpoints FastAPI
app/ingest.py     lê PDF (por página), DOCX, XLSX/CSV (por aba) e faz o chunking (1000 caracteres, sobreposição 150)
app/rag.py        pgvector, retrieve(), ask(), condense_question(), debug_retrieval(), prompts
app/hybrid.py     busca lexical (full-text do PostgreSQL) e fusão RRF
app/reranker.py   reranking com FlashRank (modelo multilíngue ms-marco-MultiBERT-L-12)
app/evaluate.py   avaliação de recuperação: Hit@1/3/5 e MRR por modo
app/static/index.html   interface web (upload, chat, seletores, "Nova conversa")
eval/             perguntas de teste (questions.json) e resultados (results/, gerado pelo container)
```

## 4. Como funciona

**Upload** (`POST /documents`): extrai texto, divide em trechos, gera embeddings (OpenAI `text-embedding-3-small`) e grava no pgvector com metadados `source` e `location` (página ou aba). Reenviar o mesmo arquivo substitui o anterior.

**Pergunta** (`POST /ask`):
1. Se houver histórico, o LLM reescreve a pergunta de forma independente ("e o prazo dele?" vira "Qual o prazo do contrato X?"). A versão reescrita é usada na busca.
2. Busca vetorial (cosseno) traz 20 candidatos.
3. Busca híbrida (opcional): a busca lexical também traz 20, e as duas listas são combinadas por RRF (`1/(60 + posição)`).
4. Reranking (opcional): FlashRank reordena e fica com os 5 melhores.
5. O LLM responde só com base nos trechos numerados e cita `[n]`. A interface mostra as fontes citadas.

Corpo do `/ask`: `question`, `k`, `rerank` (true/false/omitido), `hybrid` (idem), `history` (lista de `{role, content}`).
Se a parte lexical ou o reranker falharem, o sistema cai para a busca vetorial e registra no log.

`POST /debug/retrieval` (`question`, `contains`, `k`): mostra a posição do trecho que contém o texto `contains` em cada etapa (vetorial, lexical, fusão, reranker), sem chamar o LLM. É a ferramenta para descobrir onde um trecho certo se perde.

## 5. Variáveis de ambiente (.env)

| Variável | Para que serve |
|---|---|
| `OPENAI_API_KEY` | embeddings (sempre OpenAI) |
| `EMBEDDING_MODEL` | padrão `text-embedding-3-small` |
| `LLM_PROVIDER`, `LLM_MODEL` | `anthropic` (padrão) ou `openai`; modelo das respostas |
| `ANTHROPIC_API_KEY` | usada quando o provedor é anthropic |
| `RERANK`, `RERANK_CANDIDATES` | liga o reranking (padrão true) e quantos candidatos ele recebe (20) |
| `HYBRID`, `FTS_CONFIG`, `RRF_K` | liga a busca híbrida (true), idioma do full-text (`portuguese`) e constante do RRF (60) |

Observações: o `.env` NUNCA vai para o Git. Comentários no `.env` devem ficar em linhas próprias.

## 6. Avaliação

```bash
docker compose exec app python -m app.evaluate eval/questions.json --verbose
docker compose exec app python -m app.evaluate eval/questions.json --modes vetorial,hibrida+rerank --answers
```

`eval/questions.json`: lista de `{"question", "expect_contains": [...], "source"?, "answer_contains"?}`. `expect_contains` deve ser texto CURTO que aparece no trecho certo (a comparação ignora acento, maiúscula e quebras de linha). O JSON exige aspas retas, vírgulas entre itens e nenhuma aspa dupla dentro dos textos.
Os arquivos de `eval/results/` são criados pelo container como root: se o Git reclamar de permissão, `sudo chown -R $USER:$USER eval/results`. Coloque `eval/results/` no `.gitignore`.

## 7. Estado atual do repositório

- `main`: tudo acima (busca vetorial, híbrida, reranking, memória de conversa, avaliador de recuperação, interface).
- Branch `feat/ragas` e etiqueta `backup-com-ragas`: avaliação das respostas com RAGAS (métricas: fidelidade, relevância, precisão e cobertura do contexto). NÃO foi juntado ao `main`. Para reaplicar: use o `ragas-patch.zip` por cima do projeto e reconstrua a imagem. Observações: fixa `ragas==0.4.3`; precisa de um arquivo de compatibilidade porque o ragas 0.4.x importa um módulo removido do langchain-community atual; custa ~10 chamadas do LLM juiz por pergunta e por modo; o campo `reference` (resposta correta) é obrigatório nas perguntas.
- Login e separação por usuário: implementado em um `auth-patch.zip`, **nunca commitado no repositório**. Guarde o zip ou crie um branch para ele. Usa cadastro/login (argon2), JWT em cookie HttpOnly, filtro por `user_id` em todas as buscas; documentos antigos precisam de `python -m app.admin claim EMAIL`.

## 8. Fluxo de Git que estamos usando

```bash
git switch main
git switch -c feat/nome-da-melhoria
# altera, testa com docker compose up --build -d
git add . && git commit -m "feat: ..."
git push -u origin feat/nome-da-melhoria
git switch main && git merge --no-edit feat/nome-da-melhoria && git push
```

Antes de começar algo novo: `git status` limpo e `git branch --no-merged main` para ver o que ficou sem juntar. Se o terminal ficar preso em `(END)`, aperte `q`.

## 9. Problemas conhecidos e hipóteses

- **Perguntas sobre estrutura** ("Qual é o canto XXI?") só funcionaram no modo vetorial puro. Hipóteses: o ranking lexical do PostgreSQL (`ts_rank_cd`) não pondera palavras raras (sem IDF); a fusão RRF pode empurrar o trecho certo para fora do top 5; o reranker avalia "responde à pergunta?" e um trecho com o título não parece resposta. Ainda NÃO confirmado: use `/debug/retrieval` com `contains` para ver em que etapa o trecho cai.
- **Resultados preliminares do RAGAS** (apenas 2 a 3 perguntas, inconclusivos): `hibrida+rerank` teve precisão e cobertura menores que `vetorial`, e fidelidade/relevância 0.00 que podem ser respostas de recusa ("não encontrei").
- PDFs escaneados (só imagem) não são indexados (não há OCR).
- A pergunta com `??` duplicado no conjunto de teste deve ser corrigida.

## 10. Pendências (em ordem sugerida)

1. Ampliar `eval/questions.json` para ~15 perguntas, misturando conteúdo, termo raro, dado exato e estrutura.
2. Investigar com `/debug/retrieval` os casos que falharam e corrigir a busca lexical (exigir termos raros primeiro, ou dar peso a eles) e/ou os parâmetros da fusão; medir antes e depois com `app.evaluate`.
3. OCR para PDFs escaneados.
4. Login e separação por usuário (reaplicar o `auth-patch.zip`).
5. Reaplicar o RAGAS quando for medir a qualidade das respostas.

## 11. Como pedir ajuda numa conversa nova

Cole este documento e diga o objetivo. Para mudanças de código, cole também os arquivos envolvidos (por exemplo `app/rag.py`) e, em caso de erro, a mensagem COMPLETA e o comando que a gerou. Peça sempre: "explique o que cada mudança faz, dê os comandos completos e o jeito de testar".
