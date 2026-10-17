"""Avaliação RAGAS: mede a qualidade das RESPOSTAS e do CONTEXTO recuperado, por modo de busca.

Métricas:
  fidelidade     (faithfulness)      a resposta se apoia nos trechos? (detecta invenção)
  relevancia     (response relevancy) a resposta trata da pergunta?
  precisao_ctx   (context precision)  os trechos úteis vêm primeiro e há pouco lixo?
  cobertura_ctx  (context recall)     os trechos trazem o necessário para a resposta de referência?

Uso: docker compose exec app python -m app.ragas_eval eval/ragas_questions.json [--modes a,b] [--limit N] [--k 5]
Cada pergunta consome ~10 chamadas do LLM juiz por modo; comece com --limit 5.
"""
import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

from . import ragas_compat  # noqa: F401  (precisa vir antes de qualquer import do ragas)

# nome do modo -> (busca híbrida, reranking)
MODES = {"vetorial": (False, False), "hibrida": (True, False),
         "vetorial+rerank": (False, True), "hibrida+rerank": (True, True)}

DEFAULT_MODES = ["vetorial", "hibrida+rerank"]  # linha de base x pipeline completo
NAMES = {"faithfulness": "fidelidade", "answer_relevancy": "relevancia",
         "llm_context_precision_with_reference": "precisao_ctx", "context_recall": "cobertura_ctx"}


def load_cases(path: str, limit: int | None) -> list:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, c in enumerate(cases, start=1):
        if not c.get("question") or not c.get("reference"):
            raise SystemExit(f"Caso {i}: precisa de 'question' e 'reference' (a resposta correta, em 1 ou 2 frases).")
    return cases[:limit] if limit else cases


def judge_models():
    """LLM juiz e embeddings. JUDGE_MODEL vazio = usa LLM_MODEL (ideal: um modelo mais forte que o gerador)."""
    model = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    if os.getenv("LLM_PROVIDER", "anthropic") == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, temperature=0, max_tokens=4096)  # JSON do juiz é longo
    from langchain_openai import OpenAIEmbeddings
    return llm, OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))


def build_samples(rag, cases: list, mode: str, k: int) -> list:
    """Roda o pipeline real e monta as amostras do RAGAS (pergunta, trechos, resposta, referência)."""
    hybrid, rerank = MODES[mode]
    samples = []
    for c in cases:
        ranked, *_ = rag.retrieve(c["question"], k, rerank=rerank, hybrid_search=hybrid)
        if not ranked:
            raise SystemExit("Nenhum documento indexado. Envie os documentos antes de avaliar.")
        answer = re.sub(r"\s*\[\d+\]", "", rag.generate(c["question"], ranked))  # tira as marcas de citação
        samples.append({"user_input": c["question"], "retrieved_contexts": [d.page_content for d, _, _ in ranked],
                        "response": answer, "reference": c["reference"]})
    return samples


def score(samples: list, llm, emb):
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(llm), embeddings=LangchainEmbeddingsWrapper(emb),
        run_config=RunConfig(max_workers=int(os.getenv("RAGAS_WORKERS", "4")),
                             timeout=int(os.getenv("RAGAS_TIMEOUT", "180"))),
        show_progress=True)
    return result.to_pandas()


def metric_columns(df, samples: list) -> list:
    inputs = set(samples[0]) | {"reference_contexts"}
    return [c for c in df.columns if c not in inputs and str(df[c].dtype) != "object"]


def summarize(df, cols: list) -> dict:
    out = {}
    for c in cols:
        vals = [v for v in df[c].tolist() if v is not None and not math.isnan(v)]
        out[NAMES.get(c, c)] = {"media": sum(vals) / len(vals) if vals else float("nan"), "falhas": len(df) - len(vals)}
    return out


def report(summaries: dict, details: dict):
    names = list(next(iter(summaries.values())))
    print(f"\n{'Modo':<18}" + "".join(f"{n:>15}" for n in names))
    for mode, s in summaries.items():
        print(f"{mode:<18}" + "".join(f"{s[n]['media']:>15.2f}" for n in names))
    fails = {m: sum(v["falhas"] for v in s.values()) for m, s in summaries.items()}
    if any(fails.values()):
        print("\nAvaliações que o juiz não conseguiu calcular (contam como ausentes):", fails)
    fid = next((n for n in names if n == "fidelidade"), None)
    if fid:
        print("\nRespostas menos fiéis aos trechos (possível invenção) — confira estas primeiro:")
        for mode, rows in details.items():
            worst = sorted((r for r in rows if r.get(fid) is not None and not math.isnan(r[fid])), key=lambda r: r[fid])[:3]
            print(f"\n  {mode}:")
            for r in worst:
                print(f"    {r[fid]:.2f}  {r['pergunta']}")


def _rows(df, cols: list, samples: list) -> list:
    rows = []
    for idx, s in enumerate(samples):
        scores = {}
        for c in cols:
            v = df.iloc[idx][c]
            scores[NAMES.get(c, c)] = None if v is None or math.isnan(v) else float(v)
        rows.append({"pergunta": s["user_input"], "resposta": s["response"], "referencia": s["reference"],
                     "trechos": s["retrieved_contexts"], **scores})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"separados por vírgula: {', '.join(MODES)}")
    ap.add_argument("--limit", type=int, help="usa só as N primeiras perguntas (bom para testar o custo)")
    ap.add_argument("--k", type=int, default=5, help="trechos enviados ao LLM")
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",")]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"Modo desconhecido: {bad}")

    from . import rag
    cases = load_cases(a.file, a.limit)
    print(f"{len(cases)} perguntas x {len(modes)} modos ≈ {len(cases) * len(modes) * 10} chamadas do juiz.")
    llm, emb = judge_models()
    rag.retrieve(cases[0]["question"], a.k)  # aquece conexão e reranker

    summaries, details = {}, {}
    for m in modes:
        print(f"\n== {m} ==")
        samples = build_samples(rag, cases, m, a.k)
        df = score(samples, llm, emb)
        cols = metric_columns(df, samples)
        summaries[m] = summarize(df, cols)
        details[m] = _rows(df, cols, samples)
    report(summaries, details)
    out = Path(a.file).parent / "results" / f"ragas-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"resumo": summaries, "detalhe": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultado salvo em {out}")


if __name__ == "__main__":
    main()