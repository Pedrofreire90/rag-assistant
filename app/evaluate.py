"""Avaliação da recuperação: Hit@k e MRR para cada modo de busca, a partir de um arquivo de perguntas.

Uso: docker compose exec app python -m app.evaluate eval/questions.json [--modes a,b] [--answers]
"""
import argparse
import json
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path

# nome -> (busca híbrida, reranking)
MODES = {"vetorial": (False, False), "hibrida": (True, False),
         "vetorial+rerank": (False, True), "hibrida+rerank": (True, True)}
DEPTH = 10  # profundidade do ranking usada para calcular o MRR


def norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))


def is_relevant(doc, case: dict) -> bool:
    if case.get("source") and doc.metadata.get("source") != case["source"]:
        return False
    body = norm(doc.page_content)
    return all(norm(x) in body for x in case["expect_contains"])


def first_rank(ranked: list, case: dict):
    for i, (d, _, _) in enumerate(ranked, start=1):
        if is_relevant(d, case):
            return i
    return None


def summarize(rows: list) -> dict:
    n = len(rows)
    hit = lambda k: sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n
    out = {"hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5),
           "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
           "ms": statistics.mean(r["ms"] for r in rows)}
    done = [r["answer_ok"] for r in rows if "answer_ok" in r]
    if done:
        out["resp_ok"] = sum(done) / len(done)
    return out


def load_cases(path: str) -> list:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, c in enumerate(cases, start=1):
        if not c.get("question") or not c.get("expect_contains"):
            raise SystemExit(f"Caso {i}: precisa de 'question' e 'expect_contains' (lista de textos).")
    return cases


def run(path: str, modes: list, answers: bool) -> dict:
    from . import rag  # import tardio: só precisa do banco/chaves ao executar
    cases = load_cases(path)
    rag.retrieve(cases[0]["question"], DEPTH)  # aquece conexão e modelo, para não distorcer o tempo
    rows = {m: [] for m in modes}
    for m in modes:
        hyb, rer = MODES[m]
        for c in cases:
            t0 = time.perf_counter()
            ranked, *_ = rag.retrieve(c["question"], DEPTH, rerank=rer, hybrid_search=hyb)
            row = {"question": c["question"], "rank": first_rank(ranked, c),
                   "ms": (time.perf_counter() - t0) * 1000}
            if answers and c.get("answer_contains"):
                ans = rag.ask(c["question"], 5, rer, hyb)["answer"]
                row["answer_ok"] = all(norm(x) in norm(ans) for x in c["answer_contains"])
            rows[m].append(row)
    return rows


def report(rows: dict):
    res = {m: summarize(r) for m, r in rows.items()}
    cols = ["hit@1", "hit@3", "hit@5", "mrr", "ms"] + (["resp_ok"] if any("resp_ok" in s for s in res.values()) else [])
    n = len(next(iter(rows.values())))
    print(f"\n{n} perguntas\n")
    print(f"{'Modo':<18}" + "".join(f"{c:>9}" for c in cols))
    for m, s in res.items():
        print(f"{m:<18}" + "".join(f"{s[c]:>9.0f}" if c == "ms" else f"{s.get(c, float('nan')):>9.2f}" for c in cols))
    print("\nPerguntas fora do top 5 (posição do trecho certo, '-' = não achou em 10):")
    for m, rs in rows.items():
        miss = [r for r in rs if not r["rank"] or r["rank"] > 5]
        print(f"\n  {m}: {'nenhuma' if not miss else ''}")
        for r in miss:
            print(f"    - {r['question']}  [{r['rank'] or '-'}]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--modes", default=",".join(MODES), help=f"separados por vírgula: {', '.join(MODES)}")
    ap.add_argument("--answers", action="store_true", help="também gera respostas (usa o LLM) e confere 'answer_contains'")
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",")]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"Modo desconhecido: {bad}")
    rows = run(a.file, modes, a.answers)
    res = report(rows)
    out = Path(a.file).parent / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"resumo": res, "detalhe": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultado salvo em {out}")


if __name__ == "__main__":
    main()
