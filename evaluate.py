"""Measure retrieval quality and the refusal threshold.

    python -m rag.evaluate --index index
Prints markdown tables you can paste into the README.
"""
import argparse
import json
from pathlib import Path

from .retrieve import Retriever

SETTINGS = [
    ("EN question -> all docs", "en", None),
    ("AR question -> all docs", "ar", None),
    ("EN question -> Arabic doc", "en", "amr_ar.pdf"),
    ("AR question -> English doc", "ar", "amr_en.pdf"),
]


def make_systems(r: Retriever):
    return {
        "dense": lambda q, s: [i for i, _ in r.dense(q, s, top_n=10)],
        "hybrid": lambda q, s: [i for i, _ in r.hybrid(q, s, top_n=10)],
        "hybrid+rerank": lambda q, s: [i for i, _ in r.rerank(q, s, top_n=10, n_candidates=30)],
    }


def evaluate(rank_fn, chunks, eval_set, lang, source):
    """Hit@1/3/5 and MRR, where a hit is any retrieved chunk from a correct page."""
    h1 = h3 = h5 = mrr = 0.0
    for ex in eval_set:
        ranked = rank_fn(ex[lang], source)
        pos = next((r for r, i in enumerate(ranked) if chunks[i]["page"] in ex["pages"]), None)
        if pos is not None:
            h1 += pos == 0
            h3 += pos < 3
            h5 += pos < 5
            mrr += 1 / (pos + 1)
    n = len(eval_set)
    return h1 / n, h3 / n, h5 / n, mrr / n


def retrieval_report(r: Retriever, eval_set):
    print(f"\n### Retrieval ({len(eval_set)} questions)\n")
    print("| Setting | System | Hit@1 | Hit@3 | Hit@5 | MRR |")
    print("|---|---|---|---|---|---|")
    for sname, lang, src in SETTINGS:
        for yname, fn in make_systems(r).items():
            h1, h3, h5, m = evaluate(fn, r.chunks, eval_set, lang, src)
            print(f"| {sname} | {yname} | {h1:.2f} | {h3:.2f} | {h5:.2f} | {m:.2f} |")


def refusal_report(r: Retriever, refusal_set, min_score: float):
    """Top reranker score decides answer vs refuse; report accuracy on both groups."""
    def top(q):
        hits = r.rerank(q, top_n=1, n_candidates=30)
        return hits[0][1] if hits else 0.0

    on = [(q, top(q)) for q in refusal_set["on_topic"]]
    off = [(q, top(q)) for q in refusal_set["off_topic"]]
    answered_ok = sum(s >= min_score for _, s in on)
    refused_ok = sum(s < min_score for _, s in off)
    print(f"\n### Refusal gate (min_score = {min_score})\n")
    print(f"- On-topic answered: {answered_ok}/{len(on)}  (lowest score {min(s for _, s in on):.2f})")
    print(f"- Off-topic refused: {refused_ok}/{len(off)}  (highest score {max(s for _, s in off):.2f})")
    for q, s in on + off:
        print(f"  - {s:.2f}  {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--eval", default="evaluation/eval_set.json")
    ap.add_argument("--refusal", default="evaluation/refusal_set.json")
    ap.add_argument("--min-score", type=float, default=0.30)
    args = ap.parse_args()

    r = Retriever(args.index)
    eval_set = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    refusal_set = json.loads(Path(args.refusal).read_text(encoding="utf-8"))
    retrieval_report(r, eval_set)
    refusal_report(r, refusal_set, args.min_score)


if __name__ == "__main__":
    main()
