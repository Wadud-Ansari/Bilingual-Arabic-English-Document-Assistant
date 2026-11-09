"""Ask one question from the command line.

    python -m rag.ask "What are the five strategic objectives?" --backend api
    python -m rag.ask "ما هي الأهداف الاستراتيجية؟" --backend local
"""
import argparse
import os

from .generate import answer, make_api_generator, make_local_generator
from .retrieve import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default="index")
    ap.add_argument("--backend", choices=["local", "api"], default="api")
    ap.add_argument("--min-score", type=float, default=0.30)
    args = ap.parse_args()

    retriever = Retriever(args.index)
    gen = make_local_generator() if args.backend == "local" else make_api_generator(token=os.environ.get("HF_TOKEN"))
    text, hits = answer(args.question, retriever, gen, min_score=args.min_score)
    print(f"\n{text}\n")
    for n, (i, s) in enumerate(hits, 1):
        c = retriever.chunks[i]
        print(f"[{n}] {c['source']} p.{c['page']} (score {s:.2f})")


if __name__ == "__main__":
    main()
