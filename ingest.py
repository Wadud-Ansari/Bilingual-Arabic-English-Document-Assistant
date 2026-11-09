"""Build the search index from PDFs.

Usage:
    python -m rag.ingest data/amr_en.pdf data/amr_ar.pdf --out index
Creates index/chunks.json (text + source + page) and index/embs.npy (embeddings).
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from .text import normalize_ar

MD_LINK = re.compile(r"\[([^\]]*)\]\((?:https?:|mailto:)[^)]*\)")
SENT_SPLIT = re.compile(r"(?<=[.!?؟؛])\s+")
# Pages containing these are copyright/cataloguing boilerplate, not content.
SKIP_MARKERS = ["WHO Library Cataloguing", "All rights reserved", "جميع الحقوق محفوظة"]


def clean_page(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)      # fixes Arabic presentation forms
    text = MD_LINK.sub(r"\1", text)                 # keep link text only
    text = re.sub(r"\n\s*\n", "<PARA>", text)       # remember paragraph breaks
    text = text.replace("\n", " ")                  # join wrapped lines
    text = text.replace("<PARA>", "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 800, overlap_sentences: int = 1) -> list[str]:
    """Sentence-aware chunking with a small sentence overlap."""
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    chunks, current, length = [], [], 0
    for s in sentences:
        if length + len(s) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(x) for x in current)
        current.append(s)
        length += len(s)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(pdf_paths) -> list[dict]:
    chunks = []
    for path in pdf_paths:
        path = Path(path)
        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean_page(page.extract_text() or "")
            if len(text) < 200 or any(m in text for m in SKIP_MARKERS):
                continue
            for c in chunk_text(text):
                chunks.append({"source": path.name, "page": page_no, "text": c})
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", help="PDF files to index")
    ap.add_argument("--out", default="index", help="output directory")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

    chunks = build_chunks(args.pdfs)
    print(f"{len(chunks)} chunks")
    model = SentenceTransformer(args.embed_model)
    # Embed the normalized text; the original text is kept for display.
    embs = model.encode(
        [normalize_ar(c["text"]) for c in chunks],
        batch_size=32, normalize_embeddings=True, show_progress_bar=True,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embs.npy", embs)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"saved {embs.shape} embeddings to {out}/")


if __name__ == "__main__":
    main()
