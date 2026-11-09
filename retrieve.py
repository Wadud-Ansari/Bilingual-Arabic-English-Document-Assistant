"""Dense, hybrid (dense + BM25 via reciprocal rank fusion) and reranked retrieval."""
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .text import normalize_ar, tokenize


class Retriever:
    def __init__(
        self,
        index_dir: str = "index",
        embed_model: str = "BAAI/bge-m3",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ):
        index_dir = Path(index_dir)
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.embs = np.load(index_dir / "embs.npy")
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(embed_model, device=device)
        self.reranker = CrossEncoder(rerank_model, max_length=512, device=device)

    # ---- score helpers -------------------------------------------------
    def _mask(self, scores: np.ndarray, source: str | None) -> np.ndarray:
        """Restrict search to one file (used for cross-lingual tests)."""
        if source is None:
            return scores
        blocked = np.array([c["source"] != source for c in self.chunks])
        return np.where(blocked, -np.inf, scores)

    def _dense_scores(self, query: str, source: str | None) -> np.ndarray:
        q = self.embedder.encode([normalize_ar(query)], normalize_embeddings=True)[0]
        return self._mask(self.embs @ q, source)

    def _bm25_scores(self, query: str, source: str | None) -> np.ndarray:
        return self._mask(np.asarray(self.bm25.get_scores(tokenize(query))), source)

    # ---- public retrieval modes ---------------------------------------
    def dense(self, query: str, source: str | None = None, top_n: int = 10):
        scores = self._dense_scores(query, source)
        order = np.argsort(-scores)[:top_n]
        return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i])]

    def hybrid(self, query: str, source: str | None = None, top_n: int = 10,
               k_each: int = 20, rrf_k: int = 60, w_bm25: float = 0.5):
        """Reciprocal rank fusion of dense and BM25 rankings (BM25 down-weighted)."""
        dense = self._dense_scores(query, source)
        bm = self._bm25_scores(query, source)
        fused: dict[int, float] = {}
        for rank, i in enumerate(np.argsort(-dense)[:k_each]):
            if np.isfinite(dense[i]):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, i in enumerate(np.argsort(-bm)[:k_each]):
            if np.isfinite(bm[i]) and bm[i] > 0:   # keyword match only counts if there is one
                fused[int(i)] = fused.get(int(i), 0.0) + w_bm25 / (rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]

    def rerank(self, query: str, source: str | None = None, top_n: int = 4,
               n_candidates: int = 30):
        """Hybrid candidates re-scored by a cross-encoder. Scores are roughly 0-1."""
        cands = [i for i, _ in self.hybrid(query, source, top_n=n_candidates)]
        if not cands:
            return []
        scores = self.reranker.predict([(query, self.chunks[i]["text"]) for i in cands])
        order = np.argsort(-scores)[:top_n]
        return [(cands[j], float(scores[j])) for j in order]
