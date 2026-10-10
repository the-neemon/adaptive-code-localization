"""BM25 over code chunks, plus filling a ranked chunk list to a line budget."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse

from acl.chunking import Chunk

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i if in into is it its of on or "
    "so that the their there these this to was we were when which will with you".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased identifiers plus their snake_case and camelCase parts.

    `get_FOO_display` -> get_foo_display, get, foo, display. Keeping the whole
    identifier lets an exact name in the issue match strongly; the parts let
    `foo display` in prose match it too.
    """
    out: list[str] = []
    for word in _WORD.findall(text):
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        low = word.lower().strip("_")
        if len(parts) > 1 and len(low) > 1:
            out.append(low)
        out.extend(p for p in parts if len(p) > 1 and p not in STOPWORDS)
    return out


class BM25:
    """Okapi BM25 (Lucene idf variant) over a fixed list of documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, doc in enumerate(docs):
            for tok, tf in Counter(doc).items():
                j = self.vocab.setdefault(tok, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
        n_docs = len(docs)
        tf = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)),
                               shape=(n_docs, len(self.vocab)))
        dl = np.array([len(d) for d in docs], dtype=np.float32)
        avgdl = dl.mean() if n_docs else 1.0
        df = np.bincount(cols, minlength=len(self.vocab)).astype(np.float32)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # Precompute the saturated term weight per (doc, term) entry.
        norm = k1 * (1 - b + b * dl / avgdl)
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.weights = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: list[str]) -> np.ndarray:
        q = Counter(t for t in query if t in self.vocab)
        out = np.zeros(self.weights.shape[0], dtype=np.float32)
        for tok, qtf in q.items():
            j = self.vocab[tok]
            col = self.weights.getcol(j)
            out[col.indices] += qtf * self.idf[j] * col.data
        return out


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def chunk_document(c: Chunk) -> list[str]:
    # The path and qualified name carry a lot of signal (e.g. "admin/checks.py").
    return tokenize(f"{c.path} {c.name}\n{c.text}")


def fill_budget(ranked: list[Chunk], budget: int) -> list[tuple[str, int, int]]:
    """Turn ranked chunks into ranked (path, start, end) regions within `budget` lines.

    Matches SWE-Explore's recall@budget convention: regions are taken in rank
    order and the one that crosses the budget is cut off at the budget.
    """
    regions: list[tuple[str, int, int]] = []
    used = 0
    for c in ranked:
        for s, e in c.spans:
            take = min(e - s + 1, budget - used)
            regions.append((c.path, s, s + take - 1))
            used += take
            if used >= budget:
                return regions
    return regions
