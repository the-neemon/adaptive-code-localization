"""Retrieval actions: each turns a prepared task into a full ranking of its chunks.

- stop:   the BM25 seed ranking unchanged
- dense:  frozen CodeRankEmbed cosine scores (acl.dense)
- fusion: reciprocal-rank fusion of the BM25 and dense rankings

Every action ranks all chunks; acl.bm25.fill_budget then cuts the ranking to
the budget, so no action can win by returning more code. The expand(e)
actions are added here in Stage 3.
"""
from __future__ import annotations

import numpy as np

RRF_K = 60  # Cormack et al. (2009); tuned on dev only


def order(scores: np.ndarray) -> np.ndarray:
    """Chunk indices by descending score; ties keep chunk order (as the seed does)."""
    return np.argsort(-scores, kind="stable")


def ranks(order_: np.ndarray) -> np.ndarray:
    """1-based rank of each chunk index."""
    r = np.empty(len(order_), dtype=np.int64)
    r[order_] = np.arange(1, len(order_) + 1)
    return r


def rrf(orders: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Reciprocal-rank fusion: score(c) = sum over rankings of 1 / (k + rank(c))."""
    return sum(1.0 / (k + ranks(o)) for o in orders)


def stop(bm25: np.ndarray, dense: np.ndarray | None = None, **_) -> np.ndarray:
    return order(bm25)


def dense_action(bm25: np.ndarray, dense: np.ndarray, **_) -> np.ndarray:
    return order(dense)


def fusion(bm25: np.ndarray, dense: np.ndarray, rrf_k: int = RRF_K, **_) -> np.ndarray:
    return order(rrf([order(bm25), order(dense)], rrf_k))


ACTIONS = {"stop": stop, "dense": dense_action, "fusion": fusion}
NEEDS_DENSE = {"dense", "fusion"}
