"""Retrieval actions: each turns a prepared task into a full ranking of its chunks.

- stop:   the BM25 seed ranking unchanged
- dense:  frozen CodeRankEmbed cosine scores (acl.dense)
- fusion: reciprocal-rank fusion of the BM25 and dense rankings
- expand_<e>: the top-k seed chunks plus their depth-1 neighbours along edge
  group e, each neighbour scored max(own BM25, lambda * best parent's BM25)
  (proposal eq. 1, min-max normalised BM25)

Every action ranks all chunks; acl.bm25.fill_budget then cuts the ranking to
the budget, so no action can win by returning more code. Hyperparameters
(RRF k, and lambda and edge direction per edge group) are tuned on dev only by
acl.tune and stored in configs/tuned.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from acl.bm25 import minmax

RRF_K = 60  # Cormack et al. (2009); default until tuned
TOP_K = 10  # seed parents for expansion (proposal)
# Edge groups (stored edge types, see acl.graph). The proposal treats imports
# and containment as one group.
EDGE_GROUPS = {"calls": ("invokes",), "imports": ("imports", "contains"), "inherit": ("inherits",)}
# "out": follow edges from a parent (its callees, imports, members, bases);
# "both": also against them (callers, importers, containers, subclasses).
DIRECTIONS = ("out", "both")
TUNED = Path("configs/tuned.json")


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


def neighbour_boost(ids: list[str], s: np.ndarray, parents: np.ndarray, edges: pl.DataFrame,
                    group: str, direction: str) -> dict[int, float]:
    """{chunk index: best parent score} for depth-1 neighbours of `parents` along `group`."""
    pscore = {ids[p]: float(s[p]) for p in parents}
    e = edges.filter(pl.col("type").is_in(list(EDGE_GROUPS[group])))
    hops = [e.filter(pl.col("src").is_in(list(pscore))).select(parent="src", nb="dst")]
    if direction == "both":
        hops.append(e.filter(pl.col("dst").is_in(list(pscore))).select(parent="dst", nb="src"))
    best: dict[str, float] = {}
    for parent, nb in pl.concat(hops).iter_rows():
        best[nb] = max(best.get(nb, 0.0), pscore[parent])
    index = {c: i for i, c in enumerate(ids)}
    return {index[nb]: v for nb, v in best.items()}


def expand(bm25: np.ndarray, ids: list[str], edges: pl.DataFrame, group: str,
           direction: str, lam: float, k: int = TOP_K, **_) -> np.ndarray:
    """Proposal eq. 1: s_e(n) = max(s(n), lam * max over parents p of s(p)).

    Seed chunks keep their own score; ties fall back to the seed order, so with
    no neighbours this is exactly `stop`.
    """
    seed = order(bm25)
    s = minmax(bm25)
    new = s.copy()
    for i, best in neighbour_boost(ids, s, seed[:k], edges, group, direction).items():
        new[i] = max(new[i], lam * best)
    return np.lexsort((ranks(seed), -new))


def load_tuned(path: Path = TUNED) -> dict:
    """Dev-tuned hyperparameters; defaults until acl.tune has written them."""
    t = {"rrf_k": RRF_K, "expand": {g: {"direction": "out", "lam": 0.8} for g in EDGE_GROUPS}}
    if path.exists():
        saved = json.loads(path.read_text())
        t["rrf_k"] = saved.get("rrf_k", t["rrf_k"])
        t["expand"].update(saved.get("expand", {}))
    return t


def _expand_action(group: str):
    def act(bm25, ids, edges, tuned, **_):
        return expand(bm25, ids, edges, group, **tuned["expand"][group])
    return act


ACTIONS = {"stop": stop, "dense": dense_action,
           "fusion": lambda bm25, dense, tuned, **_: fusion(bm25, dense, tuned["rrf_k"]),
           **{f"expand_{g}": _expand_action(g) for g in EDGE_GROUPS}}
NEEDS_DENSE = {"dense", "fusion"}
NEEDS_EDGES = {f"expand_{g}" for g in EDGE_GROUPS}
