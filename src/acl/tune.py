"""Tune action hyperparameters on the dev split only, by mean nDCG@300.

- expand_<group>: decay lambda and edge direction ("out" or "both"), per group
- fusion: the RRF constant k (needs dense scores for the dev tasks)

Each grid is printed in full and the best setting is written to
configs/tuned.json, which acl.phase_a reads. Dev tasks are never used to
evaluate. Exact ties go to direction "out" and the larger lambda: a tie means
the extra expansion costs nothing on dev, and choosing the smallest lambda
can make an action identical to `stop`, which Phase A could then not measure.
RRF ties go to the smaller k.

Usage: python -m acl.tune [--what expand rrf]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from acl import actions, dense, evaluate, linecounts, splits
from acl.bm25 import fill_budget
from acl.phase_a import load_chunks
from acl.prepare import OUT as PREPARED

BUDGET = 300
LAMBDAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
RRF_KS = (5, 10, 20, 30, 60, 100)


def load_dev(need_dense: bool) -> list[dict]:
    dev = set(splits.load()["dev"]["ids"])
    counts, blank = linecounts.load(), linecounts.load_blank()
    out = []
    for t in map(json.loads, open("data/tasks.jsonl")):
        if t["instance_id"] not in dev:
            continue
        d = PREPARED / t["instance_id"]
        chunks, bm25 = load_chunks(d)
        ids = [c.id for c in chunks]
        out.append({
            "id": t["instance_id"], "chunks": chunks, "ids": ids, "bm25": bm25,
            "edges": pl.read_parquet(d / "edges.parquet"),
            "dense": dense.load_scores(t["instance_id"], ids) if need_dense else None,
            "gold": evaluate.Gold(t, counts.get(t["instance_id"], {}), blank.get(t["instance_id"], {})),
        })
    return out


def mean_ndcg(tasks: list[dict], rank) -> float:
    return float(np.mean([
        evaluate.score(t["gold"], fill_budget([t["chunks"][i] for i in rank(t)], BUDGET), BUDGET)["ndcg"]
        for t in tasks]))


def tune_expand(tasks: list[dict]) -> dict:
    base = mean_ndcg(tasks, lambda t: actions.order(t["bm25"]))
    print(f"dev n={len(tasks)}  stop nDCG@{BUDGET} = {base:.4f}")
    best = {}
    for group in actions.EDGE_GROUPS:
        grid = {}
        for direction in actions.DIRECTIONS:
            for lam in LAMBDAS:
                grid[(direction, lam)] = mean_ndcg(tasks, lambda t: actions.expand(
                    t["bm25"], t["ids"], t["edges"], group, direction, lam))
        print(f"expand_{group}: " + "  ".join(f"{d}/{lam}={v:.4f}" for (d, lam), v in grid.items()))
        (direction, lam), v = max(grid.items(), key=lambda kv: (kv[1], kv[0][0] == "out", kv[0][1]))
        best[group] = {"direction": direction, "lam": lam}
        print(f"  -> {direction}, lambda {lam}: {v:.4f} ({v - base:+.4f} vs stop)")
    return best


def tune_rrf(tasks: list[dict]) -> int:
    grid = {k: mean_ndcg(tasks, lambda t: actions.fusion(t["bm25"], t["dense"], k)) for k in RRF_KS}
    print("fusion: " + "  ".join(f"k={k}: {v:.4f}" for k, v in grid.items()))
    k = max(grid, key=grid.get)
    print(f"  -> k = {k}")
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="*", default=["expand", "rrf"], choices=["expand", "rrf"])
    args = ap.parse_args()
    tasks = load_dev(need_dense="rrf" in args.what)
    saved = json.loads(actions.TUNED.read_text()) if actions.TUNED.exists() else {}
    if "expand" in args.what:
        saved["expand"] = tune_expand(tasks)
    if "rrf" in args.what:
        saved["rrf_k"] = tune_rrf(tasks)
    saved["objective"] = f"mean nDCG@{BUDGET} on dev ({len(tasks)} tasks)"
    actions.TUNED.write_text(json.dumps(saved, indent=1) + "\n")
    print(f"-> {actions.TUNED}")


if __name__ == "__main__":
    main()
