"""Phase A runner: every action on every task, scored at every budget.

Reads data/prepared/<id>/ (Stage 1), data/dense/ scores and data/line_counts.json,
and writes one row per (task, action, budget) to data/results/phase_a.jsonl:
  {"instance_id", "repo", "source", "split", "action", "budget", "metrics": {...}}
plus the ranked regions at the largest budget, so any smaller budget can be
re-scored without re-running.

Actions whose inputs are missing for a task (e.g. no dense scores yet) are
skipped and counted. As a consistency check against Stage 1, the `stop`
regions must equal the seed regions stored in seed.json.

Usage: python -m acl.phase_a [--actions stop dense fusion] [--ids ...] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from acl import actions, dense, evaluate, linecounts, splits
from acl.bm25 import fill_budget
from acl.chunking import Chunk
from acl.prepare import BUDGETS, OUT as PREPARED

OUT = Path("data/results/phase_a.jsonl")


def load_chunks(d: Path) -> tuple[list[Chunk], np.ndarray]:
    df = pl.read_parquet(d / "chunks.parquet", columns=["id", "path", "starts", "ends", "bm25"])
    chunks = [Chunk(i, p, "", "", list(zip(s, e)))
              for i, p, s, e in zip(df["id"], df["path"], df["starts"], df["ends"])]
    return chunks, df["bm25"].to_numpy()


def run_task(task: dict, counts: dict, blank: dict, split: str, names: list[str],
             skipped: Counter) -> list[dict]:
    d = PREPARED / task["instance_id"]
    chunks, bm25 = load_chunks(d)
    try:
        dense_scores = dense.load_scores(task["instance_id"], [c.id for c in chunks])
    except FileNotFoundError:
        dense_scores = None
    gold = evaluate.Gold(task, counts, blank)
    seed = json.loads((d / "seed.json").read_text())
    rows = []
    for name in names:
        if name in actions.NEEDS_DENSE and dense_scores is None:
            skipped[name] += 1
            continue
        t0 = time.time()
        ranked = [chunks[i] for i in actions.ACTIONS[name](bm25=bm25, dense=dense_scores)]
        secs = time.time() - t0
        regions = {b: fill_budget(ranked, b) for b in BUDGETS}
        if name == "stop":
            for b in BUDGETS:
                assert [tuple(r) for r in seed["regions"][str(b)]] == regions[b], \
                    f"stop != Stage 1 seed for {task['instance_id']} at B={b}"
        for b in BUDGETS:
            rows.append({
                "instance_id": task["instance_id"], "repo": task["repo"], "source": task["source"],
                "split": split, "action": name, "budget": b, "secs": round(secs, 3),
                "metrics": evaluate.score(gold, regions[b], b),
                **({"regions": regions[b]} if b == max(BUDGETS) else {}),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", nargs="*", default=list(actions.ACTIONS))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open("data/tasks.jsonl")]
    if args.ids:
        tasks = [t for t in tasks if t["instance_id"] in set(args.ids)]
    tasks = [t for t in tasks if (PREPARED / t["instance_id"] / "seed.json").exists()]
    counts = linecounts.load()
    blank = linecounts.load_blank()
    split_of = splits.split_of(splits.load())
    skipped: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w") as f:
        for t in tasks:
            for r in run_task(t, counts.get(t["instance_id"], {}), blank.get(t["instance_id"], {}),
                              split_of[t["instance_id"]],
                              args.actions, skipped):
                f.write(json.dumps(r) + "\n")
                rows.append(r)
    print(f"{len(tasks)} prepared tasks, {len(rows)} rows -> {args.out}; skipped {dict(skipped)}")
    keys = ["recall", "ndcg", "perfect_recall", "perfect_recall_lines", "noise_region", "file_recall_patch", "off_recall", "off_ndcg"]
    print("action  B    n   " + "  ".join(f"{k:>10s}" for k in keys))
    for (a, b), m in evaluate.summarise(rows, ["action", "budget"]).items():
        n = sum(1 for r in rows if r["action"] == a and r["budget"] == b)
        print(f"{a:7s} {b:4d} {n:4d} " + "  ".join(f"{m.get(k, float('nan')):10.4f}" for k in keys))


if __name__ == "__main__":
    main()
