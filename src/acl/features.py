"""Post-seed features for the router: cheap signals available right after BM25.

Computed from the issue text, the BM25 seed, the repository graph and (if
present) the dense scores only. Nothing here reads the patch or the ground
truth, so the features cannot leak the answer.

| Feature | Meaning |
|---|---|
| has_traceback | issue contains a Python traceback |
| names_py_path | issue mentions a path ending in .py |
| names_repo_file | issue mentions a .py path (or its suffix) that exists in the snapshot |
| n_known_identifiers | distinct code-like identifiers in the issue that name a chunk in the repo |
| has_env_dump | issue pastes an environment dump (pip list, versions block, or 3+ "name: 1.2" lines) |
| issue_tokens | issue length in BM25 tokens |
| bm25_margin | 1 - normalised score of the 2nd chunk (top-1 vs runner-up) |
| bm25_topk_margin | 1 - mean normalised score of chunks 2..k |
| seed_files | distinct files among the top-k seed chunks |
| top_degree, top_degree_<group> | graph degree of the top seed chunk, in total and per edge group |
| topk_degree_mean | mean graph degree of the top-k seed chunks |
| bm25_dense_overlap | share of the BM25 top-k also in the dense top-k (None without dense) |
| source | "verified" or "pro" |

Usage: python -m acl.features [--tasks PATH] [--out PATH]  (default: data/results/features.jsonl)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from acl import actions, dense
from acl.bm25 import minmax, tokenize
from acl.prepare import OUT as PREPARED

OUT = Path("data/results/features.jsonl")
K = actions.TOP_K

_TRACEBACK = re.compile(r"Traceback \(most recent call last\)|^\s*File \"[^\"]+\", line \d+", re.M)
_PY_PATH = re.compile(r"[\w./-]*\w\.py\b")
_ENV_DUMP = re.compile(r"pip (list|freeze)|<details>|INSTALLED VERSIONS|show_versions|"
                       r"^\s*(Package|Version)\s+-{3,}", re.M | re.I)
# "numpy: 1.15.2", "attrs==19.3.0", "attrs    19.3.0"; not "x = 1.5", which repro scripts contain
_VERSION_LINE = re.compile(r"^\s*[A-Za-z][\w.-]*(?::\s*|==|\s+)v?\d+\.\d+", re.M)
# Identifiers that look like code, not prose: dotted names, snake_case,
# CamelCase with an inner capital, or a name followed by "(".
_CODE_ID = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b|\b\w*_\w+\b|\b[A-Z][a-z]+[A-Z]\w*\b|\b\w+(?=\()")


def text_features(issue: str, paths: list[str], names: set[str]) -> dict:
    py_paths = {p.lstrip("./") for p in _PY_PATH.findall(issue)}
    ids = {part for tok in _CODE_ID.findall(issue) for part in tok.split(".") if len(part) > 2}
    return {
        "has_traceback": bool(_TRACEBACK.search(issue)),
        "names_py_path": bool(py_paths),
        "names_repo_file": any(p == q or p.endswith("/" + q) for q in py_paths for p in paths),
        "n_known_identifiers": len(ids & names),
        "has_env_dump": bool(_ENV_DUMP.search(issue)) or len(_VERSION_LINE.findall(issue)) >= 3,
        "issue_tokens": len(tokenize(issue)),
    }


def seed_features(ids: list[str], paths: list[str], bm25: np.ndarray, edges: pl.DataFrame,
                  dense_scores: np.ndarray | None) -> dict:
    order = actions.order(bm25)
    top = order[:K]
    s = minmax(bm25)
    top_ids = [ids[i] for i in top]
    deg = pl.concat([edges.select(id="src", type="type"), edges.select(id="dst", type="type")])
    deg = deg.filter(pl.col("id").is_in(top_ids))
    per = {c: 0 for c in top_ids}
    for c, n in deg.group_by("id").len().iter_rows():
        per[c] = n
    top1 = deg.filter(pl.col("id") == top_ids[0])
    out = {
        "bm25_margin": float(1 - s[order[1]]),
        "bm25_topk_margin": float(1 - s[order[1:K]].mean()),
        "seed_files": len({paths[i] for i in top}),
        "top_degree": per[top_ids[0]],
        **{f"top_degree_{g}": top1.filter(pl.col("type").is_in(list(types))).height
           for g, types in actions.EDGE_GROUPS.items()},
        "topk_degree_mean": float(np.mean(list(per.values()))),
        "bm25_dense_overlap": None,
    }
    if dense_scores is not None:
        out["bm25_dense_overlap"] = len(set(top) & set(actions.order(dense_scores)[:K])) / len(top)
    return out


def task_features(task: dict) -> dict:
    d = PREPARED / task["instance_id"]
    df = pl.read_parquet(d / "chunks.parquet", columns=["id", "path", "name", "bm25"])
    ids, paths = df["id"].to_list(), df["path"].to_list()
    names = {part for n in df["name"].to_list() if n for part in n.split(".")}
    try:
        dense_scores = dense.load_scores(task["instance_id"], ids)
    except FileNotFoundError:
        dense_scores = None
    return {
        "instance_id": task["instance_id"], "repo": task["repo"], "source": task["source"],
        **text_features(task["problem_statement"], sorted(set(paths)), names),
        **seed_features(ids, paths, df["bm25"].to_numpy(), pl.read_parquet(d / "edges.parquet"),
                        dense_scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    tasks = [json.loads(line) for line in open(args.tasks)]
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [task_features(t) for t in tasks if (PREPARED / t["instance_id"] / "seed.json").exists()]
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} tasks -> {out}")
    for key in [k for k in rows[0] if k not in ("instance_id", "repo", "source")]:
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            print(f"  {key:22s} mean {np.mean(vals):8.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
