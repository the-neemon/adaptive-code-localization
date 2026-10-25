"""Stage 1 pipeline: per task, snapshot -> chunks -> BM25 seed -> repository graph.

Writes data/prepared/<instance_id>/
  chunks.parquet  every chunk with its spans and BM25 score (raw and min-max normalised)
  edges.parquet   typed chunk-to-chunk graph edges (src, dst, type)
  seed.json       seed regions at each budget, top-k seed chunk ids, coverage stats

Chunk text is not stored; re-chunking a snapshot from the mirror takes well
under a second and is deterministic.

Usage: python -m acl.prepare [--tasks PATH] [--workers N] [--limit N] [--repo owner/name] [--ids id ...]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from acl import bm25, chunking, graph, snapshot

# Old repo code has invalid escapes like "\*" that Python 3.12+ warns about on
# every parse; the warnings are noise, not parse failures.
warnings.filterwarnings("ignore", category=SyntaxWarning)

OUT = Path("data/prepared")
BUDGETS = (100, 300, 500)
TOP_K = 10


def coverage(task: dict, files: dict[str, str], chunks: list[chunking.Chunk]) -> dict:
    """How much of the core ground truth the index can reach at all.

    A core line is resolvable if its file exists at the pre-fix commit and the
    line number is in range, and indexed if some chunk contains it (so .py and
    not blank). Non-.py files, files that do not exist at the pre-fix commit,
    and line numbers past the end of the file are all unreachable. The last
    case is common: agents run `sed -n 1,240p` on short files and SWE-Explore
    stores the requested range unclipped.
    """
    in_chunk = {(c.path, ln) for c in chunks for s, e in c.spans for ln in range(s, e + 1)}
    core = {(r["path"], ln) for r in task["core_regions"] for ln in range(r["start"], r["end"] + 1)}
    n_lines = {p: len(files[p].splitlines()) for p in {p for p, _ in core} if p in files}
    resolvable = {(p, ln) for p, ln in core if p in n_lines and ln <= n_lines[p]}
    return {"core_lines": len(core), "core_resolvable": len(resolvable),
            "core_indexed": len(core & in_chunk)}


def prepare(task: dict, out_root: Path = OUT, mirrors: Path = snapshot.MIRRORS) -> dict:
    out = out_root / task["instance_id"]
    if (out / "seed.json").exists():
        return json.loads((out / "seed.json").read_text())["stats"]
    t0 = time.time()
    files = snapshot.read_python_files(snapshot.mirror_path(mirrors, task["repo"]), task["base_commit"])
    chunks, failed = chunking.chunk_repo(files)
    assert len({c.id for c in chunks}) == len(chunks), "duplicate chunk ids"
    t_chunk = time.time()

    index = bm25.BM25([bm25.chunk_document(c) for c in chunks])
    raw = index.scores(bm25.tokenize(task["problem_statement"]))
    norm = bm25.minmax(raw)
    order = np.argsort(-raw, kind="stable")
    ranked = [chunks[i] for i in order]
    t_bm25 = time.time()

    g = graph.build_graph(files)
    edges = graph.chunk_edges(g, chunks)
    t_graph = time.time()

    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "id": [c.id for c in chunks],
        "path": [c.path for c in chunks],
        "kind": [c.kind for c in chunks],
        "name": [c.name for c in chunks],
        "starts": [[s for s, _ in c.spans] for c in chunks],
        "ends": [[e for _, e in c.spans] for c in chunks],
        "n_lines": [c.n_lines for c in chunks],
        "bm25": raw,
        "bm25_norm": norm,
    }).write_parquet(out / "chunks.parquet")
    pl.DataFrame(edges, schema=["src", "dst", "type"], orient="row").write_parquet(out / "edges.parquet")

    stats = {
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "n_files": len(files),
        "n_chunks": len(chunks),
        "n_parse_failed": len(failed),
        "n_edges": dict(Counter(t for _, _, t in edges)),
        **coverage(task, files, chunks),
        "secs": {"chunk": round(t_chunk - t0, 2), "bm25": round(t_bm25 - t_chunk, 2),
                 "graph": round(t_graph - t_bm25, 2)},
    }
    seed = {
        "top_k": [c.id for c in ranked[:TOP_K]],
        "regions": {str(b): bm25.fill_budget(ranked, b) for b in BUDGETS},
        "parse_failed": failed,
        "stats": stats,
    }
    # Written last: its presence marks the task as complete for --resume.
    (out / "seed.json").write_text(json.dumps(seed, indent=1))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repo", action="append")
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open(args.tasks)]
    if args.repo:
        tasks = [t for t in tasks if t["repo"] in args.repo]
    if args.ids:
        tasks = [t for t in tasks if t["instance_id"] in set(args.ids)]
    tasks = tasks[: args.limit]
    # Not multiprocessing.Pool: if a worker is killed (e.g. out of memory) Pool
    # waits forever for its result, while this raises BrokenProcessPool.
    with ProcessPoolExecutor(args.workers) as pool:
        for s in pool.map(prepare, tasks):
            print(f"{s['instance_id']}: {s['n_chunks']} chunks, edges {s['n_edges']}, "
                  f"core indexed {s['core_indexed']}/{s['core_lines']}, secs {s['secs']}", flush=True)


if __name__ == "__main__":
    main()
